import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

from model import create_model, apply_masking, mcc_to_class
from dataloader import create_dataloader, MultiviewTimeSeriesDataset, custom_collate_fn


class CombinedTrainer:
    """Combine masked prediction losses and token-level contrastive loss"""

    def __init__(self, ts_model, bert_model, tokenizer, device,
                 learning_rate=1e-4, weight_decay=1e-5, temperature=0.07,
                 contrastive_weight=1.0, amt_mean=None, amt_std=None):
        self.ts_model = ts_model
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.amt_mean = amt_mean
        self.amt_std = amt_std

        # Freeze BERT (pretrained) for contrastive targets
        for p in self.bert_model.parameters():
            p.requires_grad = False

        # Project TS token embeddings to BERT hidden dim (768)
        self.ts_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        ).to(device)

        # Optimizer: TS model + projection
        self.optimizer = optim.AdamW([
            {'params': self.ts_model.parameters()},
            {'params': self.ts_projection.parameters()}
        ], lr=learning_rate, weight_decay=weight_decay)

        # Loss functions for prediction tasks
        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.amt_criterion = nn.MSELoss()

        # Trackers
        self.epoch_mcc_losses = []
        self.epoch_amt_losses = []
        self.epoch_contrastive_losses = []

    def _move_batch_to_device(self, batch):
        batch_on_device = {}
        for key, value in batch.items():
            if key in ['mcc_cde', 'hod', 'dow', 'wom', 'moy', 'txn_amt', 'target', 'billing_cycle_id', 'year']:
                if key == 'target':
                    batch_on_device[key] = value.to(self.device)
                else:
                    # Move raw tensors to device
                    moved = [tensor.to(self.device) for tensor in value]
                    batch_on_device[key] = moved
                    # Additionally, for MCC codes create a converted class tensor (13 coarse classes)
                    if key == 'mcc_cde':
                        mcc_class_list = []
                        for tensor in moved:
                            # tensor is 1D of raw MCC codes; map to class per element
                            try:
                                classes = [mcc_to_class(int(x.item())) for x in tensor]
                                class_tensor = torch.tensor([c if c >= 0 else -1 for c in classes], dtype=torch.long, device=self.device)
                            except Exception:
                                # Fallback: create a -1 mask
                                class_tensor = torch.full_like(tensor, -1, dtype=torch.long, device=self.device)
                            mcc_class_list.append(class_tensor)
                        batch_on_device['mcc_class'] = mcc_class_list
            else:
                batch_on_device[key] = value
        return batch_on_device

    def compute_prediction_loss(self, predictions, targets, mask_positions, attention_mask):
        """Copy of masked prediction loss calculation from trainer.py"""
        batch_size = predictions['mcc_predictions'].shape[0]
        total_mcc_loss = 0.0
        total_amt_loss = 0.0
        valid_samples = 0

        for i in range(batch_size):
            seq_len = (~attention_mask[i]).sum().item()
            if seq_len == 0:
                continue

            # mask_positions can be a list per-sample, a single list (for batch_size==1),
            # or in some edge cases an int/tensor. Normalize to a python list of ints.
            if isinstance(mask_positions, list) and len(mask_positions) > i:
                sample_mask_positions = mask_positions[i]
            else:
                sample_mask_positions = mask_positions

            # Normalize different types to list
            if sample_mask_positions is None:
                sample_mask_positions = []
            elif isinstance(sample_mask_positions, torch.Tensor):
                sample_mask_positions = sample_mask_positions.tolist()
            elif isinstance(sample_mask_positions, int):
                sample_mask_positions = [int(sample_mask_positions)]

            if len(sample_mask_positions) == 0:
                continue

            valid_mask_positions = [pos for pos in sample_mask_positions if pos < seq_len]
            if len(valid_mask_positions) == 0:
                continue

            # MCC targets
            mcc_targets = []
            for pos in valid_mask_positions:
                # Prefer already-converted class labels if available
                if 'mcc_class' in targets:
                    try:
                        mcc_targets.append(int(targets['mcc_class'][i][pos].item()))
                    except Exception:
                        mcc_targets.append(-1)
                else:
                    mcc_code = targets['mcc_cde'][i][pos].item()
                    mcc_class = mcc_to_class(mcc_code)
                    mcc_targets.append(mcc_class)

            valid_mcc_positions = []
            valid_mcc_targets = []
            for j, pos in enumerate(valid_mask_positions):
                if mcc_targets[j] != -1:
                    valid_mcc_positions.append(pos)
                    valid_mcc_targets.append(mcc_targets[j])

            if len(valid_mcc_positions) > 0:
                mcc_pred = predictions['mcc_predictions'][i][valid_mcc_positions]
                mcc_target_tensor = torch.tensor(valid_mcc_targets, dtype=torch.long, device=mcc_pred.device)
                mcc_loss = self.mcc_criterion(mcc_pred, mcc_target_tensor)
                total_mcc_loss += mcc_loss

            amt_pred = predictions['amt_predictions'][i][valid_mask_positions]
            amt_target = targets['txn_amt'][i][valid_mask_positions]
            amt_loss = self.amt_criterion(amt_pred, amt_target)

            total_amt_loss += amt_loss
            valid_samples += 1

        avg_mcc_loss = total_mcc_loss / max(valid_samples, 1)
        avg_amt_loss = total_amt_loss / max(valid_samples, 1)
        total_loss = avg_mcc_loss + avg_amt_loss

        return {
            'total_pred_loss': total_loss,
            'mcc_loss': avg_mcc_loss,
            'amt_loss': avg_amt_loss
        }

    def build_prompts_for_tokens(self, batch_on_device, seq_lengths):
        """Build Chinese prompts per transaction token using provided format"""
        prompts = []
        missing_mcc_desc_reported = False
        # Iterate samples
        for i in range(len(batch_on_device['mcc_cde'])):
            seq_len = seq_lengths[i].item() if isinstance(seq_lengths, torch.Tensor) else seq_lengths[i]
            for pos in range(seq_len):
                # Time: prefer a full datetime if available in the batch (several possible keys)
                time_val = None
                # Possible keys that may contain full datetime strings per-sample
                for key in ['txn_dte_series', 'txn_dte_tme_series', 'txn_dt_series', 'txn_datetime_series', 'txn_dte_tme', 'txn_datetime']:
                    if key in batch_on_device:
                        try:
                            candidate = batch_on_device[key][i]
                            # candidate may be a list of strings per token
                            if isinstance(candidate, (list, tuple)) and len(candidate) > pos:
                                time_val = str(candidate[pos])
                                break
                        except Exception:
                            pass

                # Fallback: construct datetime from year, month, hour if no full datetime available
                if time_val is None:
                    try:
                        year = batch_on_device['year'][i][pos].item() if 'year' in batch_on_device else 2012
                        moy = batch_on_device['moy'][i][pos].item()
                        hod = batch_on_device['hod'][i][pos].item()
                        time_val = f"{int(year):04d}-{int(moy):02d}-01 {int(hod):02d}:00:00"
                    except Exception:
                        time_val = '未知时间'

                # MCC description: prefer human-readable Chinese description if provided
                mcc_desc = None
                if 'mcc_desc_series' in batch_on_device:
                    try:
                        desc_list = batch_on_device['mcc_desc_series'][i]
                        if isinstance(desc_list, (list, tuple)) and len(desc_list) > pos:
                            mcc_desc = desc_list[pos]
                    except Exception:
                        mcc_desc = None

                if not mcc_desc:
                    # fallback to numeric mcc code
                    # Use raw MCC code for prompt if available, otherwise fall back to class label
                    if 'mcc_cde' in batch_on_device:
                        try:
                            mcc_val = str(batch_on_device['mcc_cde'][i][pos].item())
                        except Exception:
                            # fallback to class label
                            try:
                                mcc_val = str(batch_on_device['mcc_class'][i][pos].item())
                            except Exception:
                                mcc_val = '未知MCC'
                    else:
                        try:
                            mcc_val = str(batch_on_device['mcc_class'][i][pos].item())
                        except Exception:
                            mcc_val = '未知MCC'
                    if not missing_mcc_desc_reported:
                        print("Warning: `mcc_desc_series` not found or empty in data; falling back to numeric MCC codes.")
                        missing_mcc_desc_reported = True
                    mcc_display = mcc_val
                else:
                    mcc_display = str(mcc_desc)

                # Transaction description: use txn_des_series if available
                des_val = None
                if 'txn_des_series' in batch_on_device:
                    try:
                        des_list = batch_on_device['txn_des_series'][i]
                        if isinstance(des_list, (list, tuple)) and len(des_list) > pos:
                            des_val = des_list[pos]
                    except Exception:
                        des_val = None

                if not des_val:
                    des_val = '未知交易'

                # Billing cycle ID: extract from batch (0-11)
                billing_cycle_id = '未知账单周期'
                if 'billing_cycle_id' in batch_on_device:
                    try:
                        billing_cycle_id = str(int(batch_on_device['billing_cycle_id'][i][pos].item()))
                    except Exception:
                        pass

                # Amount: reconstruct original amount if possible
                try:
                    amt_val = batch_on_device['txn_amt'][i][pos].item()
                    # If dataset normalized amounts: stored = (log(1+raw) - mean)/std
                    if (self.amt_mean is not None) and (self.amt_std is not None):
                        logged = float(amt_val) * float(self.amt_std) + float(self.amt_mean)
                        raw_amt = float(np.exp(logged) - 1.0)
                    else:
                        # assume stored value is log(1+raw)
                        logged = float(amt_val)
                        raw_amt = float(np.exp(logged) - 1.0)
                    amt_display = f"{raw_amt:.2f}"
                except Exception:
                    amt_display = '未知金额'

                prompt = f"时间：{time_val}，商户类别：{mcc_display}，交易描述：{des_val}，金额{amt_display}元。"
                prompts.append(prompt)

        return prompts

    def compute_token_contrastive_loss(self, token_embeddings, attention_mask, batch_on_device):
        """Compute InfoNCE at token level. Negatives are all tokens in the batch."""
        # token_embeddings: [batch, max_len, d_model]
        # attention_mask: True=ignore
        seq_lengths = (~attention_mask).sum(dim=1)

        # Collect valid token embeddings into list
        ts_emb_list = []
        for i in range(token_embeddings.shape[0]):
            seq_len = seq_lengths[i].item()
            if seq_len > 0:
                valid = token_embeddings[i, :seq_len, :]
                ts_emb_list.append(valid)

        if len(ts_emb_list) == 0:
            return torch.tensor(0.0, device=self.device)

        ts_flat = torch.cat(ts_emb_list, dim=0)  # [N, d_model]

        # Build prompts for each valid token
        prompts = self.build_prompts_for_tokens(batch_on_device, seq_lengths)
        if len(prompts) == 0:
            return torch.tensor(0.0, device=self.device)

        # Encode prompts with BERT (frozen)
        tokenized = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(self.device)
        attention = tokenized['attention_mask'].to(self.device)

        with torch.no_grad():
            bert_out = self.bert_model(input_ids=input_ids, attention_mask=attention)
            # Use mean pooling over valid tokens (use attention mask) instead of pooler_output
            # bert_out.last_hidden_state: [N, seq_len, hidden]
            last_hidden = bert_out.last_hidden_state  # [N, L, D]
            # attention: [N, L] with 1 for real tokens
            att = attention.unsqueeze(-1).to(last_hidden.dtype)  # [N, L, 1]
            summed = (last_hidden * att).sum(dim=1)  # [N, D]
            lengths = att.sum(dim=1)  # [N, 1]
            lengths = lengths.clamp(min=1e-6)
            text_emb = summed / lengths  # [N, D]

        # Project TS embeddings
        ts_proj = self.ts_projection(ts_flat)  # [N, bert_dim]

        # Normalize
        ts_norm = torch.nn.functional.normalize(ts_proj, dim=1)
        text_norm = torch.nn.functional.normalize(text_emb, dim=1)

        # Similarity matrix
        sim = torch.matmul(ts_norm, text_norm.T) / self.temperature  # [N, N]

        labels = torch.arange(sim.shape[0], device=self.device)
        loss = torch.nn.functional.cross_entropy(sim, labels)

        return loss

    def train_epoch(self, dataloader, mask_prob=0.15, log_interval=50):
        self.ts_model.train()
        total_mcc = 0.0
        total_amt = 0.0
        total_contra = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Training')):
            self.optimizer.zero_grad()

            batch_on_device = self._move_batch_to_device(batch)

            # Apply masking
            masked_batch, mask_positions = apply_masking(batch_on_device, mask_prob=mask_prob, device=self.device)

            # Forward pass with predictions
            predictions = self.ts_model.forward_with_predictions(masked_batch, mask_prob)

            # Prediction losses
            pred_losses = self.compute_prediction_loss(predictions, batch_on_device, mask_positions, predictions['attention_mask'])

            # Contrastive loss (token-level)
            # We use token embeddings from the (masked) forward; positives are themselves
            token_embeddings = predictions['token_embeddings']
            attention_mask = predictions['attention_mask']
            contra_loss = self.compute_token_contrastive_loss(token_embeddings, attention_mask, batch_on_device)

            total_loss = pred_losses['total_pred_loss'] + self.contrastive_weight * contra_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ts_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.ts_projection.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_mcc = pred_losses['mcc_loss'].item() if isinstance(pred_losses['mcc_loss'], torch.Tensor) else float(pred_losses['mcc_loss'])
            batch_amt = pred_losses['amt_loss'].item() if isinstance(pred_losses['amt_loss'], torch.Tensor) else float(pred_losses['amt_loss'])
            batch_contra = contra_loss.item() if isinstance(contra_loss, torch.Tensor) else float(contra_loss)

            total_mcc += batch_mcc
            total_amt += batch_amt
            total_contra += batch_contra
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(f"Batch {batch_idx+1}: mcc={batch_mcc:.4f}, amt={batch_amt:.4f}, contra={batch_contra:.4f}")

        avg_mcc = total_mcc / max(num_batches, 1)
        avg_amt = total_amt / max(num_batches, 1)
        avg_contra = total_contra / max(num_batches, 1)

        self.epoch_mcc_losses.append(avg_mcc)
        self.epoch_amt_losses.append(avg_amt)
        self.epoch_contrastive_losses.append(avg_contra)

        return {
            'mcc_loss': avg_mcc,
            'amt_loss': avg_amt,
            'contrastive_loss': avg_contra
        }

    def save_embeddings_npy(self, dataloader, output_path):
        """Compute token embeddings for all samples in dataloader and save as npy dict"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.ts_model.eval()
        embeddings = {}
        idx = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Embedding Save'):
                batch_on_device = self._move_batch_to_device(batch)
                out = self.ts_model(batch_on_device)
                token_embeddings = out['token_embeddings']  # [B, L, d]
                seq_lengths = (~out['attention_mask']).sum(dim=1)

                for i in range(token_embeddings.shape[0]):
                    seq_len = seq_lengths[i].item()
                    emb = token_embeddings[i, :seq_len, :].cpu().numpy()
                    embeddings[idx] = emb
                    idx += 1

        np.save(output_path, embeddings)
        print(f"Saved embeddings to {output_path}")

    def plot_losses(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        epochs = range(1, len(self.epoch_mcc_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.epoch_mcc_losses, label='MCC Loss', color='tab:blue')
        plt.plot(epochs, self.epoch_amt_losses, label='Amount Loss', color='tab:green')
        plt.plot(epochs, self.epoch_contrastive_losses, label='Contrastive Loss', color='tab:red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(save_dir, 'combined_training_losses.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined loss plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Combined training: prediction + token-level contrastive')
    parser.add_argument('--train_file', type=str,
                        default='data/processed_data/ts_processed_data/samples_min12mo_fixed_2test_billingcycle.jsonl')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='checkpoint/MultiviewLLM/TSModel/combined')
    parser.add_argument('--plot_dir', type=str, default='Fig/TScombined')
    parser.add_argument('--npy_path', type=str, default='checkpoint/MultiviewLLM/TSModel/samples_combined.npy')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--limit_samples', type=int, default=-1,
                        help='Randomly sample this many sequences for fast training (-1 for all)')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for sampling indices')
    args = parser.parse_args()

    # Normalize device string: prefer explicit cuda index (e.g. 'cuda:0', 'cuda:1')
    requested_device = args.device
    if torch.cuda.is_available():
        if requested_device == 'cuda':
            # make explicit cuda:0 to avoid ambiguity
            requested_device = 'cuda:0'
        # If user provided 'cuda' or 'cuda:<idx>' keep it; otherwise allow 'cpu'
        if requested_device.startswith('cuda'):
            # extract index if provided
            if ':' in requested_device:
                try:
                    dev_idx = int(requested_device.split(':', 1)[1])
                except Exception:
                    dev_idx = 0
            else:
                dev_idx = 0
            # set current CUDA device to keep .cuda() calls consistent
            try:
                torch.cuda.set_device(dev_idx)
            except Exception:
                pass
    device = torch.device(requested_device if torch.cuda.is_available() or requested_device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and BERT
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)

    # Create time series model
    # Load training data first to compute amount statistics
    stats_dataset = MultiviewTimeSeriesDataset(args.train_file)
    
    # MCC is converted to 13 coarse classes in _move_batch_to_device, so num_mcc is fixed at 13
    # Other categorical features use their natural ranges
    # billing_cycle_id has 12 unique values (0-11)
    num_mcc = 13
    num_hod = 24
    num_dow = 7
    num_wom = 6
    num_moy = 12
    billing_cycle_classes = 12
    ts_model = create_model(device=device, num_mcc=num_mcc, num_hod=num_hod, num_dow=num_dow, num_wom=num_wom, num_moy=num_moy, billing_cycle_classes=billing_cycle_classes)
    ts_model = ts_model.to(device)

    # Compute amount normalization stats from training data (used to invert normalization in prompts)
    print("Computing amount statistics from training data...")
    amt_mean, amt_std = stats_dataset.compute_amount_stats()

    # Create dataset with normalization (so model input matches training setup)
    full_dataset = MultiviewTimeSeriesDataset(args.train_file, amt_mean=amt_mean, amt_std=amt_std)
    # Optionally sample a subset for fast training
    if args.limit_samples is not None and args.limit_samples > 0:
        total = len(full_dataset)
        k = min(args.limit_samples, total)
        rng = np.random.RandomState(args.sample_seed)
        indices = rng.choice(np.arange(total), size=k, replace=False)
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, indices.tolist())
        print(f"Using a random subset of {k} / {total} samples for fast training.")
    else:
        train_dataset = full_dataset

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    trainer = CombinedTrainer(ts_model, bert_model, tokenizer, device,
                              learning_rate=args.lr, temperature=args.temperature,
                              contrastive_weight=args.contrastive_weight,
                              amt_mean=amt_mean, amt_std=amt_std)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting combined training...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        losses = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1} - MCC: {losses['mcc_loss']:.4f}, AMT: {losses['amt_loss']:.4f}, CONTRA: {losses['contrastive_loss']:.4f}")

        # Save embeddings (npy) after each epoch following contrastive trainer's save logic
        npy_out = os.path.join(args.output_dir, f'samples_epoch_{epoch+1}.npy')
        trainer.save_embeddings_npy(dataloader, npy_out)

        # Save model checkpoint each epoch
        ckpt_dir = os.path.join(args.output_dir, 'modelckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': trainer.ts_model.state_dict(),
            'ts_projection_state_dict': trainer.ts_projection.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'losses': losses,
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        # Free cached GPU memory to avoid allocator growth across epochs
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass

    # After training, plot losses
    trainer.plot_losses(args.plot_dir)
    # Also save final combined npy at specified path
    trainer.save_embeddings_npy(dataloader, args.npy_path)
    # Final cache cleanup
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("Combined training completed")


if __name__ == '__main__':
    main()
