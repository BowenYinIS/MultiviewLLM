import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from model import TimeSeriesTransformer, create_model

class ContrastiveTimeSeriesDataset(Dataset):
    """Dataset for contrastive learning with time series and text prompts"""
    
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading data from {jsonl_file}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} samples")
    
    def create_chinese_prompt(self, sample):
        """Create Chinese prompt from time series data"""
        time_series = sample.get('time_series', {})
        txn_des_series = sample.get('txn_des_series', [])
        mcc_desc_series = sample.get('mcc_desc_series', [])
        
        # Extract transaction information
        txn_amts = time_series.get('txn_amt', [])
        txn_dates = []
        
        # Create transaction descriptions
        transactions = []
        for i in range(len(txn_des_series)):
            if i < len(txn_amts):
                amt = txn_amts[i]
                desc = txn_des_series[i] if i < len(txn_des_series) else "未知交易"
                mcc_desc = mcc_desc_series[i] if i < len(mcc_desc_series) else "未知类别"
                
                # TODO: add time 
                transactions.append(f"交易金额: {amt}元, 描述: {desc}, 类别: {mcc_desc}")
        
        # Create Chinese prompt
        prompt = f"用户交易记录分析:\n"
        prompt += f"总交易次数: {len(transactions)}\n"
        prompt += f"交易详情:\n"
        
        for i, txn in enumerate(transactions[:10]):  # Limit to first 10 transactions
            prompt += f"{i+1}. {txn}\n"
        
        if len(transactions) > 10:
            prompt += f"... 还有{len(transactions)-10}笔交易\n"
        
        prompt += f"请分析该用户的消费行为和信用风险。"
        
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Create Chinese prompt
        prompt = self.create_chinese_prompt(sample)
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract time series features
        time_series = sample.get('time_series', {})
        
        # Prepare time series data (similar to original model input)
        mcc_codes = time_series.get('mcc_cde', [])
        amounts = time_series.get('txn_amt', [])
        hod = time_series.get('hod', [])
        dow = time_series.get('dow', [])
        wom = time_series.get('wom', [])
        moy = time_series.get('moy', [])
        
        # Find the maximum length among all sequences
        max_len = max(len(mcc_codes), len(amounts), len(hod), len(dow), len(wom), len(moy))
        
        if max_len == 0:
            max_len = 1
        
        # Pad all sequences to the same length
        def pad_sequence(seq, max_len, pad_value=0):
            if len(seq) < max_len:
                return seq + [pad_value] * (max_len - len(seq))
            return seq[:max_len]
        
        
        mcc_codes = pad_sequence(mcc_codes, max_len, 0)
        amounts = pad_sequence(amounts, max_len, 0.0)
        hod = pad_sequence(hod, max_len, 0)
        dow = pad_sequence(dow, max_len, 0)
        wom = pad_sequence(wom, max_len, 0)
        moy = pad_sequence(moy, max_len, 0)
        
        return {
            'prompt_input_ids': prompt_tokens['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'mcc_codes': torch.tensor(mcc_codes, dtype=torch.long),
            'amounts': torch.tensor(amounts, dtype=torch.float),
            'hod': torch.tensor(hod, dtype=torch.long),
            'dow': torch.tensor(dow, dtype=torch.long),
            'wom': torch.tensor(wom, dtype=torch.long),
            'moy': torch.tensor(moy, dtype=torch.long),
            'target_delinquency': torch.tensor(sample.get('target_delinquency', 0), dtype=torch.long),
            'sequence_length': max_len  # Store the actual sequence length
        }

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences without padding"""
    # Separate different types of data
    prompt_input_ids = []
    prompt_attention_masks = []
    batch_data_list = []
    target_delinquencies = []
    
    for item in batch:
        prompt_input_ids.append(item['prompt_input_ids'])
        prompt_attention_masks.append(item['prompt_attention_mask'])
        target_delinquencies.append(item['target_delinquency'])
        
        # Store raw sequences for individual processing
        batch_data_list.append({
            'mcc_cde': item['mcc_codes'],
            'hod': item['hod'],
            'dow': item['dow'],
            'wom': item['wom'],
            'moy': item['moy'],
            'txn_amt': item['amounts'],
            'target': torch.tensor(0)  # Dummy target
        })
    
    return {
        'prompt_input_ids': torch.stack(prompt_input_ids),
        'prompt_attention_mask': torch.stack(prompt_attention_masks),
        'batch_data_list': batch_data_list,
        'target_delinquency': torch.stack(target_delinquencies)
    }

class ContrastiveTrainer:
    """Trainer for contrastive learning between time series and text"""
    
    def __init__(self, ts_model, bert_model, device, learning_rate=1e-4, temperature=0.07):
        self.ts_model = ts_model
        self.bert_model = bert_model
        self.device = device
        self.temperature = temperature
        
        # Freeze BERT model - only train TS encoder and projector
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # Projection layer (MLP) for time series embeddings
        self.ts_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768)  # Final BERT dimension
        )
        
        # Only optimize TS model and projector
        self.optimizer = optim.AdamW([
            {'params': self.ts_model.parameters()},
            {'params': self.ts_projection.parameters()}
        ], lr=learning_rate, eps=1e-8, weight_decay=1e-4)
        
        # Initialize projector weights properly
        self._init_projection_weights()
        
        # Training metrics
        self.train_losses = []
        self.epoch_losses = []  # Store loss for each epoch
        self.batch_losses = []  # Store loss for each batch (for detailed plotting)
        self.best_loss = float('inf')  # Track best loss for checkpoint saving
    
    def _init_projection_weights(self):
        """Initialize projection weights properly"""
        for module in self.ts_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
    def forward_ts_model(self, batch):
        """Forward pass through time series model with individual processing and mean pooling"""
        batch_data_list = batch['batch_data_list']
        batch_size = len(batch_data_list)
        
        # Process each sequence individually to avoid padding issues
        pooled_embeddings = []
        
        for i, sample_data in enumerate(batch_data_list):
            # Move individual sample to device
            sample_data_device = {}
            for key, value in sample_data.items():
                if isinstance(value, torch.Tensor):
                    sample_data_device[key] = value.to(self.device)
                else:
                    sample_data_device[key] = value
            
            
            # Create single-sample batch for model
            single_batch = {
                'mcc_cde': [sample_data_device['mcc_cde']],
                'hod': [sample_data_device['hod']],
                'dow': [sample_data_device['dow']],
                'wom': [sample_data_device['wom']],
                'moy': [sample_data_device['moy']],
                'txn_amt': [sample_data_device['txn_amt']],
                'target': [sample_data_device['target']]
            }
            
            # Forward pass for single sample
            with torch.no_grad():  # TS model is frozen
                outputs = self.ts_model(single_batch)
                token_embeddings = outputs['token_embeddings']  # [1, seq_len, hidden_size]
                attention_mask = outputs['attention_mask']  # [1, seq_len]
            
            
            # Simple mean pooling over valid tokens
            valid_mask = ~attention_mask  # Convert from True=ignore to True=valid
            if valid_mask.sum() > 0:
                # Mean pooling over valid tokens only
                valid_embeddings = token_embeddings[valid_mask]  # [num_valid_tokens, hidden_size]
                pooled_embedding = valid_embeddings.mean(dim=0)  # [hidden_size]
            else:
                # Fallback: mean over all tokens
                pooled_embedding = token_embeddings.mean(dim=1).squeeze(0)  # [hidden_size]
            
            
            pooled_embeddings.append(pooled_embedding)
        
        # Stack all pooled embeddings
        pooled_output = torch.stack(pooled_embeddings)  # [batch_size, hidden_size]
        
        return pooled_output
    
    def forward_bert_model(self, batch):
        """Forward pass through BERT model (frozen)"""
        input_ids = batch['prompt_input_ids'].to(self.device)
        attention_mask = batch['prompt_attention_mask'].to(self.device)
        
        with torch.no_grad():  # BERT is frozen
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use [CLS] token representation
            pooled_output = outputs.pooler_output # TODO: do mean pooling? 
        
        return pooled_output
    
    def compute_infonce_loss(self, ts_embeddings, text_embeddings):
        """Compute InfoNCE loss for contrastive learning"""
        batch_size = ts_embeddings.shape[0]
        
        # Project time series embeddings through MLP to match BERT dimension
        ts_proj = self.ts_projection(ts_embeddings)  # [batch_size, 768]
        
        # Normalize embeddings
        ts_proj = torch.nn.functional.normalize(ts_proj, dim=1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)
        
        # Compute similarity matrix with temperature scaling
        similarity_matrix = torch.matmul(ts_proj, text_embeddings.T) / self.temperature
        
        # Create labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(self.device)
        
        # InfoNCE loss: for each time series, its corresponding text is positive
        # All other texts in the batch are negatives
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def train_epoch(self, dataloader, log_interval=100):
        """Train for one epoch"""
        self.ts_model.train()
        self.bert_model.train()
        
        total_loss = 0.0
        num_batches = 0
        epoch_batch_losses = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            self.optimizer.zero_grad()
            
            # Forward pass through both models
            ts_embeddings = self.forward_ts_model(batch)
            text_embeddings = self.forward_bert_model(batch)
            
            # Compute InfoNCE loss
            loss = self.compute_infonce_loss(ts_embeddings, text_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.ts_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.ts_projection.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            epoch_batch_losses.append(batch_loss)
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f"Batch {batch_idx + 1}, Loss: {batch_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.epoch_losses.append(avg_loss)
        self.batch_losses.extend(epoch_batch_losses)
        
        return avg_loss
    
    def save_best_checkpoint(self, output_dir, epoch, current_loss):
        """Save checkpoint only if current loss is the best so far"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            
            # Remove previous best checkpoint if it exists
            best_checkpoint_path = os.path.join(output_dir, 'best_contrastive_checkpoint.pt')
            if os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            
            # Save new best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.ts_model.state_dict(),  # 使用原始键名
                'ts_projection_state_dict': self.ts_projection.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': current_loss,
                'best_loss': self.best_loss,
                'epoch_losses': self.epoch_losses
            }, best_checkpoint_path)
            
            print(f"New best checkpoint saved! Loss: {current_loss:.4f} (Previous best: {self.best_loss:.4f})")
            return True
        else:
            print(f"Current loss: {current_loss:.4f} (Best: {self.best_loss:.4f}) - No checkpoint saved")
            return False
    
    def plot_loss_curves(self, output_dir, epoch):
        """Plot InfoNCE loss curves"""
        # Create output directory for figures
        fig_dir = os.path.join('Fig', 'TSContrastive')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot epoch-level loss curve
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Epoch loss curve
        plt.subplot(2, 1, 1)
        epochs = range(1, len(self.epoch_losses) + 1)
        plt.plot(epochs, self.epoch_losses, 'b-', linewidth=2, marker='o', markersize=6)
        plt.title('InfoNCE Loss Curve - Epoch Level', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average InfoNCE Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(['InfoNCE Loss'])
        
        # Add value annotations on the plot
        for i, loss in enumerate(self.epoch_losses):
            plt.annotate(f'{loss:.4f}', (i+1, loss), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        # Subplot 2: Batch-level loss curve (if available)
        if len(self.batch_losses) > 0:
            plt.subplot(2, 1, 2)
            batches = range(1, len(self.batch_losses) + 1)
            plt.plot(batches, self.batch_losses, 'r-', linewidth=1, alpha=0.7)
            plt.title('InfoNCE Loss Curve - Batch Level', fontsize=14, fontweight='bold')
            plt.xlabel('Batch', fontsize=12)
            plt.ylabel('InfoNCE Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(['Batch Loss'])
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(fig_dir, f'infonce_loss_curves_epoch_{epoch}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curves saved to: {plot_path}")
        
        # Also save a final plot at the end
        if epoch == len(self.epoch_losses):
            final_plot_path = os.path.join(fig_dir, 'infonce_loss_curves_final.png')
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.epoch_losses, 'b-', linewidth=2, marker='o', markersize=8)
            plt.title('InfoNCE Loss Curve - Final', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Average InfoNCE Loss', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(['InfoNCE Loss'], fontsize=12)
            
            # Add final statistics
            min_loss = min(self.epoch_losses)
            final_loss = self.epoch_losses[-1]
            plt.text(0.02, 0.98, f'Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Final loss curve saved to: {final_plot_path}")

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Checkpoint loaded successfully")
    return model

def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning for Time Series Transformer')
    
    # Data arguments
    parser.add_argument('--data_file', type=str, 
                       default='data/processed_data/ts_processed_data/samples_min12mo_fixed_2test.jsonl',
                       help='Path to training data JSONL file')
    
    # Model arguments
    parser.add_argument('--ts_checkpoint', type=str,
                       default='checkpoint/MultiviewLLM/TSModel/checkpoint_best_samples_min12mo_fixed_2test.pt',
                       help='Path to time series model checkpoint')
    parser.add_argument('--bert_model', type=str,
                       default='bert-base-chinese',
                       help='Chinese BERT model name from HuggingFace')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (larger batch = more negatives)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate for TS encoder and projector')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature parameter for InfoNCE loss')
    
    # Projector arguments
    parser.add_argument('--projector_hidden_dim', type=int, default=512,
                       help='Hidden dimension in projector MLP')
    parser.add_argument('--projector_output_dim', type=int, default=128,
                       help='Output dimension of projector')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                       default='checkpoint/MultiviewLLM/TSModel/contrastive',
                       help='Directory to save contrastive learning checkpoints')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Log training progress every N batches')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Chinese BERT model
    print(f"Loading Chinese BERT model: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model)
    
    # Create time series model
    print("Creating time series model...")
    ts_model = create_model()
    
    # Load time series model checkpoint
    ts_model = load_checkpoint(ts_model, args.ts_checkpoint)
    
    # Set TS model to eval mode to prevent dropout/batch norm issues
    ts_model.eval()
    
    # Move models to device
    ts_model = ts_model.to(device)
    bert_model = bert_model.to(device)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = ContrastiveTimeSeriesDataset(args.data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create trainer with arguments
    trainer = ContrastiveTrainer(
        ts_model, 
        bert_model, 
        device, 
        learning_rate=args.learning_rate,
        temperature=args.temperature
    )
    
    # Update projector dimensions based on arguments
    trainer.ts_projection = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, args.projector_hidden_dim),
        nn.ReLU(),
        nn.Linear(args.projector_hidden_dim, 768)  # Match BERT dimension
    ).to(device)
    
    # Update optimizer with new projector
    trainer.optimizer = optim.AdamW([
        {'params': trainer.ts_model.parameters()},
        {'params': trainer.ts_projection.parameters()}
    ], lr=args.learning_rate)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("Starting contrastive learning...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        avg_loss = trainer.train_epoch(dataloader, args.log_interval)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Plot loss curves
        trainer.plot_loss_curves(args.output_dir, epoch + 1)
        
        # Save best checkpoint only
        trainer.save_best_checkpoint(args.output_dir, epoch + 1, avg_loss)
    
    print("Contrastive learning completed!")

if __name__ == '__main__':
    main()
