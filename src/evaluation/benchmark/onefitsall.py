"""
One-Fits-All Baseline: Credit Risk Prediction with PatchTST + Qwen2.5 7B

Pipeline:
1. Load time series data (JSONL format)
2. One-hot encode 7 features (hod, dow, wom, moy, billingcycleid, amt, mcc_class)
3. Apply instance norm (optional)
4. Patch the time series [batch, 7, seq_len] -> [batch, 7, num_patches, patch_size]
5. PatchTST encoder: linear projection Wp*xp + position encoding
6. Feed to Qwen2.5 7B instruct model
7. Add MLP output projector for binary delinquency prediction
8. Train with frozen LLM (only layer norms trainable)
9. Evaluate on train/test split
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import argparse
import warnings
import csv
from hmeasure import h_score

# Set Hugging Face cache directory
os.environ['HF_HOME'] = '/data/mjmao/ood/hf_models'

def get_env_cache_dir() -> str:
    env_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    return env_dir if env_dir else "/data/mjmao/ood/hf_models"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")

warnings.filterwarnings('ignore')


# ==================== Feature Encoding ====================

def mcc_to_class(mcc_code):
    """
    Convert MCC code to class (0-12)
    Classes: [0001-1499, 1500-2999, 3000-3299, 3300-3499, 3500-3999, 4000-4799,
              4800-4999, 5000-5599, 5600-5699, 5700-7299, 7300-7999, 8000-8999, 9000-9999]
    """
    if mcc_code < 0:
        return -1
    if 1 <= mcc_code <= 1499:
        return 0
    elif 1500 <= mcc_code <= 2999:
        return 1
    elif 3000 <= mcc_code <= 3299:
        return 2
    elif 3300 <= mcc_code <= 3499:
        return 3
    elif 3500 <= mcc_code <= 3999:
        return 4
    elif 4000 <= mcc_code <= 4799:
        return 5
    elif 4800 <= mcc_code <= 4999:
        return 6
    elif 5000 <= mcc_code <= 5599:
        return 7
    elif 5600 <= mcc_code <= 5699:
        return 8
    elif 5700 <= mcc_code <= 7299:
        return 9
    elif 7300 <= mcc_code <= 7999:
        return 10
    elif 8000 <= mcc_code <= 8999:
        return 11
    elif 9000 <= mcc_code <= 9999:
        return 12
    else:
        return 12


class TimeSeriesDataset(Dataset):
    """
    Load time series data from JSONL files.
    Features: hod (24), dow (7), wom (6), moy (12), billingcycleid (12), amt (1), mcc_class (13)
    Total: 7 channels
    """
    
    def __init__(self, file_path, amt_mean=None, amt_std=None):
        self.data = []
        self.amt_mean = amt_mean
        self.amt_std = amt_std
        self.split_type = None
        self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load JSONL data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                self.data.append(record)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        time_series = record['time_series']
        
        # Extract raw features
        seq_len = len(time_series['mcc_cde'])
        
        # hod: 0-23 (24 classes)
        hod = torch.tensor(time_series['hod'], dtype=torch.long)
        hod_onehot = F.one_hot(hod.clamp(min=0, max=23), num_classes=24).float()
        
        # dow: 0-6 (7 classes)
        dow = torch.tensor(time_series['dow'], dtype=torch.long)
        dow_onehot = F.one_hot(dow.clamp(min=0, max=6), num_classes=7).float()
        
        # wom: 0-5 (6 classes, weeks of month)
        wom = torch.tensor(time_series['wom'], dtype=torch.long)
        wom_onehot = F.one_hot(wom.clamp(min=0, max=5), num_classes=6).float()
        
        # moy: 0-11 (12 classes, months of year)
        moy = torch.tensor(time_series['moy'], dtype=torch.long)
        moy_onehot = F.one_hot(moy.clamp(min=0, max=11), num_classes=12).float()
        
        # billing_cycle_id: 0-11 (12 classes)
        billing_cycle_id = torch.tensor(time_series.get('billing_cycle_id', [-1]*seq_len), dtype=torch.long)
        billing_cycle_onehot = F.one_hot(billing_cycle_id.clamp(min=0, max=11), num_classes=12).float()
        
        # mcc_cde: map to mcc_class (0-12, 13 classes)
        mcc_cde = torch.tensor(time_series['mcc_cde'], dtype=torch.long)
        mcc_classes = torch.tensor([mcc_to_class(int(x)) for x in mcc_cde], dtype=torch.long)
        mcc_onehot = F.one_hot(mcc_classes.clamp(min=0, max=12), num_classes=13).float()
        
        # amt: continuous, log(1+x) normalized
        raw_amt = torch.tensor(time_series['txn_amt'], dtype=torch.float32)
        amt_log = torch.log(1 + raw_amt)
        if self.amt_mean is not None and self.amt_std is not None:
            amt_normalized = (amt_log - self.amt_mean) / self.amt_std
        else:
            amt_normalized = amt_log
        
        # Stack: [seq_len, 24 + 7 + 6 + 12 + 12 + 13 + 1] = [seq_len, 75]
        # But we'll concatenate along feature dimension for compatibility
        features = torch.cat([
            hod_onehot,           # 24
            dow_onehot,           # 7
            wom_onehot,           # 6
            moy_onehot,           # 12
            billing_cycle_onehot, # 12
            mcc_onehot,           # 13
            amt_normalized.unsqueeze(-1)  # 1
        ], dim=1)  # [seq_len, 75]
        
        # Target: binary delinquency label
        target = torch.tensor(record['target_delinquency'], dtype=torch.long)
        
        # Split: train or test
        split = record.get('split', 'train')
        
        return {
            'features': features,  # [seq_len, 75]
            'target': target,      # scalar
            'split': split,        # 'train' or 'test'
            'act_idn_sky': record['act_idn_sky'],
            'seq_len': seq_len
        }
    
    def compute_amount_stats(self):
        """Compute statistics of log-transformed amounts"""
        all_amounts = []
        for record in self.data:
            time_series = record['time_series']
            raw_amounts = torch.tensor(time_series['txn_amt'], dtype=torch.float32)
            log_amounts = torch.log(1 + raw_amounts)
            all_amounts.extend(log_amounts.tolist())
        
        all_amounts = torch.tensor(all_amounts)
        mean = all_amounts.mean().item()
        std = all_amounts.std().item()
        print(f"Amount statistics - Mean: {mean:.4f}, Std: {std:.4f}")
        return mean, std


def custom_collate_fn(batch):
    """Custom collate function for variable length sequences"""
    features = [item['features'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    splits = [item['split'] for item in batch]
    acts = [item['act_idn_sky'] for item in batch]
    seq_lens = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    
    # Pad sequences to max length in batch
    max_len = max([f.shape[0] for f in features])
    padded_features = []
    masks = []
    
    for f in features:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], f.shape[1])
            f_padded = torch.cat([f, padding], dim=0)
            mask = torch.cat([torch.ones(f.shape[0]), torch.zeros(max_len - f.shape[0])])
        else:
            f_padded = f
            mask = torch.ones(max_len)
        padded_features.append(f_padded)
        masks.append(mask)
    
    features = torch.stack(padded_features)  # [batch, seq_len, 75]
    masks = torch.stack(masks)  # [batch, seq_len]
    
    return {
        'features': features,
        'targets': targets,
        'masks': masks,
        'splits': splits,
        'acts': acts,
        'seq_lens': seq_lens
    }


# ==================== Instance Normalization ====================

class InstanceNorm(nn.Module):
    """Apply instance normalization per sample"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        x: [batch, seq_len, features]
        Apply normalization per sample across sequence dimension
        """
        # Compute mean and std per sample
        mean = x.mean(dim=1, keepdim=True)  # [batch, 1, features]
        std = x.std(dim=1, keepdim=True)    # [batch, 1, features]
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        # Affine transformation
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        
        return x_norm


# ==================== Patching ====================

class PatchEmbedding(nn.Module):
    """
    Patch time series data and project to embedding dimension.
    
    Input: [batch, seq_len, feature_dim]
    1. Reshape to [batch, num_patches, patch_size, feature_dim]
    2. Flatten patches: [batch, num_patches, patch_size * feature_dim]
    3. Project with Wp: [batch, num_patches, d_model]
    4. Add position encoding: [batch, num_patches, d_model]
    """
    
    def __init__(self, patch_size=16, feature_dim=75, d_model=256, max_patches=512):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.patch_dim = patch_size * feature_dim
        
        # Linear projection Wp: [patch_dim] -> [d_model]
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        
        # Position encoding: [max_patches, d_model]
        self.register_buffer('pos_encoding', self._create_position_encoding(max_patches, d_model))
    
    def _create_position_encoding(self, max_patches, d_model):
        """Create learnable position encoding"""
        pe = torch.zeros(max_patches, d_model)
        position = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        """
        x: [batch, seq_len, feature_dim]
        Returns: [batch, num_patches, d_model]
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Pad sequence to be divisible by patch_size
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            # Repeat last value for padding
            last_val = x[:, -1:, :].expand(batch_size, pad_len, feature_dim)
            x = torch.cat([x, last_val], dim=1)
        
        seq_len_padded = x.shape[1]
        num_patches = seq_len_padded // self.patch_size
        
        # Reshape to [batch, num_patches, patch_size, feature_dim]
        x = x.view(batch_size, num_patches, self.patch_size, feature_dim)
        
        # Flatten patches: [batch, num_patches, patch_size * feature_dim]
        x = x.view(batch_size, num_patches, self.patch_dim)
        
        # Project patches: [batch, num_patches, d_model]
        x = self.patch_proj(x)
        
        # Add position encoding
        x = x + self.pos_encoding[:num_patches, :].unsqueeze(0)
        
        return x, num_patches


# ==================== Qwen2.5 7B + Projector ====================

class OutputProjector(nn.Module):
    """MLP for delinquency prediction"""
    
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
    
    def forward(self, x):
        """x: [batch, d_model] or [batch, seq_len, d_model] -> [batch, 2]"""
        if x.dim() == 3:
            # Mean pooling over sequence: [batch, seq_len, d_model] -> [batch, d_model]
            x = x.mean(dim=1)
        return self.mlp(x)


class QwenWithLLMAdapter(nn.Module):
    """
    Qwen2.5 7B LLM with frozen parameters (except layer norms).
    Adapter to inject time series embeddings via prompt/embedding injection.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", d_model=256, 
                 freeze_llm=True, train_ln_only=True):
        super().__init__()
        self.d_model = d_model
        self.freeze_llm = freeze_llm
        self.train_ln_only = train_ln_only
        
        # Load Qwen model
        print(f"Loading {model_name}...")
        cache_dir = get_env_cache_dir()
        # Don't use device_map='auto' to avoid multi-GPU issues with layer norms
        # Instead, load to a single device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.hidden_dim = self.model.config.hidden_size
        
        # Adapter: project time series embeddings to LLM hidden dimension
        self.ts_proj = nn.Linear(d_model, self.hidden_dim)
        
        # Optional: learnable adapter tokens (can be injected into context)
        self.adapter_tokens = nn.Parameter(torch.randn(1, 8, self.hidden_dim) * 0.02)
        
        # Freeze/unfreeze parameters
        if freeze_llm:
            self._setup_frozen_params()
    
    def _setup_frozen_params(self):
        """Freeze all LLM parameters except layer norms"""
        for name, param in self.model.named_parameters():
            if self.train_ln_only:
                # Only train layer norm parameters
                if 'norm' in name.lower():  # LayerNorm or RMSNorm
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
        print("LLM parameters frozen. Training layer norms only.")
    
    def forward(self, input_ids, attention_mask, ts_embeddings=None):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        ts_embeddings: [batch, num_patches, d_model]
        
        Returns: [batch, hidden_dim] (pooled representation)
        """
        # Get embeddings
        with torch.no_grad() if not self.train_ln_only else torch.enable_grad():
            embedding_output = self.model.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden]
        
        # Project and inject time series embeddings
        if ts_embeddings is not None:
            ts_proj = self.ts_proj(ts_embeddings)  # [batch, num_patches, hidden]
            # Concatenate: [batch, num_patches + seq_len, hidden]
            embedding_output = torch.cat([ts_proj, embedding_output], dim=1)
            
            # Extend attention mask
            ts_mask = torch.ones(ts_embeddings.shape[0], ts_embeddings.shape[1], 
                                device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([ts_mask, attention_mask], dim=1)
        
        # Forward through LLM (without language modeling head)
        outputs = self.model(
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state, pooled representation
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len_total, hidden]
        
        # Global average pooling
        pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / \
                 attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        return pooled


class OneForAllModel(nn.Module):
    """Complete pipeline: Patch -> PatchTST -> Qwen2.5 7B -> Output Projector"""
    
    def __init__(self, patch_size=16, d_model=256, qwen_model="Qwen/Qwen2.5-7B-Instruct",
                 feature_dim=75, freeze_llm=True):
        super().__init__()
        
        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(patch_size=patch_size, feature_dim=feature_dim, 
                                          d_model=d_model)
        
        # 2. Optional instance norm
        self.inst_norm = InstanceNorm(feature_dim)
        
        # 3. Qwen adapter
        self.qwen_adapter = QwenWithLLMAdapter(model_name=qwen_model, d_model=d_model,
                                               freeze_llm=freeze_llm)
        
        # 4. Output projector
        self.output_proj = OutputProjector(input_dim=self.qwen_adapter.hidden_dim, 
                                          hidden_dim=512, dropout=0.1)
    
    def forward(self, features, attention_mask=None):
        """
        features: [batch, seq_len, feature_dim]
        attention_mask: [batch, seq_len] (1 for valid, 0 for padding)
        Returns: logits [batch, 2]
        """
        # Apply instance norm
        features = self.inst_norm(features)
        
        # Patch embedding
        patch_emb, num_patches = self.patch_embed(features)
        
        # Create simple text prompt (can be enhanced)
        batch_size = features.shape[0]
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        # Simple prompt about transaction patterns
        prompts = ["This customer has the following transaction patterns."] * batch_size
        inputs = tokenizer(prompts, padding=True, return_tensors='pt', truncation=True)
        input_ids = inputs['input_ids'].to(features.device)
        prompt_attn_mask = inputs['attention_mask'].to(features.device)
        
        # Qwen forward
        pooled = self.qwen_adapter(input_ids, prompt_attn_mask, ts_embeddings=patch_emb)
        
        # Output projection
        logits = self.output_proj(pooled)
        
        return logits


# ==================== Training ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights [num_classes]
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        inputs: [batch, num_classes] logits
        targets: [batch] class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, train_loader, optimizer, device, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)
        
        # Forward
        logits = model(features, attention_mask=masks)
        loss = criterion(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, data_loader, device, split_name='eval'):
    """Evaluate model on dataset"""
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Evaluating {split_name}'):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            logits = model(features, attention_mask=masks)
            preds = logits.argmax(dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    # AUC and baseline: use probability of positive class
    probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()[:, 1]
    
    # Baseline using ground-truth proportion: set top-k by probability as 1, others 0
    total_samples = all_targets.shape[0]
    pos_count = int(all_targets.sum())
    proportion = pos_count / max(total_samples, 1)
    k = int(round(proportion * total_samples))
    baseline_preds = np.zeros(total_samples, dtype=int)
    if k > 0:
        topk_idx = np.argpartition(probs, -k)[-k:]
        baseline_preds[topk_idx] = 1
    baseline_precision = precision_score(all_targets, baseline_preds, zero_division=0)
    baseline_recall = recall_score(all_targets, baseline_preds, zero_division=0)
    baseline_f1 = f1_score(all_targets, baseline_preds, zero_division=0)
    auc = roc_auc_score(all_targets, probs)
    
    # Additional probability-based metrics
    # PCC: Pearson Correlation Coefficient between predicted probabilities and targets
    try:
        pcc = float(np.corrcoef(probs, all_targets)[0, 1])
    except Exception:
        pcc = float('nan')
    
    # BS: Brier Score
    bs = float(np.mean((probs - all_targets) ** 2))
    
    # KS: Kolmogorov-Smirnov statistic between positive and negative distributions
    try:
        pos_scores = probs[all_targets == 1]
        neg_scores = probs[all_targets == 0]
        # Compute empirical CDFs over sorted unique thresholds
        thresholds = np.sort(np.unique(probs))
        if thresholds.size == 0:
            ks = 0.0
        else:
            # CDF at each threshold
            pos_cdf = np.searchsorted(np.sort(pos_scores), thresholds, side='right') / max(len(pos_scores), 1)
            neg_cdf = np.searchsorted(np.sort(neg_scores), thresholds, side='right') / max(len(neg_scores), 1)
            ks = float(np.max(np.abs(pos_cdf - neg_cdf)))
    except Exception:
        ks = float('nan')
    
    # PG: Population Gini (commonly defined as 2*AUC - 1)
    pg = 2.0 * float(auc) - 1.0
    
    # H-measure: Not trivial to compute without external package; set to NaN placeholder
    h_measure = h_score(baseline_preds, all_targets)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'baseline_precision': baseline_precision,
        'baseline_recall': baseline_recall,
        'baseline_f1': baseline_f1,
        'auc': auc,
        'preds': all_preds,
        'targets': all_targets,
        'probs': probs,
        'pcc': pcc,
        'bs': bs,
        'ks': ks,
        'pg': pg,
        'h_measure': h_measure
    }


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='One-Fits-All Baseline')
    parser.add_argument('--data_file', type=str, 
                       default='data/processed_data/ts_processed_data/samples_min12mo_fixed_2test.jsonl',
                       help='Path to JSONL data file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='checkpoint/MultiviewLLM/TSModel/onefitsall')
    parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    # parser.add_argument('--qwen_model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs with DataParallel')
    parser.add_argument('--use_focal_loss', default=True, action='store_true', help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--focal_alpha', type=float, default=None, help='Alpha for Focal Loss (None for no weighting)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    dataset = TimeSeriesDataset(args.data_file)
    
    # Compute amount statistics
    amt_mean, amt_std = dataset.compute_amount_stats()
    
    # Recreate dataset with normalization
    dataset = TimeSeriesDataset(args.data_file, amt_mean=amt_mean, amt_std=amt_std)
    
    # Split into train/test based on 'split' field
    train_indices = [i for i, item in enumerate(dataset.data) if item.get('split', 'train') == 'train']
    test_indices = [i for i, item in enumerate(dataset.data) if item.get('split', 'train') == 'test']
    
    # Random sample 500 training indices for fast debug
    if len(train_indices) > 500:
        train_indices = np.random.choice(train_indices, size=500, replace=False).tolist()

    if len(test_indices) > 200:
        test_indices = np.random.choice(test_indices, size=200, replace=False).tolist()
    
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    # Create data loaders
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=custom_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=0)
    
    # Create model
    print(f"Creating OneForAll model...")
    model = OneForAllModel(patch_size=args.patch_size, d_model=args.d_model,
                          qwen_model=args.qwen_model, feature_dim=75, freeze_llm=True)
    model = model.to(device)
    
    # Use DataParallel for multi-GPU training
    # print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    # model = nn.DataParallel(model)
    
    # Setup optimizer (only train newly added parameters)
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    
    # Setup loss function
    if args.use_focal_loss:
        alpha = torch.tensor([args.focal_alpha, 1.0 - args.focal_alpha]).to(device) if args.focal_alpha else None
        criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss with gamma={args.focal_gamma}, alpha={alpha}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")
    
    # Training loop
    print("Starting training...")
    # Prepare CSV writers
    train_csv_path = os.path.join(args.output_dir, 'train_metrics.csv')
    test_csv_path = os.path.join(args.output_dir, 'test_metrics.csv')
    # Define the metric columns to persist (ordered)
    metric_columns = [
        'epoch',
        'baseline_precision', 'baseline_recall', 'baseline_f1', 'auc',
        'pcc', 'bs', 'ks', 'pg', 'h_measure'
    ]
    # Initialize CSV files with headers if they don't exist
    if not os.path.exists(train_csv_path):
        with open(train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metric_columns)
    if not os.path.exists(test_csv_path):
        with open(test_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metric_columns)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate on train and test
        print("Evaluating on train set...")
        train_metrics = evaluate(model, train_loader, device, split_name='train')
        print(f"Train - Baseline Precision: {train_metrics['baseline_precision']:.4f}, "
            f"Baseline Recall: {train_metrics['baseline_recall']:.4f}, "
            f"Baseline F1: {train_metrics['baseline_f1']:.4f}, AUC: {train_metrics['auc']:.4f}, "
            f"PCC: {train_metrics['pcc']:.4f}, BS: {train_metrics['bs']:.4f}, KS: {train_metrics['ks']:.4f}, PG: {train_metrics['pg']:.4f}, H-measure: {train_metrics['h_measure']:.4f}")
        # Append train metrics to CSV
        with open(train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['baseline_precision'], train_metrics['baseline_recall'],
                train_metrics['baseline_f1'], train_metrics['auc'],
                train_metrics['pcc'], train_metrics['bs'], train_metrics['ks'], train_metrics['pg'], train_metrics['h_measure']
            ])
        
        print("Evaluating on test set...")
        test_metrics = evaluate(model, test_loader, device, split_name='test')
        print(f"Test  - Baseline Precision: {test_metrics['baseline_precision']:.4f}, "
            f"Baseline Recall: {test_metrics['baseline_recall']:.4f}, "
            f"Baseline F1: {test_metrics['baseline_f1']:.4f}, AUC: {test_metrics['auc']:.4f}, "
            f"PCC: {test_metrics['pcc']:.4f}, BS: {test_metrics['bs']:.4f}, KS: {test_metrics['ks']:.4f}, PG: {test_metrics['pg']:.4f}, H-measure: {test_metrics['h_measure']:.4f}")
        # Append test metrics to CSV
        with open(test_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                test_metrics['baseline_precision'], test_metrics['baseline_recall'],
                test_metrics['baseline_f1'], test_metrics['auc'],
                test_metrics['pcc'], test_metrics['bs'], test_metrics['ks'], test_metrics['pg'], test_metrics['h_measure']
            ])
        
        # Save checkpoint
        # ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch + 1}.pt')
        # # Handle DataParallel wrapper when saving
        # model_to_save = model.module if hasattr(model, 'module') else model
        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model_to_save.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_metrics': train_metrics,
        #     'test_metrics': test_metrics,
        # }, ckpt_path)
        # print(f"Saved checkpoint to {ckpt_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print("\nTrain Set Metrics (Baseline):")
    print(f"  Precision: {train_metrics['baseline_precision']:.4f}")
    print(f"  Recall:    {train_metrics['baseline_recall']:.4f}")
    print(f"  F1:        {train_metrics['baseline_f1']:.4f}")
    print(f"  AUC:       {train_metrics['auc']:.4f}")
    print(f"  PCC:       {train_metrics['pcc']:.4f}")
    print(f"  BS:        {train_metrics['bs']:.4f}")
    print(f"  KS:        {train_metrics['ks']:.4f}")
    print(f"  PG:        {train_metrics['pg']:.4f}")
    
    print("\nTest Set Metrics (Baseline):")
    print(f"  Precision: {test_metrics['baseline_precision']:.4f}")
    print(f"  Recall:    {test_metrics['baseline_recall']:.4f}")
    print(f"  F1:        {test_metrics['baseline_f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  PCC:       {test_metrics['pcc']:.4f}")
    print(f"  BS:        {test_metrics['bs']:.4f}")
    print(f"  KS:        {test_metrics['ks']:.4f}")
    print(f"  PG:        {test_metrics['pg']:.4f}")


if __name__ == '__main__':
    main()
