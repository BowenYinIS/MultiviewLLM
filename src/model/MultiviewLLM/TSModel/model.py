import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from dataloader import create_dataloader
import argparse

class TransactionEmbedding(nn.Module):
    """MLP to transform 6 transaction variables into embeddings"""
    # TODO: categorical features as vocabulary embedding 
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

# New: per-feature one-hot projection to d_model, then summed
class PerFeatureEmbedding(nn.Module):
    """Project per-feature one-hot (or scalar) inputs to d_model and sum them."""
    def __init__(self, d_model=256,
                 mcc_classes=13, hod_classes=24, dow_classes=7, wom_classes=6, moy_classes=12):
        super().__init__()
        self.d_model = d_model
        # Linear layers for categorical one-hot inputs
        self.embed_mcc = nn.Linear(mcc_classes, d_model)
        self.embed_hod = nn.Linear(hod_classes, d_model)
        self.embed_dow = nn.Linear(dow_classes, d_model)
        self.embed_wom = nn.Linear(wom_classes, d_model)
        self.embed_moy = nn.Linear(moy_classes, d_model)
        # Linear for amount scalar
        self.embed_amt = nn.Linear(1, d_model)

    def forward(self, mcc_onehot, hod_onehot, dow_onehot, wom_onehot, moy_onehot, amt_scalar):
        # Inputs shapes: [seq_len, C] for onehots, amt_scalar: [seq_len, 1]
        # Project each to [seq_len, d_model]
        mch = self.embed_mcc(mcc_onehot)
        h = self.embed_hod(hod_onehot)
        d = self.embed_dow(dow_onehot)
        w = self.embed_wom(wom_onehot)
        mo = self.embed_moy(moy_onehot)
        a = self.embed_amt(amt_scalar)

        # Sum embeddings
        emb = mch + h + d + w + mo + a
        return emb

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for time series data"""
    
    def __init__(self, 
                 input_dim=6,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_len=2000,
                 num_mcc=13,
                 num_hod=24,
                 num_dow=7,
                 num_wom=6,
                 num_moy=12):
        super().__init__()
        
        self.d_model = d_model
        # store category sizes for one-hot encoding
        self.num_mcc = num_mcc
        self.num_hod = num_hod
        self.num_dow = num_dow
        self.num_wom = num_wom
        self.num_moy = num_moy

        # Use per-feature one-hot projection + sum into d_model
        # For MCC we use 13 coarse classes (mcc_to_class), other categories use provided sizes
        self.transaction_embedding = PerFeatureEmbedding(
            d_model=d_model,
            mcc_classes=13,
            hod_classes=self.num_hod,
            dow_classes=self.num_dow,
            wom_classes=self.num_wom,
            moy_classes=self.num_moy
        )
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Prediction heads for self-supervised learning
        self.mcc_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 13)  # Predict MCC class (13 classes)
        )
        
        self.amt_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Predict transaction amount (regression)
        )
        
    
    def forward(self, batch_data):
        """
        Forward pass for time series transformer
        Returns token embeddings for each transaction
        batch_data: dict with keys ['mcc_cde', 'hod', 'dow', 'wom', 'moy', 'txn_amt', 'target']
        """
        batch_size = len(batch_data['mcc_cde'])
        batch_embeddings = []
        
        for i in range(batch_size):
            # Get sequence length
            seq_len = len(batch_data['mcc_cde'][i])
            
            # Build per-feature one-hot encodings and amount scalar
            device = batch_data['mcc_cde'][i].device

            # mcc: map raw codes to 13 coarse classes via mcc_to_class (masked -> -1)
            mcc_codes = batch_data['mcc_cde'][i]
            mcc_classes = [mcc_to_class(x.item()) for x in mcc_codes]
            mcc_idx = torch.tensor([c if c >= 0 else 0 for c in mcc_classes], dtype=torch.long, device=device)
            mcc_onehot = F.one_hot(mcc_idx, num_classes=13).float()
            # zero-out masked positions (where class == -1)
            mask_mcc_valid = torch.tensor([1.0 if c >= 0 else 0.0 for c in mcc_classes], dtype=mcc_onehot.dtype, device=device).unsqueeze(1)
            mcc_onehot = mcc_onehot * mask_mcc_valid

            # hod (0-23)
            hod_idx = batch_data['hod'][i].long().clamp(min=0, max=self.num_hod - 1).to(device)
            hod_onehot = F.one_hot(hod_idx, num_classes=self.num_hod).float()

            # dow (0-6)
            dow_idx = batch_data['dow'][i].long().clamp(min=0, max=self.num_dow - 1).to(device)
            dow_onehot = F.one_hot(dow_idx, num_classes=self.num_dow).float()

            # wom (usually 1-5) keep as provided (clamp)
            wom_idx = batch_data['wom'][i].long().clamp(min=0, max=self.num_wom - 1).to(device)
            wom_onehot = F.one_hot(wom_idx, num_classes=self.num_wom).float()

            # moy (1-12) may be 1-based; clamp to range
            moy_idx = batch_data['moy'][i].long().clamp(min=0, max=self.num_moy - 1).to(device)
            moy_onehot = F.one_hot(moy_idx, num_classes=self.num_moy).float()

            amt_col = batch_data['txn_amt'][i].unsqueeze(1).float().to(device)

            # Project per-feature one-hots and amount and sum
            embeddings = self.transaction_embedding(mcc_onehot, hod_onehot, dow_onehot, wom_onehot, moy_onehot, amt_col)
            batch_embeddings.append(embeddings)
        
        # Pad sequences to same length
        max_len = min(max(emb.shape[0] for emb in batch_embeddings), 2000)
        padded_embeddings = []
        attention_masks = []
        
        for emb in batch_embeddings:
            seq_len = emb.shape[0]
            # Pad with zeros
            if seq_len > max_len:
                emb = emb[:max_len, :]
            if seq_len < max_len:
                # create padding on the same device as embeddings to avoid device mismatch
                padding = torch.zeros(max_len - seq_len, self.d_model, device=emb.device)
                padded_emb = torch.cat([emb, padding], dim=0)
                mask = torch.cat([torch.ones(seq_len, device=emb.device), torch.zeros(max_len - seq_len, device=emb.device)])
            else:
                padded_emb = emb
                mask = torch.ones(max_len, device=emb.device)
            
            padded_embeddings.append(padded_emb)
            attention_masks.append(mask)
        
        # Stack into batch tensor
        x = torch.stack(padded_embeddings)  # Shape: [batch_size, max_len, d_model]
        attention_mask = torch.stack(attention_masks)  # Shape: [batch_size, max_len]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [max_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, max_len, d_model]
        
        # Create attention mask for transformer (True means ignore)
        # Ensure attention mask is on the same device as the model input
        attention_mask = (attention_mask == 0).bool().to(x.device)
        
        # Transformer encoder
        token_embeddings = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        
        return {
            'token_embeddings': token_embeddings,  # [batch_size, max_len, d_model]
            'attention_mask': attention_mask,      # [batch_size, max_len]
            'sequence_lengths': (~attention_mask).sum(dim=1)  # [batch_size]
        }
    
    def forward_with_predictions(self, batch_data, mask_prob=0.15):
        """
        Forward pass with self-supervised learning predictions
        """
        # Get base embeddings
        output = self.forward(batch_data)
        token_embeddings = output['token_embeddings']
        attention_mask = output['attention_mask']
        
        # Predict MCC codes and amounts
        mcc_predictions = self.mcc_predictor(token_embeddings)  # [batch_size, max_len, 1]
        amt_predictions = self.amt_predictor(token_embeddings)   # [batch_size, max_len, 1]
        
        return {
            'token_embeddings': token_embeddings,
            'attention_mask': attention_mask,
            'sequence_lengths': output['sequence_lengths'],
            'mcc_predictions': mcc_predictions,  # [batch_size, max_len, 13]
            'amt_predictions': amt_predictions.squeeze(-1)    # [batch_size, max_len]
        }
    

def apply_masking(batch_data, mask_prob=0.15, device='cuda'):
    """
    Apply masking to batch data for self-supervised learning
    Replaces mask_prob percentage of tokens with [MASK] tokens
    """
    batch_size = len(batch_data['mcc_cde'])
    masked_batch = {}
    
    # Copy all data
    for key, value in batch_data.items():
        masked_batch[key] = value.copy() if isinstance(value, list) else value
    
    # Create mask tokens (use special values for [MASK])
    MASK_MCC = -1  # Special value for masked MCC
    MASK_AMT = -1.0  # Special value for masked amount (will be handled in loss computation)
    
    for i in range(batch_size):
        seq_len = len(batch_data['mcc_cde'][i])
        
        # Calculate number of tokens to mask
        num_mask = max(1, int(seq_len * mask_prob))
        
        # Randomly select positions to mask
        mask_positions = random.sample(range(seq_len), min(num_mask, seq_len))
        
        # Apply masking
        for pos in mask_positions:
            masked_batch['mcc_cde'][i][pos] = MASK_MCC
            masked_batch['txn_amt'][i][pos] = MASK_AMT
            # Keep other features (hod, dow, wom, moy) unchanged
    
    return masked_batch, mask_positions if batch_size == 1 else [mask_positions for _ in range(batch_size)]

def mcc_to_class(mcc_code):
    """
    Convert MCC code to class (0-12)
    Classes: [0001-1499, 1500-2999, 3000-3299, 3300-3499, 3500-3999, 4000-4799, 
              4800-4999, 5000-5599, 5600-5699, 5700-7299, 7300-7999, 8000-8999, 9000-9999]
    """
    if mcc_code < 0:  # Masked token
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
        return 12  # Default to last class for unknown codes

def create_model(d_model=256, nhead=8, num_layers=6, device='cuda',
                 num_mcc=13, num_hod=24, num_dow=7, num_wom=6, num_moy=12):
    """Create the time series transformer model"""
    model = TimeSeriesTransformer(
        input_dim=6,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model*2,
        dropout=0.1,
        num_mcc=num_mcc,
        num_hod=num_hod,
        num_dow=num_dow,
        num_wom=num_wom,
        num_moy=num_moy
    )

    # Move model to device
    model = model.to(device)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time Series Transformer')
    parser.add_argument('--file_path', type=str, 
                       default="data/processed_data/ts_processed_data/multivariate_timeseries_train_435.jsonl",
                       help='Path to the JSONL data file')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for the DataLoader')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='Whether to shuffle the data')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker processes for data loading')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create dataloader
    dataloader = create_dataloader(
        file_path=args.file_path,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )
    
    # Create model
    model = create_model(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Testing batch {i+1}...")
            
            # Move batch data to device
            batch_on_device = {}
            for key, value in batch.items():
                if key in ['mcc_cde', 'hod', 'dow', 'wom', 'moy', 'txn_amt', 'target']:
                    if key == 'target':
                        batch_on_device[key] = value.to(device)
                    else:
                        # For lists of tensors, move each tensor to device
                        batch_on_device[key] = [tensor.to(device) for tensor in value]
                else:
                    batch_on_device[key] = value  # Keep non-tensor data as is
            
            output = model(batch_on_device)
            token_embeddings = output['token_embeddings']
            attention_mask = output['attention_mask']
            sequence_lengths = output['sequence_lengths']
            
            print(f"  Input batch size: {len(batch['target'])}")
            print(f"  Token embeddings shape: {token_embeddings.shape}")
            print(f"  Token embeddings device: {token_embeddings.device}")
            print(f"  Attention mask shape: {attention_mask.shape}")
            print(f"  Sequence lengths: {sequence_lengths.tolist()}")
            print(f"  First sample token embeddings shape: {token_embeddings[0].shape}")
            print(f"  First sample valid tokens: {sequence_lengths[0].item()}")
            if i >= 2:  # Only test first 3 batches
                break