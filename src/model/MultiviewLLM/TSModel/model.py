import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from dataloader import create_dataloader
import argparse

class TransactionEmbedding(nn.Module):
    """MLP to transform 6 transaction variables into embeddings"""
    
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
                 max_len=1000):
        super().__init__()
        
        self.d_model = d_model
        self.transaction_embedding = TransactionEmbedding(input_dim, d_model//2, d_model)
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
            
            # Stack the 6 features: [mcc_cde, hod, dow, wom, moy, txn_amt]
            features = torch.stack([
                batch_data['mcc_cde'][i].float(),
                batch_data['hod'][i].float(),
                batch_data['dow'][i].float(),
                batch_data['wom'][i].float(),
                batch_data['moy'][i].float(),
                batch_data['txn_amt'][i]
            ], dim=1)  # Shape: [seq_len, 6]
            
            # Transform through MLP
            embeddings = self.transaction_embedding(features)  # Shape: [seq_len, d_model]
            batch_embeddings.append(embeddings)
        
        # Pad sequences to same length
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        padded_embeddings = []
        attention_masks = []
        
        for emb in batch_embeddings:
            seq_len = emb.shape[0]
            # Pad with zeros
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, self.d_model).cuda()
                padded_emb = torch.cat([emb, padding], dim=0)
                mask = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])
            else:
                padded_emb = emb
                mask = torch.ones(max_len)
            
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
        attention_mask = (attention_mask == 0).bool().cuda()
        
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

def create_model(d_model=256, nhead=8, num_layers=6, device='cuda'):
    """Create the time series transformer model"""
    model = TimeSeriesTransformer(
        input_dim=6,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model*2,
        dropout=0.1
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