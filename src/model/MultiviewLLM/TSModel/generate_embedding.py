import torch
import json
import os
from tqdm import tqdm
import numpy as np
from model import TimeSeriesTransformer, create_model

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Checkpoint loaded successfully")
    return model

def load_data(file_path):
    """Load JSONL data file"""
    print(f"Loading data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} samples")
    return data

def prepare_batch_data(sample, device='cuda'):
    """Convert a single sample to batch format for the model"""
    time_series = sample['time_series']
    
    # Convert to tensors and create batch format
    batch_data = {
        'mcc_cde': [torch.tensor(time_series['mcc_cde'], dtype=torch.long).to(device)],
        'hod': [torch.tensor(time_series['hod'], dtype=torch.long).to(device)],
        'dow': [torch.tensor(time_series['dow'], dtype=torch.long).to(device)],
        'wom': [torch.tensor(time_series['wom'], dtype=torch.long).to(device)],
        'moy': [torch.tensor(time_series['moy'], dtype=torch.long).to(device)],
        'txn_amt': [torch.tensor(time_series['txn_amt'], dtype=torch.float).to(device)],
        'target': torch.tensor([sample['target_delinquency']], dtype=torch.long).to(device)
    }
    
    return batch_data

def compute_embeddings(model, data, device='cuda', batch_size=1):
    """Compute embeddings for all data samples"""
    model.eval()
    embeddings = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch_samples = data[i:i+batch_size]
            batch_embeddings = []
            
            for sample in batch_samples:
                # Prepare batch data for single sample
                batch_data = prepare_batch_data(sample, device)
                
                # Forward pass
                output = model(batch_data)
                token_embeddings = output['token_embeddings']  # [1, seq_len, d_model]
                attention_mask = output['attention_mask']  # [1, seq_len]
                sequence_length = output['sequence_lengths'][0].item()  # actual sequence length
                
                # Extract only the valid tokens (remove padding)
                valid_embeddings = token_embeddings[0, :sequence_length, :]  # [seq_len, d_model]
                
                # Convert to numpy and store
                batch_embeddings.append({
                    'act_idn_sky': sample['act_idn_sky'],
                    'embeddings': valid_embeddings.cpu().numpy(),
                    'sequence_length': sequence_length,
                    'target_delinquency': sample['target_delinquency']
                })
            
            embeddings.extend(batch_embeddings)
    
    return embeddings

def save_embeddings(embeddings, output_path):
    """Save embeddings to file"""
    print(f"Saving embeddings to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as numpy file
    np.save(output_path, embeddings)
    print(f"Embeddings saved successfully")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    model_path = "/data/mjmao/credit/MultiviewLLM/src/model/MultiviewLLM/TSModel/model.py"
    checkpoint_path = "/data/mjmao/credit/MultiviewLLM/checkpoint/MultiviewLLM/TSModel/checkpoint_best.pt"
    data_path = "/data/mjmao/credit/MultiviewLLM/data/processed_data/ts_processed_data/multivariate_timeseries_train_435.jsonl"
    output_path = "/data/mjmao/credit/MultiviewLLM/checkpoint/MultiviewLLM/TSModel/ts_embeddings.npy"
    
    # Create model
    print("Creating model...")
    model = create_model(d_model=256, nhead=8, num_layers=6, device=device)
    
    # Load checkpoint
    model = load_checkpoint(model, checkpoint_path)
    
    # Load data
    data = load_data(data_path)
    
    # Compute embeddings
    embeddings = compute_embeddings(model, data, device=device, batch_size=1)
    
    # Save embeddings
    save_embeddings(embeddings, output_path)
    
    print("Embedding generation completed!")
    print(f"Generated embeddings for {len(embeddings)} samples")
    print(f"Embedding shape for first sample: {embeddings[0]['embeddings'].shape}")

if __name__ == "__main__":
    main()
