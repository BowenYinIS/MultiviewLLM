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
    batch_embeddings = {}  # dict: key=row_index, value=embedding_matrix
    
    print("Computing embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch_samples = data[i:i+batch_size]
            
            for j, sample in enumerate(batch_samples):
                # Prepare batch data for single sample
                batch_data = prepare_batch_data(sample, device)
                
                # Forward pass
                output = model(batch_data)
                token_embeddings = output['token_embeddings']  # [1, seq_len, d_model]
                attention_mask = output['attention_mask']  # [1, seq_len]
                sequence_length = output['sequence_lengths'][0].item()  # actual sequence length
                
                # Extract only the valid tokens (remove padding)
                valid_embeddings = token_embeddings[0, :sequence_length, :]  # [seq_len, d_model]
                
                # Store in dict format: key=row_index (行数), value=embedding_matrix
                row_index = i + j  # 当前行在JSONL文件中的索引
                batch_embeddings[row_index] = valid_embeddings.cpu().numpy()
    
    return batch_embeddings

def save_embeddings(batch_embeddings, output_path):
    """Save embeddings to file"""
    print(f"Saving embeddings to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as numpy file
    np.save(output_path, batch_embeddings)
    print(f"Embeddings saved successfully")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    model_path = "/data/mjmao/credit/MultiviewLLM/src/model/MultiviewLLM/TSModel/model.py"
    checkpoint_path = "checkpoint/MultiviewLLM/TSModel/contrastive/best_contrastive_checkpoint_samples_min12mo_fixed_2test.pt"
    data_path = "/data/mjmao/credit/MultiviewLLM/data/processed_data/ts_processed_data/samples_min12mo_fixed_2test.jsonl"
    output_path = "/data/mjmao/credit/MultiviewLLM/checkpoint/MultiviewLLM/TSModel/samples_min12mo_fixed_2test.npy"
    
    # Create model
    print("Creating model...")
    model = create_model(d_model=256, nhead=8, num_layers=6, device=device)
    
    # Load checkpoint
    model = load_checkpoint(model, checkpoint_path)
    
    # Load data
    data = load_data(data_path)
    
    # # Select specific rows to process
    # selected_rows = [41837, 41838, 41839]  # 选择特定的行号
    # print(f"Data length: {len(data)}")
    # print(f"Selected rows: {selected_rows}")
    
    # # Filter data to only include selected rows
    # filtered_data = [data[i] for i in selected_rows if i < len(data)]
    # print(f"Filtered data length: {len(filtered_data)}")
    
    # Compute embeddings for filtered data
    batch_embeddings = compute_embeddings(model, data, device=device, batch_size=1)
    
    # Save embeddings
    save_embeddings(batch_embeddings, output_path)
    
    print("Embedding generation completed!")
    print(f"Generated embeddings for {len(batch_embeddings)} samples")
    # Get first key and show embedding shape
    first_key = list(batch_embeddings.keys())[0]
    print(f"Embedding shape for row {first_key}: {batch_embeddings[first_key].shape}")

if __name__ == "__main__":
    main()
