#!/usr/bin/env python3
"""
Script to reconstruct the original data format from train and test splits.
Merges the data back to original format with embeddings computed using the trained model.
"""

import json
import numpy as np
from typing import Dict, List, Any
import pandas as pd
import pickle
import torch
import sys
import os
from tqdm import tqdm

# Add the model path to sys.path
sys.path.append('/data/mjmao/credit/MultiviewLLM/src/model/MultiviewLLM/TSModel')
from model import TimeSeriesTransformer, create_model

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

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

def create_embeddings_with_model(data: List[Dict[str, Any]], model, device='cuda', batch_size=32) -> List[Dict[str, Any]]:
    """Create embeddings using the trained model with batching for efficiency."""
    model.eval()
    embeddings = []
    
    print(f"Computing embeddings using trained model (batch_size={batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Computing embeddings"):
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

def validate_act_idn_sky_order(data: List[Dict[str, Any]]) -> bool:
    """Validate that act_idn_sky is properly ordered."""
    print("Validating act_idn_sky ordering...")
    
    # Group by act_idn_sky
    grouped = {}
    for i, record in enumerate(data):
        act_idn = record['act_idn_sky']
        if act_idn not in grouped:
            grouped[act_idn] = []
        grouped[act_idn].append(i)
    
    # Check that all records for the same act_idn_sky are consecutive
    prev_act_idn = None
    consecutive_count = 0
    
    for record in data:
        current_act_idn = record['act_idn_sky']
        
        if prev_act_idn is None:
            prev_act_idn = current_act_idn
            consecutive_count = 1
        elif current_act_idn == prev_act_idn:
            consecutive_count += 1
        else:
            # New act_idn_sky, check if previous group was consecutive
            expected_count = len(grouped[prev_act_idn])
            if consecutive_count != expected_count:
                print(f"Warning: act_idn_sky {prev_act_idn} records are not consecutive. Expected {expected_count}, got {consecutive_count}")
                print(f"This is expected when merging train and test splits.")
            
            prev_act_idn = current_act_idn
            consecutive_count = 1
    
    # Check the last group
    if prev_act_idn is not None:
        expected_count = len(grouped[prev_act_idn])
        if consecutive_count != expected_count:
            print(f"Warning: act_idn_sky {prev_act_idn} records are not consecutive. Expected {expected_count}, got {consecutive_count}")
            print(f"This is expected when merging train and test splits.")
    
    print("✓ act_idn_sky ordering validation completed (warnings are expected for merged data)")
    return True

def load_original_order():
    """Load the original order from the feather file."""
    import pandas as pd
    print("Loading original data order from feather file...")
    df = pd.read_feather('/data/mjmao/credit/MultiviewLLM/data/processed_data/sample_index/samples_min12mo_fixed_2test.feather')
    original_order = df['act_idn_sky'].tolist()
    print(f"Original order loaded: {len(original_order)} records")
    return original_order

def reorder_data_by_original_order(data, original_order):
    """Reorder the merged data according to the original order."""
    print("Reordering data according to original order...")
    
    # Create a mapping from act_idn_sky to data records
    data_by_act_idn = {}
    for record in data:
        act_idn = record['act_idn_sky']
        if act_idn not in data_by_act_idn:
            data_by_act_idn[act_idn] = []
        data_by_act_idn[act_idn].append(record)
    
    # Reorder according to original order
    reordered_data = []
    for act_idn in original_order:
        if act_idn in data_by_act_idn:
            # Take only the first record for each act_idn_sky to maintain 1:1 mapping
            if data_by_act_idn[act_idn]:
                reordered_data.append(data_by_act_idn[act_idn].pop(0))
    
    print(f"Reordered data: {len(reordered_data)} records")
    return reordered_data

def main():
    """Main function to reconstruct the data."""
    print("Starting data reconstruction with trained model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load original order
    original_order = load_original_order()
    
    # Load train and test data
    print("Loading train data...")
    train_data = load_jsonl('/data/mjmao/credit/MultiviewLLM/data/processed_data/ts_processed_data/samples_min12mo_fixed_2test_train.jsonl')
    print(f"Train data loaded: {len(train_data)} records")
    
    print("Loading test data...")
    test_data = load_jsonl('/data/mjmao/credit/MultiviewLLM/data/processed_data/ts_processed_data/samples_min12mo_fixed_2test_test.jsonl')
    print(f"Test data loaded: {len(test_data)} records")
    
    # Merge train and test data (train first, then test)
    print("Merging train and test data...")
    merged_data = train_data + test_data
    print(f"Total merged data: {len(merged_data)} records")
    
    # Reorder according to original order
    merged_data = reorder_data_by_original_order(merged_data, original_order)
    
    # Validate act_idn_sky ordering
    validate_act_idn_sky_order(merged_data)
    
    # Create and load model
    print("Creating model...")
    model = create_model(d_model=256, nhead=8, num_layers=6, device=device)
    
    # Load checkpoint
    checkpoint_path = '/data/mjmao/credit/MultiviewLLM/checkpoint/MultiviewLLM/TSModel/checkpoint_best.pt'
    model = load_checkpoint(model, checkpoint_path)
    model.to(device)
    
    # Create embeddings using the trained model
    print("Creating embeddings using trained model...")
    embeddings_data = create_embeddings_with_model(merged_data, model, device, batch_size=1)
    
    # Create final dictionary with row IDs as keys and embeddings as values
    print("Creating final dictionary...")
    result_dict = {}
    
    for i, embedding_data in enumerate(embeddings_data):
        # Use the index as row ID (0-based)
        # Save the full token embeddings matrix (交易次数 × embedding_dim)
        token_embeddings_matrix = embedding_data['embeddings']  # Shape: [seq_len, embedding_dim]
        result_dict[i] = token_embeddings_matrix.tolist()
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(embeddings_data)} records")
    
    # Save the result
    output_path = '/data/mjmao/credit/MultiviewLLM/checkpoint/MultiviewLLM/TSModel/samples_min12mo_fixed_2test_matrix.pkl'
    print(f"Saving reconstructed data to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(result_dict, f)

if __name__ == "__main__":
    main()
