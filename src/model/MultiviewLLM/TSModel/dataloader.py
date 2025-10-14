import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultiviewTimeSeriesDataset(Dataset):
    """PyTorch Dataset for Multiview Time Series Data"""
    
    def __init__(self, file_path, amt_mean=None, amt_std=None):
        self.data = []
        self.amt_mean = amt_mean
        self.amt_std = amt_std
        self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load JSONL data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        
        # Extract time series features
        time_series = record['time_series']
        
        # Convert to tensors
        mcc_cde = torch.tensor(time_series['mcc_cde'], dtype=torch.long)
        hod = torch.tensor(time_series['hod'], dtype=torch.long)
        dow = torch.tensor(time_series['dow'], dtype=torch.long)
        wom = torch.tensor(time_series['wom'], dtype=torch.long)
        moy = torch.tensor(time_series['moy'], dtype=torch.long)
        
        # Apply log(1+x) transformation to transaction amounts
        raw_txn_amt = torch.tensor(time_series['txn_amt'], dtype=torch.float32)
        txn_amt = torch.log(1 + raw_txn_amt)
        
        # Apply normalization if statistics are provided
        if self.amt_mean is not None and self.amt_std is not None:
            txn_amt = (txn_amt - self.amt_mean) / self.amt_std
        
        # Target
        target = torch.tensor(record['target_delinquency'], dtype=torch.long)
        
        return {
            'mcc_cde': mcc_cde,
            'hod': hod,
            'dow': dow,
            'wom': wom,
            'moy': moy,
            'txn_amt': txn_amt,
            'target': target,
            'act_idn_sky': record['act_idn_sky'],
            'txn_des_series': record['txn_des_series'],
            'mcc_desc_series': record['mcc_desc_series']
        }
    
    def compute_amount_stats(self):
        """Compute mean and std of log-transformed amounts"""
        all_amounts = []
        for record in self.data:
            time_series = record['time_series']
            raw_amounts = torch.tensor(time_series['txn_amt'], dtype=torch.float32)
            log_amounts = torch.log(1 + raw_amounts)
            all_amounts.extend(log_amounts.tolist())
        
        all_amounts = torch.tensor(all_amounts)
        mean = all_amounts.mean().item()
        std = all_amounts.std().item()
        
        print(f"Amount statistics - Mean: {mean:.2f}, Std: {std:.2f}")
        return mean, std

def create_dataloader(file_path, batch_size=32, shuffle=True, num_workers=0, amt_mean=None, amt_std=None):
    """Create PyTorch DataLoader"""
    dataset = MultiviewTimeSeriesDataset(file_path, amt_mean=amt_mean, amt_std=amt_std)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    return dataloader

def custom_collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Separate the data
    mcc_cde = [item['mcc_cde'] for item in batch]
    hod = [item['hod'] for item in batch]
    dow = [item['dow'] for item in batch]
    wom = [item['wom'] for item in batch]
    moy = [item['moy'] for item in batch]
    txn_amt = [item['txn_amt'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    # Keep non-tensor data as lists
    act_ids = [item['act_idn_sky'] for item in batch]
    txn_des = [item['txn_des_series'] for item in batch]
    mcc_desc = [item['mcc_desc_series'] for item in batch]
    
    return {
        'mcc_cde': mcc_cde,
        'hod': hod,
        'dow': dow,
        'wom': wom,
        'moy': moy,
        'txn_amt': txn_amt,
        'target': targets,
        'act_idn_sky': act_ids,
        'txn_des_series': txn_des,
        'mcc_desc_series': mcc_desc
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load multivariate time series data into PyTorch DataLoader')
    parser.add_argument('--file_path', type=str, 
                       default="data/processed_data/ts_processed_data/multivariate_timeseries_train_435.jsonl",
                       help='Path to the JSONL data file')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for the DataLoader')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='Whether to shuffle the data')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    print("Creating PyTorch DataLoader...")
    print(f"File path: {args.file_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Shuffle: {args.shuffle}")
    print(f"Num workers: {args.num_workers}")
    
    dataloader = create_dataloader(
        file_path=args.file_path,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )
        
    # Test the DataLoader
    print("\nTesting DataLoader...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Batch size: {len(batch['target'])}")
        print(f"  Target shape: {batch['target'].shape}")
        print(f"  MCC codes length (first sample): {len(batch['mcc_cde'][0])}")
        print(f"  Transaction amounts length (first sample): {len(batch['txn_amt'][0])}")
        print(f"  Targets: {batch['target'][:5].tolist()}")
        if i >= 2:  
            break
