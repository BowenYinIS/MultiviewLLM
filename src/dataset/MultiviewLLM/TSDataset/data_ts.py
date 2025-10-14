import pandas as pd
import argparse
import os
import json
import numpy as np 

class TSDataReader:
    def __init__(self, processed_data_path, raw_data_path):
        self.processed_data_path = processed_data_path
        self.raw_data_path = raw_data_path
        self.data = pd.read_feather(self.processed_data_path)
        self.raw_data = pd.read_feather(self.raw_data_path)
    
    def get_transactions_for_row(self, row_idx):
        """
        Get all transactions for a specific row based on act_idn_sky and billing_dates
        
        Args:
            row_idx: Index of the row in self.data
            
        Returns:
            DataFrame with filtered transactions
        """
        row = self.data.iloc[row_idx]
        act_idn_sky = row['act_idn_sky']
        billing_dates = row['billing_dates']
        
        # Filter transactions with same account ID and billing date
        filtered_transactions = self.raw_data[
            (self.raw_data['act_idn_sky'] == act_idn_sky) & 
            (self.raw_data['billing_date'].isin(billing_dates))
        ]
        return filtered_transactions
    
    def get_all_time_series(self):
        time_series_data = {}
        for idx in range(len(self.data)):
            time_series_data[idx] = self.get_transactions_for_row(idx)
        return time_series_data
    
    def get_train_data_with_sorted_transactions(self):
        """
        Filter for train data and get associated transactions sorted by txn_dte and txn_tme
        
        Returns:
            Dictionary with train row indices as keys and sorted transactions as values
        """
        # Filter for train data only
        train_data = self.data[self.data['split'] == 'train']
        train_indices = train_data.index.tolist()
        
        train_time_series = {}
        for idx in train_indices:
            # Get transactions for this train row
            transactions = self.get_transactions_for_row(idx)
            # Sort by txn_dte and txn_tme
            if not transactions.empty:
                transactions = transactions.sort_values(['txn_dte', 'txn_tme'])
            train_time_series[idx] = transactions
        return train_time_series
    
    def create_multivariate_timeseries(self, transactions_df):
        """
        Convert transactions into multivariate time series with features:
        - mcc_cde: Merchant category code
        - hod: Hour of day
        - dow: Day of week  
        - wom: Week of month
        - moy: Month of year
        - txn_amt: Transaction amount
        - txn_desc: Transaction description
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            Dictionary with feature names as keys and ordered lists as values
        """
        if transactions_df.empty:
            return {}
        
        # Create a copy to avoid modifying original
        ts_data = transactions_df.copy()
        
        # Extract time features from txn_dte and txn_tme
        ts_data['txn_dte'] = pd.to_datetime(ts_data['txn_dte'])
        ts_data['txn_tme'] = pd.to_datetime(ts_data['txn_tme'], format='%H:%M:%S').dt.time
        
        # Combine date and time for full datetime
        ts_data['datetime'] = pd.to_datetime(ts_data['txn_dte'].astype(str) + ' ' + ts_data['txn_tme'].astype(str))
        
        # Extract features
        ts_data['hod'] = ts_data['datetime'].dt.hour  # Hour of day
        ts_data['dow'] = ts_data['datetime'].dt.dayofweek  # Day of week (0=Monday)
        ts_data['wom'] = ((ts_data['datetime'].dt.day - 1) // 7) + 1  # Week of month
        ts_data['moy'] = ts_data['datetime'].dt.month  # Month of year
        
        # Select the multivariate features
        multivariate_features = ['mcc_cde', 'hod', 'dow', 'wom', 'moy', 'txn_amt', 'txn_desc']
        
        # Check which columns exist in the data
        available_features = [col for col in multivariate_features if col in ts_data.columns]
        
        # Convert to dictionary with ordered lists
        result = {}
        for feature in available_features:
            values = ts_data[feature].tolist()
            
            # Handle None values in mcc_cde and convert to int
            if feature == 'mcc_cde':
                values = [9999 if x is None else int(x) for x in values]
            # Handle None values in txn_desc and convert to string
            elif feature == 'txn_desc':
                values = ['UNKNOWN' if x is None else str(x) for x in values]
            
            result[feature] = values
        
        return result
    
    def create_txn_des_timeseries(self, transactions_df):
        """
        Extract txn_des as a separate time series
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            List of txn_des values ordered by time
        """
        if transactions_df.empty:
            return []
        
        # Create a copy to avoid modifying original
        ts_data = transactions_df.copy()
        
        # Extract time features from txn_dte and txn_tme
        ts_data['txn_dte'] = pd.to_datetime(ts_data['txn_dte'])
        ts_data['txn_tme'] = pd.to_datetime(ts_data['txn_tme'], format='%H:%M:%S').dt.time
        
        # Combine date and time for full datetime
        ts_data['datetime'] = pd.to_datetime(ts_data['txn_dte'].astype(str) + ' ' + ts_data['txn_tme'].astype(str))
        
        # Sort by datetime to ensure chronological order
        ts_data = ts_data.sort_values('datetime')
        
        # Extract txn_des values
        if 'txn_des' in ts_data.columns:
            txn_des_values = ts_data['txn_des'].tolist()
            # Handle None values
            txn_des_values = ['UNKNOWN' if x is None else str(x) for x in txn_des_values]
            return txn_des_values
        else:
            return []
    
    def get_all_multivariate_timeseries(self):
        """
        Create multivariate time series for all train data
        
        Returns:
            Dictionary with row indices as keys and multivariate time series as values
        """
        train_time_series = self.get_train_data_with_sorted_transactions()
        
        all_multivariate_ts = {}
        for row_idx, transactions in train_time_series.items():
            multivariate_ts = self.create_multivariate_timeseries(transactions)
            all_multivariate_ts[row_idx] = multivariate_ts
        
        return all_multivariate_ts
    
    def get_all_txn_des_timeseries(self):
        """
        Create txn_des time series for all train data
        
        Returns:
            Dictionary with row indices as keys and txn_des time series as values
        """
        train_time_series = self.get_train_data_with_sorted_transactions()
        
        all_txn_des_ts = {}
        for row_idx, transactions in train_time_series.items():
            txn_des_ts = self.create_txn_des_timeseries(transactions)
            all_txn_des_ts[row_idx] = txn_des_ts
        
        return all_txn_des_ts
    
    def create_mcc_desc_timeseries(self, transactions_df):
        """
        Extract mcc_desc as a separate time series
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            List of mcc_desc values ordered by time
        """
        if transactions_df.empty:
            return []
        
        # Create a copy to avoid modifying original
        ts_data = transactions_df.copy()
        
        # Extract time features from txn_dte and txn_tme
        ts_data['txn_dte'] = pd.to_datetime(ts_data['txn_dte'])
        ts_data['txn_tme'] = pd.to_datetime(ts_data['txn_tme'], format='%H:%M:%S').dt.time
        
        # Combine date and time for full datetime
        ts_data['datetime'] = pd.to_datetime(ts_data['txn_dte'].astype(str) + ' ' + ts_data['txn_tme'].astype(str))
        
        # Sort by datetime to ensure chronological order
        ts_data = ts_data.sort_values('datetime')
        
        # Extract mcc_desc values
        if 'mcc_desc' in ts_data.columns:
            mcc_desc_values = ts_data['mcc_desc'].tolist()
            # Handle None values
            mcc_desc_values = ['UNKNOWN' if x is None else str(x) for x in mcc_desc_values]
            return mcc_desc_values
        else:
            return []
    
    def get_all_mcc_desc_timeseries(self):
        """
        Create mcc_desc time series for all train data
        
        Returns:
            Dictionary with row indices as keys and mcc_desc time series as values
        """
        train_time_series = self.get_train_data_with_sorted_transactions()
        
        all_mcc_desc_ts = {}
        for row_idx, transactions in train_time_series.items():
            mcc_desc_ts = self.create_mcc_desc_timeseries(transactions)
            all_mcc_desc_ts[row_idx] = mcc_desc_ts
        
        return all_mcc_desc_ts
    
    def get_time_series_for_split(self, split):
        """
        Get time series data for a specific split (train or test)
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with row indices as keys and transactions as values
        """
        # Filter for specified split
        split_data = self.data[self.data['split'] == split]
        split_indices = split_data.index.tolist()
        
        split_time_series = {}
        for idx in split_indices:
            # Get transactions for this split row
            transactions = self.get_transactions_for_row(idx)
            # Sort by txn_dte and txn_tme
            if not transactions.empty:
                transactions = transactions.sort_values(['txn_dte', 'txn_tme'])
            split_time_series[idx] = transactions
        return split_time_series
    
    def get_multivariate_timeseries_for_split(self, split):
        """
        Create multivariate time series for specified split
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with row indices as keys and multivariate time series as values
        """
        split_time_series = self.get_time_series_for_split(split)
        
        split_multivariate_ts = {}
        for row_idx, transactions in split_time_series.items():
            multivariate_ts = self.create_multivariate_timeseries(transactions)
            split_multivariate_ts[row_idx] = multivariate_ts
        
        return split_multivariate_ts
    
    def get_txn_des_timeseries_for_split(self, split):
        """
        Create txn_des time series for specified split
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with row indices as keys and txn_des time series as values
        """
        split_time_series = self.get_time_series_for_split(split)
        
        split_txn_des_ts = {}
        for row_idx, transactions in split_time_series.items():
            txn_des_ts = self.create_txn_des_timeseries(transactions)
            split_txn_des_ts[row_idx] = txn_des_ts
        
        return split_txn_des_ts
    
    def get_mcc_desc_timeseries_for_split(self, split):
        """
        Create mcc_desc time series for specified split
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with row indices as keys and mcc_desc time series as values
        """
        split_time_series = self.get_time_series_for_split(split)
        
        split_mcc_desc_ts = {}
        for row_idx, transactions in split_time_series.items():
            mcc_desc_ts = self.create_mcc_desc_timeseries(transactions)
            split_mcc_desc_ts[row_idx] = mcc_desc_ts
        
        return split_mcc_desc_ts
    
    def save_multivariate_timeseries_to_dataframe(self, output_dir='data/processed_data/ts_processed_data', split='train'):
        """
        Convert multivariate time series to DataFrame and save to files
        
        Args:
            output_dir: Directory to save the processed data
            split: 'train' or 'test' to specify which data split to process
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all multivariate time series, txn_des time series, and mcc_desc time series for specified split
        all_multivariate_ts = self.get_multivariate_timeseries_for_split(split)
        all_txn_des_ts = self.get_txn_des_timeseries_for_split(split)
        all_mcc_desc_ts = self.get_mcc_desc_timeseries_for_split(split) 
        
        # Convert to DataFrame format - each row is one account
        rows = []
        for row_idx, multivariate_ts in all_multivariate_ts.items():
            # Get the original row data
            original_row = self.data.iloc[row_idx]
            
            # Convert target_delinquency boolean to int
            target_delinquency = original_row['target_delinquency']
            if isinstance(target_delinquency, (bool, np.bool_)):
                target_delinquency = 1 if target_delinquency else 0
            else:
                target_delinquency = int(target_delinquency)
            
            # Get txn_des and mcc_desc time series for this row
            txn_des_ts = all_txn_des_ts.get(row_idx, [])
            mcc_desc_ts = all_mcc_desc_ts.get(row_idx, [])
            
            row_data = {
                'act_idn_sky': original_row['act_idn_sky'],
                'time_series': multivariate_ts,  # Keep as dict, not JSON string
                'txn_des_series': txn_des_ts,  # Separate txn_des time series
                'mcc_desc_series': mcc_desc_ts,  # Separate mcc_desc time series
                'target_delinquency': target_delinquency
            }
            
            rows.append(row_data)
        
        # Save to JSONL file
        output_file = os.path.join(output_dir, f'multivariate_timeseries_{split}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        print(f"Saved multivariate time series to: {output_file}")
        print(f"Number of accounts: {len(rows)}")
        
        return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Data Reader')
    parser.add_argument('--processed_data', type=str, 
                       default='data/processed_data/sample_index/samples_min12mo_fixed_1test.feather',
                       help='Path to processed data feather file')
    parser.add_argument('--raw_data', type=str,
                       default='data/raw_data/sample_transaction.feather', 
                       help='Path to raw transaction data feather file')
    parser.add_argument('--output_dir', type=str,
                       default='data/processed_data/ts_processed_data',
                       help='Directory to save the processed time series data')
    
    args = parser.parse_args()
    
    TSData = TSDataReader(args.processed_data, args.raw_data)
    
    # TSData.data = TSData.data.sample(500, random_state=42).reset_index(drop=True)
    print(f"Sampled data shape: {TSData.data.shape}")
    
    # Save multivariate time series to DataFrame
    df = TSData.save_multivariate_timeseries_to_dataframe(
        output_dir=args.output_dir,
        split='train'
    )

    df = TSData.save_multivariate_timeseries_to_dataframe(
        output_dir=args.output_dir,
        split='test'
    )
        