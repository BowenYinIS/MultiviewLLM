'''
生成指令数据集
'''


import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.config.paths import paths
from torch_geometric.data import Data


class GenerateGraphDataset:
    def __init__(self, config: dict):
        """
        Initialize the dataset builder with configuration and load source data.

        Args:
            config (dict): Configuration dictionary containing:
                - sample_index_path (str): Path to the sample index file
                - encoder_path (str): Path to save/load the mcc_cde and mcc_desc encoder
                - normalize_edge_weight (bool): Whether to normalize edge weights
                - output_data_dir (str): Directory path for saving the dataset
        """
        # store configuration
        self.config = config
        self.sample_index_path = config["sample_index_path"]
        self.encoder_path = config["encoder_path"]
        self.normalize_edge_weight = config.get("normalize_edge_weight", True)
        self.output_data_dir = config["output_data_dir"]

        # load source data
        self.transaction = pd.read_feather(paths.sample_transaction)
        self.sample_index = pd.read_feather(self.sample_index_path)

        # preprocess transaction data
        self._preprocess_data()

    def _preprocess_data(self):
        '''
        Preprocess the transaction data by handling missing values, generating datetime columns,
        sorting, and encoding mcc_cde.

        Preprocess the sample index data by adding an index column.
        '''
        # fill missing mcc_cde with 9999 and mcc_13cat with 'Insurance'
        self.transaction['mcc_cde'] = self.transaction['mcc_cde'].fillna(9999)
        self.transaction['mcc_13cat'] = self.transaction['mcc_13cat'].fillna('Insurance')
        self.transaction.loc[self.transaction['mcc_cde'] == 9999, 'mcc_desc'] = '保险扣款'
        self.transaction['mcc_cde'] = self.transaction['mcc_cde'].astype(str)

        # generate txn_dte_tme and sort
        self.transaction['txn_dte_tme'] = self.transaction['txn_dte'].astype(str) + ' ' + self.transaction['txn_tme'].astype(str)
        self.transaction['txn_dte_tme'] = pd.to_datetime(self.transaction['txn_dte_tme'], format='%Y-%m-%d %H:%M:%S')
        self.transaction = self.transaction.sort_values(by=['act_idn_sky', 'txn_dte_tme'])

        # encode mcc_cde and generate next_mcc_cde_index
        self._create_encoder()
        self.transaction['mcc_cde_index'] = self.transaction['mcc_cde'].map(self.mcc_cde_encoder)
        self.transaction['next_mcc_cde_index'] = self.transaction.groupby("act_idn_sky")['mcc_cde_index'].shift(-1)
        self.transaction = self.transaction.dropna(subset=['next_mcc_cde_index'])
        self.transaction['next_mcc_cde_index'] = self.transaction['next_mcc_cde_index'].astype(int)

        # add index column to sample_index
        self.sample_index['index'] = self.sample_index.index

    def _create_encoder(self):
        '''
        Create or load the mcc_cde and mcc_desc encoder from the specified path.
        '''
        if self.encoder_path.exists():
            with open(self.encoder_path, 'r') as f:
                self.encoder = json.load(f)
            self.mcc_cde_encoder = self.encoder['mcc_cde_encoder']
            self.mcc_cde_decoder = self.encoder['mcc_cde_decoder']
            self.mcc_desc_decoder = self.encoder['mcc_desc_decoder']
        else:
            mcc_cde_list = self.transaction['mcc_cde'].unique().tolist()
            self.mcc_cde_encoder = {mcc: idx for idx, mcc in enumerate(mcc_cde_list)}
            self.mcc_cde_decoder = {idx: mcc for mcc, idx in self.mcc_cde_encoder.items()}

            mcc_cde2desc = self.transaction[['mcc_cde', 'mcc_desc']].drop_duplicates().set_index('mcc_cde')['mcc_desc'].to_dict()
            self.mcc_desc_decoder = {idx: mcc_cde2desc[mcc] for mcc, idx in self.mcc_cde_encoder.items()}

            self.encoder = {
                'mcc_cde_encoder': self.mcc_cde_encoder,
                'mcc_cde_decoder': self.mcc_cde_decoder,
                'mcc_desc_decoder': self.mcc_desc_decoder,
            }

    def _bulid_one(self, act_idn_sky, billing_dates, index, label):
        '''
        Build a graph data object for one sample based on act_idn_sky and billing_dates.

        Args:
            act_idn_sky (str): The account ID.
            billing_dates (list): List of billing dates in the window.
            index (int): The index of the sample in the sample index.
            label (int): The target delinquency label for the sample.

        Returns:
            graph (torch_geometric.data.Data): The constructed graph data object.
                - x: Node features (mcc_cde_index)
                - edge_index: Edge indices
                - edge_attr: Edge weights
                - meta_info: Metadata including index and act_idn_sky
                - y: Target label
        '''
        # filter transaction data for the specific account and billing dates
        working_data = self.transaction[
            (self.transaction['act_idn_sky'] == act_idn_sky) &
            (self.transaction['billing_date'].isin(billing_dates))
        ].copy()

        # construct graph
        nodes_list = list(set(working_data['mcc_cde_index'].tolist() + working_data['next_mcc_cde_index'].tolist()))
        node_encoder = {node: idx for idx, node in enumerate(nodes_list)}
        working_data['src'] = working_data['mcc_cde_index'].map(node_encoder)
        working_data['tgt'] = working_data['next_mcc_cde_index'].map(node_encoder)
        edge_attr = working_data.groupby(['src', 'tgt']).size().reset_index(name='weight')
        edge_index = edge_attr[['src', 'tgt']].values.T

        if self.normalize_edge_weight:
            out_degree = edge_attr.groupby('src')['weight'].sum().to_dict()
            edge_weight = [w / out_degree[src] for src, w in zip(edge_attr['src'], edge_attr['weight'])]
        else:
            edge_weight = edge_attr['weight'].values

        graph = Data(
            x=torch.tensor([nodes_list], dtype=torch.long).T,  # Node features (2, mcc_cde_index)
            edge_index=torch.tensor(edge_index, dtype=torch.long),  # Edge indices (2, num_edges)
            edge_attr=torch.tensor([edge_weight], dtype=torch.float).T,  # Edge weights (num_edges, 1)
            meta_info={'index': index, 'act_idn_sky': act_idn_sky},
            y=torch.tensor([label], dtype=torch.long)  # Target label (1, )
        )
        return graph

    def build(self):
        '''
        Build the entire dataset by iterating over the sample index and constructing graph data objects.
        '''
        dataset = {}
        splits = self.sample_index['split'].unique()

        for split in splits:
            print("Building dataset for split:", split)
            dataset_split = []
            sample_index_split = self.sample_index[self.sample_index['split'] == split]
            for _, row in tqdm(sample_index_split.iterrows(), total=sample_index_split.shape[0]):
                act_idn_sky = row['act_idn_sky']
                billing_dates = row['billing_dates']
                index = row['index']
                label = row['target_delinquency']

                graph_data = self._bulid_one(act_idn_sky, billing_dates, index, label)
                dataset_split.append(graph_data)
            dataset[split] = dataset_split

        # save dataset and encoder
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, self.output_data_dir / self.sample_index_path.name.replace('.feather', '_graph.pt'))

        if not self.encoder_path.exists():
            with open(self.encoder_path, 'w') as f:
                json.dump(self.encoder, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # Define configuration
    from src.config.MultiviewLLM.Instruction.config import generate_dataset_config as config

    for file in Path('/data/bwyin/project/MultiviewLLM/processed_data/sample_index').glob('samples_*.feather'):
        # Update sample index path in config
        config['sample_index_path'] = file
        print(f"Processing sample index: {file.name}")
        # Generate dataset
        generator = GenerateGraphDataset(config)
        generator.build()