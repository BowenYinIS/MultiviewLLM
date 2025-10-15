'''
Configuration for graph encoder tasks in MultiviewLLM.

This file contains configurations for:
    - Generating graph datasets from sample indices. (used in src/data/MultiviewLLM/GraphEncoder/generate_graph_dataset.py)
    - Generating MCC embeddings using a pre-trained language model. (used in src/data/MultiviewLLM/GraphEncoder/generate_mcc_embed.py)
    - Training the graph encoder model.  (used in src/trainer/MultiviewLLM/GraphEncoder/train_encoder.py)
'''


from pathlib import Path
from src.config.paths import paths
import torch


generate_dataset_config = {
    "sample_index_path": Path(paths.processed_data_dir, 'sample_index', 'samples_min6mo_allprevious_2test.feather'),
    "encoder_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'encoder.json'),
    "normalize_edge_weight": True,
    "output_data_dir": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder'),
}


generate_mcc_embed_config = {
    "model_name": "bert-base-chinese",
    "encoder_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'encoder.json'),
    "output_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'mcc_embed.pt'),
}


train_config = {
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "seed": 42,
    "encoder_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'encoder.json'),
    "dataset_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'samples_min12mo_fixed_2test_graph.pt'),
    "mcc_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'mcc_embed.pt'),
    "model_save_path": Path(paths.checkpoint_dir, 'MultiviewLLM', 'GraphEncoder'),
    "embed_save_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder'),

    "writer_path": Path(paths.tensorboard_log_dir, 'MultiviewLLM', 'GraphEncoder'),  # for TensorBoard
    "entity": "bwyin-peking-university",  # for wandb
    "project": "MultiviewLLM_GraphEncoder",  # for wandb
    "batch_size": 128,
    "epochs": 20,
    "learning_rate": 0.01,
    "semantic_initial": False,  # Set False is better for this task
    "semantic_contrastive": True,  # Set True is better for this task

    "layer_mode": 'GINE',  # GINE, SAGE; Typically, SAGE is better for this task. But GINE achieves the best result.
    "use_edge_attr": False,  # Set False is better for this task
    "hidden_dim": 32,
    "num_layers": 2,

    "feature_mask_prob": 0.1,
    "node_drop_prob": 0.1,
    "edge_drop_prob": 0.1,
    "rws_start_size": 100,  # Due to unknown bug, we do not use RWSampling currently
    "rws_walk_length": 10,  # Due to unknown bug, we do not use RWSampling currently
}