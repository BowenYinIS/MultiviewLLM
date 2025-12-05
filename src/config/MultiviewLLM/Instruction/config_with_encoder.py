'''
Configuration file for Instruction tasks in MultiviewLLM.

This file contains configurations for:
    - Generating instruction datasets from sample indices. (used in src/dataset/MultiviewLLM/Instruction/generate_dataset.py)
'''



from pathlib import Path
from src.config.paths import paths
import torch


generate_dataset_config = {
    "seed": 42,

    "sample_index_path": Path(paths.processed_data_dir, 'sample_index', 'samples_min12mo_fixed_2test.feather'),
    "output_data_dir": Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction'),

    "task_mode": "Text-Graph",  # options: "Text-Graph", "Text-TimeSeries", "Graph-TimeSeries", "Text-Graph-TimeSeries", "Delinquency-Prediction"
    "negative_mode_ratio": {"random": 1, "other_act": 0, "same_act": 0},
    "negative_ratio": 3,  # number of negative samples per positive sample
    "text_graph_ts_ratio": {"graph_unmatched": 0.33, "ts_unmatched": 0.33, "both_unmatched": 0.34},  # only used when task_mode is "Text-Graph-TimeSeries", defines the ratio of output types
    "graph_ts_ratio": {"graph_unmatched": 0.5, "ts_unmatched": 0.5},  # only used when task_mode is "Graph-TimeSeries", defines the ratio of output types

    "mcc_top_k": 3,  # top k MCCs to consider for constructing recent transaction prompt
    "transition_top_k": 3,  # top k transitions to consider for constructing recent transaction prompt
}


train_config = {
    # General settings
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "seed": 42,

    # Logging settings
    "entity": "bwyin-peking-university",  # for wandb
    "project": "MultiviewLLM_v2",  # for wandb
    "run_name": "V2_Qwen2.5_7B_Instruct_12fixed",  # for wandb
    "log_interval": 10,  # steps
    "save_dir": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'V2'),
    "model_save_name": "projector_g{graph_query_num}_t{ts_query_num}_match_12mo_fixed",

    # Data settings
    "dataset_path_dict": {'profile': paths.act_info,
                          'transaction': paths.sample_transaction,
                          'index': Path(paths.processed_data_dir, 'sample_index', 'samples_min12mo_fixed_2test.feather'),
                          'graph': Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'samples_min12mo_fixed_2test_graph.pt'),
                          'ts': Path(paths.processed_data_dir, 'MultiviewLLM', 'TSEncoder', 'samples_min12mo_fixed_2test_billingcycle.jsonl'),},
    "keep_views": ['TS'],  # 'Graph', 'TS'
    'txn_amt_mean': 5.489291191101074,
    'txn_amt_std': 1.5969253778457642,
    "padding_length": 1000,  # max length for padding/truncating input sequences
    "sft_mode_strict": True,  # whether to strictly train only on

    # Model settings
    "billing_cycle_num": 12,
    ## LLM
    "backbone": "Qwen/Qwen2.5-7B-Instruct",
    "backbone_cache_dir": "/data/huggingface-cache/hub",
    "graph_query_num": 5,
    "ts_query_num": 2,
    "placeholder_num": 25,  # number of placeholder tokens to add to the tokenizer and model embeddings
    "llm_hidden_size": 1280,  # will be updated after loading the model
    ## Graph
    "graph_mcc_num": 239,
    "graph_edge_dim": None,
    "graph_layer_mode": 'GINE',
    "graph_hidden_dim": 32,
    "graph_num_layers": 2,
    "graph_mcc_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'mcc_embed.pt'),
    "graph_semantic_initial": False,
    "graph_output_dim": 64,
    "graph_checkpoint_path": Path(paths.checkpoint_dir, 'MultiviewLLM', 'GraphEncoder', 'samples_min12mo_fixed_2test_model.pth'),
    ## TS
    "ts_input_dim": 6,
    "ts_d_model": 256,
    "ts_nhead": 8,
    "ts_num_layers": 6,
    "ts_dim_feedforward": 512,
    "ts_dropout": 0.1,
    "ts_num_mcc": 13,
    "ts_num_hod": 24,
    "ts_num_dow": 7,
    "ts_num_wom": 6,
    "ts_num_moy": 12,
    "ts_checkpoint_path": Path(paths.checkpoint_dir, 'MultiviewLLM', 'TSEncoder', 'epoch_10.pt'),
    ## projector
    "projector_type": "MLP", # options: "MLP", "Attention"
    # view query num 在上方 LLM 设置中定义
    # view input dim 在各自model中定义
    # output_dim是llm hidden_size，会在加载模型后更新

    # useless in MLP projector
    # "projector_hidden_dim": 256,
    # "projector_num_heads": 8,
    # "projector_dropout": 0.1,
    # "projector_prenorm": True,
    # "projector_ffw_ratio": 2.0,

    # Training settings
    "batch_size": 8,
    "mixed_precision": "bf16",  # options: "no", "fp16", "bf16"
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "num_epochs": 1,
    "grad_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_scheduler": "cosine",  # options: "linear", "cosine"
    "max_grad_norm": 1.0,
}

train_match_config = {
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "seed": 42,
    "task_mode": "Match",

    "entity": "bwyin-peking-university",  # for wandb
    "project": "MultiviewLLM_Match",  # for wandb
    "run_name": "Match_Q12_Qwen2.5_7B_Instruct_12fixed",  # for wandb
    "log_interval": 50,  # steps

    "dataset_path_lis": [('Index', Path(paths.processed_data_dir, 'sample_index', 'samples_min12mo_fixed_2test.feather'),),
                         ('Graph', Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'samples_min6mo_fixed_2test_graph.pt')),
                         ('TS', Path(paths.processed_data_dir, 'MultiviewLLM', 'TSEncoder', 'samples_min12mo_fixed_2test.jsonl')),],
    "graph_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'samples_min12mo_fixed_2test_node_embed.pt'),
    "ts_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'TSEncoder', 'samples_min12mo_fixed_2test.npy'),
    "test_ratio": 0.2,
    "padding_length": 750,  # max length for padding/truncating input sequences
    "graph_query_num": 5,
    "ts_query_num": 12,
    "placeholder_num": 25,  # number of placeholder tokens to add to the tokenizer and model embeddings
    "save_dir": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'Match'),
    "model_save_name": "projector_g{graph_query_num}_t{ts_query_num}_match_12mo_fixed",

    "mixed_precision": "bf16",  # options: "no", "fp16", "bf16"
    "grad_accumulation_steps": 8,
    "num_epochs": 1,
    "batch_size": 8,
    "warmup_ratio": 0.03,
    "lr_projector": 1e-4,
    "lr_scheduler": "cosine",  # options: "linear", "cosine"
    "weight_decay": 1e-2,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "max_grad_norm": 1.0,

    "backbone": "Qwen/Qwen2.5-7B-Instruct",
    "backbone_cache_dir": "/data/huggingface-cache/hub",
    "load_checkpoint_path": None,  # Path to a checkpoint to load the projector weights from
    "graph_input_dim": 64,
    "ts_input_dim": 256,
    "llm_hidden_size": 1280,  # will be updated after loading the model
    "projector_hidden_dim": 256,
    "projector_num_heads": 8,
    "projector_dropout": 0.1,
    "projector_prenorm": True,
    "projector_ffw_ratio": 2.0,
}


train_delinquency_config = {
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "seed": 42,
    "task_mode": "Delinquency-Prediction",

    "entity": "bwyin-peking-university",  # for wandb
    "project": "MultiviewLLM_Delinquency_Prediction",  # for wandb
    "run_name": "Delinquency_Prediction_Qwen2.5_7B_Instruct_12fixed",  # for wandb
    "log_interval": 50,  # steps

    "dataset_path_lis": [('DP', Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction', 'samples_min12mo_fixed_2test_Delinquency-Prediction_dataset.feather')),],
    "graph_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder', 'samples_min12mo_fixed_2test_node_embed.pt'),
    "ts_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'TSEncoder', 'samples_min12mo_fixed_2test.npy'),
    "test_ratio": 0.2,
    "padding_length": 750,  # max length for padding/truncating input sequences
    "query_num": 4,
    "placeholder_num": 20,  # number of placeholder tokens to add to the tokenizer and model embeddings
    "save_dir": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'Delinquency_Prediction'),
    "model_save_name": "projector_delinquency_prediction_12mo_fixed_nm",

    "mixed_precision": "bf16",  # options: "no", "fp16", "bf16"
    "grad_accumulation_steps": 8,
    "num_epochs": 1,
    "batch_size": 8,
    "warmup_ratio": 0.05,
    "lr_projector": 3e-5,
    "lr_scheduler": "cosine",  # options: "linear", "cosine"
    "weight_decay": 1e-3,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "max_grad_norm": 0.5,

    "backbone": "Qwen/Qwen2.5-7B-Instruct",
    "backbone_cache_dir": "/data/huggingface-cache/hub",
    # "load_checkpoint_path": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'Match', 'projector_match_12mo_fixed_final_step18068.pt'),  # Path to a checkpoint to load the projector weights from
    "load_checkpoint_path": None,
    "graph_input_dim": 64,
    "ts_input_dim": 256,
    "llm_hidden_size": 1280,  # will be updated after loading the model
    "projector_hidden_dim": 256,
    "projector_num_heads": 8,
    "projector_dropout": 0.1,
    "projector_prenorm": True,
    "projector_ffw_ratio": 2.0,
}
