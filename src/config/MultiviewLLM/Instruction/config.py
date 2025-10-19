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

    "sample_index_path": Path(paths.processed_data_dir, 'sample_index', 'samples_min6mo_fixed_2test.feather'),
    "output_data_dir": Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction'),

    "task_mode": "Text-Graph",  # options: "Text-Graph", "Text-TimeSeries", "Graph-TimeSeries", "Text-Graph-TimeSeries", "Delinquency-Prediction"
    "negative_mode_ratio": {"random": 1, "other_act": 0, "same_act": 0},
    "negative_ratio": 3,  # number of negative samples per positive sample
    "text_graph_ts_ratio": {"graph_unmatched": 0.33, "ts_unmatched": 0.33, "both_unmatched": 0.34},  # only used when task_mode is "Text-Graph-TimeSeries", defines the ratio of output types
    "graph_ts_ratio": {"graph_unmatched": 0.5, "ts_unmatched": 0.5},  # only used when task_mode is "Graph-TimeSeries", defines the ratio of output types

    "mcc_top_k": 3,  # top k MCCs to consider for constructing recent transaction prompt
    "transition_top_k": 3,  # top k transitions to consider for constructing recent transaction prompt
}

train_match_config = {
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "seed": 42,
    "task_mode": "match",

    "entity": "bwyin-peking-university",  # for wandb
    "project": "MultiviewLLM_Match",  # for wandb
    "run_name": "Instruction_Projector_Qwen2-7B_Instruct_graph_ts_fixed_2test",  # for wandb
    "log_interval": 50,  # steps

    "dataset_path_lis": [('GT', Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction', 'samples_min6mo_fixed_2test_Graph-TimeSeries_dataset.feather')),
                         ('GX', Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction', 'samples_min6mo_fixed_2test_Text-Graph_dataset.feather')),
                         ('TX', Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction', 'samples_min6mo_fixed_2test_Text-TimeSeries_dataset.feather')),
                         ('TGX', Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction', 'samples_min6mo_fixed_2test_Text-Graph-TimeSeries_dataset.feather'))],
    "test_ratio": 0.2,
    "system_prompt": "你是一个有用的助手，能够根据用户的需求，结合图结构信息和时间序列信息，提供准确且有帮助的回答。",
    "padding_length": 550,  # max length for padding/truncating input sequences
    "query_num": 8,
    "graph_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder',
                             'samples_min6mo_fixed_2test_node_embed.pt'),
    "ts_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'TSEncoder',
                          'samples_min6mo_fixed_2test.npy'),

    "mixed_precision": "bf16",  # options: "no", "fp16", "bf16"
    "num_epochs": 3,
    "batch_size": 8,
    "grad_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_projector": 1e-4,
    "lr_llm": 1e-5,
    "lr_scheduler": "cosine",  # options: "linear", "cosine"
    "weight_decay": 1e-2,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "save_dir": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'match'),
    "max_grad_norm": 1.0,

    "placeholder_num": 100,  # number of placeholder tokens to add to the tokenizer and model embeddings
    "backbone": "Qwen/Qwen2-7B-Instruct",
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
    "run_name": "Instruction_Projector_Qwen2-Task",  # for wandb
    "log_interval": 50,  # steps

    "dataset_path_lis": [('DP', Path(paths.processed_data_dir, 'MultiviewLLM', 'Instruction', 'samples_min6mo_fixed_2test_Delinquency-Prediction_dataset.feather')),],
    "test_ratio": 0.2,
    "system_prompt": "你是一个有用的助手，能够根据用户的需求，结合图结构信息和时间序列信息，提供准确且有帮助的回答。",
    "padding_length": 550,  # max length for padding/truncating input sequences
    "query_num": 8,
    "graph_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'GraphEncoder',
                             'samples_min6mo_fixed_2test_node_embed.pt'),
    "ts_embed_path": Path(paths.processed_data_dir, 'MultiviewLLM', 'TSEncoder',
                          'samples_min6mo_fixed_2test.npy'),

    "mixed_precision": "bf16",  # options: "no", "fp16", "bf16"
    "num_epochs": 3,
    "batch_size": 8,
    "grad_accumulation_steps": 8,
    "warmup_ratio": 0.05,
    "lr_projector": 3e-5,
    "lr_llm": 1e-5,
    "lr_scheduler": "cosine",  # options: "linear", "cosine"
    "weight_decay": 1e-3,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "save_dir": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'Delinquency_Prediction'),
    "max_grad_norm": 0.5,

    "placeholder_num": 100,  # number of placeholder tokens to add to the tokenizer and model embeddings
    "backbone": "Qwen/Qwen2-7B-Instruct",
    "backbone_cache_dir": "/data/huggingface-cache/hub",
    "load_checkpoint_path": Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'match', 'projector_epoch1_step40082.pt'),  # Path to a checkpoint to load the projector weights from

    "graph_input_dim": 64,
    "ts_input_dim": 256,
    "llm_hidden_size": 1280,  # will be updated after loading the model
    "projector_hidden_dim": 256,
    "projector_num_heads": 8,
    "projector_dropout": 0.1,
    "projector_prenorm": True,
    "projector_ffw_ratio": 2.0,
}
