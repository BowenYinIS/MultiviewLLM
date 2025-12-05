import torch
from torch.utils.data import DataLoader
from src.dataset.MultiviewLLM.Instruction.dataset_with_encoder import InstructionDataset
from src.model.MultiviewLLM.Instruction.projector_with_encoder import Projector
from src.model.MultiviewLLM.Instruction.language_model_with_encoder import MultiviewLLM
from src.model.MultiviewLLM.GraphEncoder.model import EncoderLearner
from src.model.MultiviewLLM.TSModel.model import TimeSeriesTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import pandas as pd
from pathlib import Path
from src.config.paths import paths
from torch_geometric.data import Batch, Data
import pickle
import numpy as np
import json


def create_tokenizer(config):
    # Create tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(config['backbone'], cache_dir=config['backbone_cache_dir'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    g_special = [f"<G_PLACEHOLDER {i}>" for i in range(1, 1+config['placeholder_num'])]
    ts_special = [f"<TS_PLACEHOLDER {i}>" for i in range(1, 1+config['placeholder_num'])]
    all_specials = {"additional_special_tokens": g_special + ts_special}
    tokenizer.add_special_tokens(all_specials)
    return tokenizer


def load_all_data(data_path_dict):
    if 'profile' in data_path_dict:
        profile_data = pd.read_feather(data_path_dict['profile'])
    if 'transaction' in data_path_dict:
        transaction_data = pd.read_feather(data_path_dict['transaction'])
    if 'index' in data_path_dict:
        sample_index = pd.read_feather(data_path_dict['index'])
        sample_index['index'] = sample_index.index
    if 'graph' in data_path_dict:
        graph_data = torch.load(data_path_dict['graph'], weights_only=False)
        graph_data = graph_data['train'] + graph_data['test']
        graph_data = sorted(graph_data, key=lambda x: int(x.meta_info['index']))
    if 'ts' in data_path_dict:
        with open(data_path_dict['ts'], 'r', encoding='utf-8') as f:
            ts_data = [json.loads(line) for line in f]
    data_dict = {'profile_data': profile_data,
                 'transaction_data': transaction_data,
                 'index_data': sample_index,
                 'graph_data': graph_data,
                 'ts_data': ts_data}
    return data_dict


def create_dataloader(config, tokenizer):
    # load original data
    data_dict = load_all_data(config['dataset_path_dict'])

    # split data
    train_dataset = InstructionDataset(data_dict, config, tokenizer, is_test=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  drop_last=True, pin_memory=True, shuffle=True,
                                  collate_fn=train_dataset.collate_fn)

    test_dataset = InstructionDataset(data_dict, config, tokenizer, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                 drop_last=False, pin_memory=True, shuffle=False,
                                 collate_fn=test_dataset.collate_fn)
    return train_dataloader, test_dataloader


def create_model_and_optimizer(config, tokenizer, dataloader):
    # device = config['device']

    # graph model
    mcc_embed = torch.load(config['graph_mcc_embed_path'])
    graph_model = EncoderLearner(mcc_num=config['graph_mcc_num'],
                                 edge_dim=config['graph_edge_dim'],
                                 layer_mode=config['graph_layer_mode'],
                                 hidden_dim=config['graph_hidden_dim'],
                                 num_layers=config['graph_num_layers'],
                                 augmentor=(None, None),
                                 mcc_embed=mcc_embed,
                                 semantic_initial=config["graph_semantic_initial"])
    graph_model.load_state_dict(torch.load(config['graph_checkpoint_path']))
    # graph_model = graph_model.to(device)

    # ts model
    ts_model = TimeSeriesTransformer(
        input_dim=config['ts_input_dim'],
        d_model=config['ts_d_model'],
        nhead=config['ts_nhead'],
        num_layers=config['ts_num_layers'],
        dim_feedforward=config['ts_dim_feedforward'],
        dropout=config['ts_dropout'],
        num_mcc=config['ts_num_mcc'],
        num_hod=config['ts_num_hod'],
        num_dow=config['ts_num_dow'],
        num_wom=config['ts_num_wom'],
        num_moy=config['ts_num_moy']
    )
    # ts_model.load_state_dict(torch.load(Path(paths.checkpoint_dir, 'MultiviewLLM', 'TSEncoder', 'samples_min12mo_fixed_2test_model.pth')))
    # ts_model = ts_model.to(device)

    # llm backbone
    language_model = AutoModelForCausalLM.from_pretrained(
        config['backbone'],
        cache_dir=config['backbone_cache_dir'],
        dtype=torch.bfloat16,
        use_cache=False,
    )
    language_model.resize_token_embeddings(len(tokenizer))
    for param in language_model.parameters():
        param.requires_grad = False
    language_model.eval()

    # projector
    config['llm_hidden_size'] = language_model.config.hidden_size
    projector = Projector(
        graph_query_num=config['graph_query_num'],
        ts_query_num=config['ts_query_num'],
        graph_input_dim=config['graph_output_dim'],
        ts_input_dim=config['ts_d_model'],
        output_dim=config['llm_hidden_size'],
        llm_embed=language_model.get_input_embeddings().weight,
    )

    model = MultiviewLLM(graph_model, ts_model, language_model, projector)
    model = model.to(config['device'])

    param_groups = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) > 0:
        param_groups.append({"params": trainable_params, "lr": config['lr_projector'], "weight_decay": config['weight_decay']})

    # language_model = language_model.to(config['device'])  # if use accelerator, move later
    # projector = projector.to(config['device'])  # if use accelerator, move later

    # proj_params = [p for p in projector.parameters() if p.requires_grad]
    # param_groups = []
    # if len(proj_params) > 0:
    #     param_groups.append({"params": proj_params, "lr": config['lr_projector'], "weight_decay": config['weight_decay']})

    optimizer = torch.optim.AdamW(
        param_groups,
        eps=config['adam_eps'],
        betas=(config['adam_beta1'], config['adam_beta2'])
    )

    num_epochs = config['num_epochs']
    grad_accumulation_steps = config['grad_accumulation_steps']
    total_steps = (len(dataloader) // grad_accumulation_steps) * num_epochs
    num_warmup_steps = int(total_steps * config['warmup_ratio'])

    scheduler = get_scheduler(
        name=config['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    return model, optimizer, scheduler


def to_device(obj, device):
    """
    递归地把 obj 里的所有 torch.Tensor 移到指定 device 上。
    支持：dict / list / tuple / tensor 本身
    其他类型保持不变。
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    # dict: 递归处理 value
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}

    # list: 递归处理每个元素
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]

    # tuple: 递归处理每个元素，并转回 tuple
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)

    # torch_geometric Batch 对象
    # --- 2. PyG Data or Batch ---
    if isinstance(obj, (Data, Batch)):
        return obj.to(device)

    # 其他类型（比如 int/float/str/None）直接返回
    return obj

