import torch
from torch.utils.data import DataLoader
from src.dataset.MultiviewLLM.Instruction.dataset import InstructionDataset
from src.model.MultiviewLLM.Instruction.projector import Projector
from src.model.MultiviewLLM.Instruction.language_model import MultiviewLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import pandas as pd
import pickle
import numpy as np


def print_mem(prefix=""):
    print(f"\n[{prefix}]")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    print(f"Max Reserved : {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")


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


def stack_embed_dict(embed_dict):
    keys = sorted(list(embed_dict.keys()))
    seqs = [embed_dict[k] for k in keys]

    d = seqs[0].size(-1)
    L = max(t.size(0) for t in seqs)
    N = max(keys) + 2  # 假设 keys 是从 0 开始的连续整数

    bank = torch.zeros(N, L, d)
    pad = torch.zeros(N, L, dtype=torch.bool)

    for k in keys:
        t = embed_dict[k]
        l = t.size(0)
        bank[k, :l] = t
        pad[k, :l] = 1  # 1=keep, 0=pad
    pad[N-1, :] = 1  # 防止attn_mask全0报错
    return bank, pad, N-1


def load_graph_and_ts_embed(config):
    # with open(config['ts_embed_path'], 'rb') as f:
    #     ts_embed = pickle.load(f)
    # ts_embed = {k: torch.tensor(v, dtype=torch.float32) for k, v in ts_embed.items()}
    ts_embed = np.load(config['ts_embed_path'], allow_pickle=True).item()
    ts_embed = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in ts_embed.items()}

    graph_embed = torch.load(config['graph_embed_path'])
    graph_embed = {int(k): v for k, v in graph_embed.items()}

    g_bank, g_pad, g_pad_index = stack_embed_dict(graph_embed)
    ts_bank, ts_pad, ts_pad_index = stack_embed_dict(ts_embed)
    return g_bank, g_pad, g_pad_index, ts_bank, ts_pad, ts_pad_index


def load_and_split_data(data_path_lis, task_mode, test_ratio):
    # load and concatenate data
    data_lis = []
    for tag, file in data_path_lis:
        data = pd.read_feather(file)
        data['original_tag'] = data.index.astype(str) + f"_{tag}"
        data_lis.append(data)
    data = pd.concat(data_lis, ignore_index=True)
    data = data.reset_index(drop=True)

    # process data
    if "graph_index" in data.columns:
        data["graph_index"] = data["graph_index"].fillna(-1).astype(int)
    else:
        data["graph_index"] = -1
    if "ts_index" in data.columns:
        data["ts_index"] = data["ts_index"].fillna(-1).astype(int)
    else:
        data["ts_index"] = -1

    # split data
    if task_mode != "Delinquency-Prediction":
        data = data.sample(frac=1).reset_index(drop=True)
        test_size = int(len(data) * test_ratio)
        test_data = data.iloc[:test_size].reset_index(drop=True)
        train_data = data.iloc[test_size:].reset_index(drop=True)
        return train_data, test_data
    else:
        train_data = data[data['split'] == 'train'].reset_index(drop=True)
        test_data = data[data['split'] == 'test'].reset_index(drop=True)
        return train_data, test_data


def create_dataloader(config, tokenizer, remove_graph=False, remove_ts=False):
    g_bank, g_pad, g_pad_index, ts_bank, ts_pad, ts_pad_index = load_graph_and_ts_embed(config)

    train_data, test_data = load_and_split_data(config['dataset_path_lis'],
                                                config['task_mode'],
                                                config['test_ratio'])

    train_dataset = InstructionDataset(tokenizer, config, train_data,
                                       g_bank=g_bank,
                                       g_pad=g_pad,
                                       g_pad_index=g_pad_index,
                                       ts_bank=ts_bank,
                                       ts_pad=ts_pad,
                                       ts_pad_index=ts_pad_index)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  drop_last=True, pin_memory=True, shuffle=True,
                                  collate_fn=train_dataset.collate_fn)

    test_dataset = InstructionDataset(tokenizer, config, test_data,
                                      g_bank=g_bank,
                                      g_pad=g_pad,
                                      g_pad_index=g_pad_index,
                                      ts_bank=ts_bank,
                                      ts_pad=ts_pad,
                                      ts_pad_index=ts_pad_index,
                                      is_test=True,
                                      remove_graph=remove_graph,
                                      remove_ts=remove_ts)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                 drop_last=False, pin_memory=True, shuffle=False,
                                 collate_fn=test_dataset.collate_fn)
    return train_dataloader, test_dataloader


def create_model_and_optimizer(config, tokenizer, dataloader):
    language_model = AutoModelForCausalLM.from_pretrained(
        config['backbone'],
        cache_dir=config['backbone_cache_dir'],
        dtype=torch.bfloat16,
        use_cache=False,
    )
    language_model.resize_token_embeddings(len(tokenizer))
    for param in language_model.parameters():
        param.requires_grad = False
    # # if you want use gradient checkpointing, you must set model to train mode. If so, you should disable dropout
    # for m in language_model.modules():
    #     if isinstance(m, torch.nn.Dropout):
    #         m.p = 0.0
    # language_model.gradient_checkpointing_enable()
    # # if you don't use gradient checkpointing, you can set model to eval mode
    language_model.eval()
    # language_model.config.use_cache = False
    # language_model.enable_input_require_grads()

    config['llm_hidden_size'] = language_model.config.hidden_size
    projector = Projector(
        graph_query_num=config['graph_query_num'],
        ts_query_num=config['ts_query_num'],
        graph_input_dim=config['graph_input_dim'],
        ts_input_dim=config['ts_input_dim'],
        hidden_dim=config['projector_hidden_dim'],
        output_dim=config['llm_hidden_size'],
        num_heads=config['projector_num_heads'],
        dropout=config['projector_dropout'],
        prenorm=config['projector_prenorm'],
        ffw_ratio=config['projector_ffw_ratio'],
        llm_embed=language_model.get_input_embeddings(),
    )

    model = MultiviewLLM(language_model, projector)

    param_groups = []
    trainable_params = [p for p in model.projector.parameters() if p.requires_grad]
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
