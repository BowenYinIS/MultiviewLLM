#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel


@torch.no_grad()
def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B,T,H], attention_mask: [B,T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def generate_mcc_embed(config):
    # 1) 读取 encoder.json
    with open(config["encoder_path"], "r") as f:
        enc = json.load(f)
    dec = enc["mcc_desc_decoder"]  # {idx(str或int): desc(str)}
    items = sorted([(int(k), ("" if v is None else str(v)).strip()) for k, v in dec.items()], key=lambda x: x[0])
    texts = [t for _, t in items]

    # 2) 模型与设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config["model_name"], cache_dir='/data/huggingface-cache/hub')
    mdl = AutoModel.from_pretrained(config["model_name"], cache_dir='/data/huggingface-cache/hub').to(device).eval()

    # 3) 批量编码
    enc_out = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
    enc_out = {k: v.to(device) for k, v in enc_out.items()}
    out = mdl(**enc_out).last_hidden_state                # [B,T,H]
    vec = mean_pool(out, enc_out["attention_mask"])       # [B,H]
    embs = vec.cpu()                                      # [B,H]

    # 4) 保存
    config['output_path'].parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, config["output_path"])


if __name__ == "__main__":
    # Define configuration
    from src.config.MultiviewLLM.GraphEncoder.config import generate_mcc_embed_config as config

    generate_mcc_embed(config)
