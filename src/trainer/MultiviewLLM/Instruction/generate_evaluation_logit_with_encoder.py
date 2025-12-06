import json
from pathlib import Path
import torch
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration
from src.utils.MultiviewLLM.Instruction.utils_with_encoder import *
from src.utils.seed_everything import seed_everything
import pandas as pd
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList


class RestrictToBoolean(LogitsProcessor):
    def __init__(self, tokenizer):
        self.true_id = tokenizer.encode("true", add_special_tokens=False)[0]
        self.false_id = tokenizer.encode("false", add_special_tokens=False)[0]
        self.true_id_1 = tokenizer.encode("True", add_special_tokens=False)[0]
        self.false_id_1 = tokenizer.encode("False", add_special_tokens=False)[0]
        self.valid_ids = [self.true_id, self.false_id, self.true_id_1, self.false_id_1]

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.valid_ids] = scores[:, self.valid_ids]
        return mask


@torch.no_grad()
def generate_on_loader_no_accel(config, model, tokenizer, test_loader, save_path: Path, use_amp=None, n_samples=1, gen_kwargs=None):
    # device
    device = config['device']
    # 精度
    autocast_dtype = torch.bfloat16 if use_amp and torch.cuda.is_available() else None
    # 生成参数
    if gen_kwargs is None:
        gen_kwargs = dict(
            max_new_tokens=16,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id
                          if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
        )
    # 预先获取目标token的 id，并做单token断言
    focal_tokens = [" true", " false"]
    focal_token_ids = torch.tensor([tokenizer.encode(tok, add_special_tokens=False)[0] for tok in focal_tokens], device=device)


    rows = []
    for batch in tqdm(test_loader, desc="[Eval] Generating"):
        # basic info
        indexs = batch.get("indexs", None)
        gts = batch.get("gts", None)

        # move to device
        batch = to_device(batch, device)

        # 1) projector 融合得到 embeds
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                embeds, attn_mask, labels = model.forward_embeddings(batch)
        else:
            embeds, attn_mask, labels = model.forward_embeddings(batch)

        # 2) sample 重复
        embeds_rep = embeds.repeat_interleave(n_samples, dim=0)
        attn_mask_rep = batch['attn_mask'].repeat_interleave(n_samples, dim=0)

        # 3) LLM.generate
        outputs = model.language_model.generate(
            inputs_embeds=embeds_rep,
            attention_mask=attn_mask_rep,
            output_logits=True,
            return_dict_in_generate=True,
            # logits_processor=processors,
            **gen_kwargs
        )

        # 4) 解码
        seqs = outputs.sequences  # [B*n_samples, seq_len]
        texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)  # list of str, len B*n_samples
        logits_sel = torch.stack([scores_t[:, focal_token_ids] for scores_t in outputs.logits], dim=1)  # [B*n_samples, seq_len, 2]
        logprobs_sel = torch.stack(
            [F.log_softmax(scores_t.float(), dim=-1)[:, focal_token_ids] for scores_t in outputs.logits],
            dim=1
        )  # [B*n_samples, seq_len, 2]

        # 5) 组装输出
        for b in range(len(indexs)):
            tag = indexs[b]
            gt = gts[b]
            for s in range(n_samples):
                idx = b * n_samples + s  # 当前序号

                # 获取该样本本次采样的完整生成序列和对应的 scores
                gen_tokens = seqs[idx]

                # 命中位置：生成出来的是特殊 token 之一
                hits_mask = torch.isin(gen_tokens, focal_token_ids)  # shape [seq_len], bool
                hit_steps = hits_mask.nonzero(as_tuple=False).squeeze(-1).tolist()  # e.g., [3, 7]

                hit_info = []
                for t in hit_steps:
                    picked_id = int(gen_tokens[t].item())
                    picked_str = tokenizer.decode([picked_id], skip_special_tokens=False)

                    # 读取该样本该步的四个特殊 token 的 logit/logprob
                    # logits_sel[idx, t] 形状是 [4]，依次对应 true/false/True/False
                    row = {
                        "step_idx": t,
                        "picked_token_id": picked_id,
                        "picked_token_str": picked_str,
                    }
                    for i, tok in enumerate(focal_tokens):
                        row[f"logit_{tok.strip()}"] = logits_sel[idx, t, i].item()
                        row[f"logprob_{tok.strip()}"] = logprobs_sel[idx, t, i].item()
                    hit_info.append(row)

                # === 记录整体输出 ===
                rows.append({
                    "original_tag": tag.item(),
                    "label": gt.item(),
                    "sample_id": s,
                    "generation": texts[idx],
                    "hit_steps": hit_steps,  # 命中的步索引（可能多个）
                    "hit_info": hit_info,  # 每个命中的详细分数信息
                })

    # 5) 写盘
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)
    print(f"[Eval] Saved {len(rows)} generations to: {save_path}")


if __name__ == '__main__':
    from src.config.paths import paths
    from src.config.MultiviewLLM.Instruction.config_with_encoder import train_config as config

    exp_id = 2
    for checkpoint_path in Path(paths.checkpoint_dir, 'MultiviewLLM', 'Instruction', 'V2').glob('*_final_*.pt'):
        if f'exp{exp_id}' not in checkpoint_path.stem:
            continue

        # update config
        graph_query_num = int(checkpoint_path.stem.split('_g')[1].split('_')[0])
        ts_query_num = int(checkpoint_path.stem.split('_t')[1].split('_')[0])
        config['n_samples'] = 1
        config['batch_size'] = 1
        config['graph_query_num'] = graph_query_num
        config['ts_query_num'] = ts_query_num

        # tokenizer
        tokenizer = create_tokenizer(config)
        # tokenizer.padding_side = 'left'

        # dataloader
        train_loader, test_loader = create_dataloader(config, tokenizer)

        # model
        model = create_model_and_optimizer(config, tokenizer, train_loader)[0]
        checkpoint_pt = torch.load(checkpoint_path, map_location='cpu')
        model.graph_model.load_state_dict(checkpoint_pt['graph_model'])
        model.ts_model.load_state_dict(checkpoint_pt['ts_model'])
        model.projector.load_state_dict(checkpoint_pt['projector'])
        model = model.to(config['device']).eval()

        save_name = f"{checkpoint_path.stem}" + f"_logit.json"
        if Path('/data/bwyin/project/MultiviewLLM/evaluation_results1', save_name).exists():
            print(f"[Eval] Skip existing file: {save_name}")
            continue
        else:
            generate_on_loader_no_accel(
                config=config,
                model=model,
                tokenizer=tokenizer,
                test_loader=test_loader,
                save_path=Path('/data/bwyin/project/MultiviewLLM/evaluation_results', save_name),
                use_amp=None,
                n_samples=config['n_samples'],
            )