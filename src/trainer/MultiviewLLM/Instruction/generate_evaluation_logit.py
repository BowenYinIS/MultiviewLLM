import json
from pathlib import Path
import torch
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration
from src.utils.MultiviewLLM.Instruction.utils import *
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


def create_model(config, tokenizer):
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
    return model


@torch.no_grad()
def generate_on_loader_no_accel(projector,
                                language_model,
                                tokenizer,
                                test_loader,
                                save_path: Path,
                                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                gen_kwargs: dict = None,
                                use_amp: bool = True,
                                n_samples: int = 1,
                                seed_base: int = 42,
                                ):
    # 预先获取 4 个目标 token 的 id，并做单 token 断言
    ids_map = {
        "true": tokenizer.encode("true", add_special_tokens=False),
        "false": tokenizer.encode("false", add_special_tokens=False),
        "True": tokenizer.encode("True", add_special_tokens=False),
        "False": tokenizer.encode("False", add_special_tokens=False),
    }
    for k, v in ids_map.items():
        assert len(v) == 1, f"'{k}' 不是单个 token，请改用与词表对齐的写法（如在前面加空格）或实现多 token 打分。"
    ids = {k: v[0] for k, v in ids_map.items()}
    special_ids = torch.tensor(
        [ids["true"], ids["false"], ids["True"], ids["False"]],
        device=language_model.device
    )
    col_idx = {"true": 0, "false": 1, "True": 2, "False": 3}

    language_model.to(device).eval()
    projector.to(device).eval()

    # processors = LogitsProcessorList([RestrictToBoolean(tokenizer)])

    df = test_loader.dataset.data
    label_dict = df[['original_tag', 'target_delinquency']].set_index('original_tag')['target_delinquency'].to_dict()

    # 生成参数
    if gen_kwargs is None:
        gen_kwargs = dict(
            max_new_tokens=16,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            # repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id
                          if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
        )

    rows = []
    rows_token = []

    autocast_dtype = torch.bfloat16 if use_amp and torch.cuda.is_available() else None

    for batch in tqdm(test_loader, desc="[Eval] Generating"):
        # basic info
        original_tags = batch.get("original_tags", None)
        labels = [label_dict[tag] for tag in original_tags] if original_tags is not None else None
        B = len(original_tags)

        # move to device
        batch = {k: (v.to(device) if hasattr(v, "to") else v)
                 for k, v in batch.items()
                 if k != "original_tags"}

        # 1) projector 融合得到 embeds
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                embeds = projector(
                    input_ids=batch['input_ids'],
                    is_graph=batch['is_graph'],
                    is_ts=batch['is_ts'],
                    graph_x=batch['graph_x'],
                    graph_x_pad=batch['graph_x_pad'],
                    ts_x=batch['ts_x'],
                    ts_x_pad=batch['ts_x_pad'],
                )
        else:
            embeds = projector(
                input_ids=batch['input_ids'],
                is_graph=batch['is_graph'],
                is_ts=batch['is_ts'],
                graph_x=batch['graph_x'],
                graph_x_pad=batch['graph_x_pad'],
                ts_x=batch['ts_x'],
                ts_x_pad=batch['ts_x_pad'],
            )

        # sample 重复
        embeds_rep = embeds.repeat_interleave(n_samples, dim=0)
        attn_mask_rep = batch['attn_mask'].repeat_interleave(n_samples, dim=0)

        # 2) LLM.generate
        outputs = language_model.generate(
            inputs_embeds=embeds_rep,
            attention_mask=attn_mask_rep,
            output_logits=True,
            return_dict_in_generate=True,
            # logits_processor=processors,
            **gen_kwargs
        )

        seqs = outputs.sequences  # [B*n_samples, seq_len]

        # 3) 解码
        texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)

        # === 4) 组装输出 + 命中 token 分数 ===
        logits_sel = torch.stack([scores_t[:, special_ids] for scores_t in outputs.logits], dim=1)
        logprobs_sel = torch.stack(
            [F.log_softmax(scores_t.float(), dim=-1)[:, special_ids] for scores_t in outputs.logits],
            dim=1
        )
        for b in range(B):
            tag = original_tags[b]
            lab = labels[b] if labels is not None else None
            for s in range(n_samples):
                idx = b * n_samples + s  # 当前序号

                # 获取该样本本次采样的完整生成序列和对应的 scores
                gen_tokens = seqs[idx]

                # 命中位置：生成出来的是四个特殊 token 之一
                hits_mask = (
                        (gen_tokens == special_ids[0]) |
                        (gen_tokens == special_ids[1]) |
                        (gen_tokens == special_ids[2]) |
                        (gen_tokens == special_ids[3])
                )
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
                        "true_logit": float(logits_sel[idx, t, col_idx["true"]].item()),
                        "true_logprob": float(logprobs_sel[idx, t, col_idx["true"]].item()),
                        "false_logit": float(logits_sel[idx, t, col_idx["false"]].item()),
                        "false_logprob": float(logprobs_sel[idx, t, col_idx["false"]].item()),
                        "True_logit": float(logits_sel[idx, t, col_idx["True"]].item()),
                        "True_logprob": float(logprobs_sel[idx, t, col_idx["True"]].item()),
                        "False_logit": float(logits_sel[idx, t, col_idx["False"]].item()),
                        "False_logprob": float(logprobs_sel[idx, t, col_idx["False"]].item()),
                    }
                    hit_info.append(row)

                # === 记录整体输出 ===
                rows.append({
                    "original_tag": tag,
                    "label": lab,
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
    from src.config.MultiviewLLM.Instruction.config import train_delinquency_config as config

    for checkpoint_path in Path('/data/bwyin/project/MultiviewLLM/checkpoint/MultiviewLLM/Instruction/Match').glob('*_final_*.pt'):
        if checkpoint_path.stem != 'projector_g8_t8_match_12mo_fixed_final_step18068':
            continue

        graph_query_num = int(checkpoint_path.stem.split('_g')[1].split('_')[0])
        ts_query_num = int(checkpoint_path.stem.split('_t')[1].split('_')[0])
        mode = 'match_only'

        config['n_samples'] = 1
        config['batch_size'] = 128
        config['graph_query_num'] = graph_query_num
        config['ts_query_num'] = ts_query_num

        checkpoint_pt = torch.load(checkpoint_path, map_location='cpu')
        palceholder_num = (checkpoint_pt['llm_embed.weight'].shape[0] - 151665)//2
        config['placeholder_num'] = palceholder_num
        print(f'placeholder_num: {config["placeholder_num"]}')

        n_samples = config['n_samples']
        print(f'sample_num: {config["n_samples"]}, batch_size: {config["batch_size"]}')

        tokenizer = create_tokenizer(config)
        tokenizer.padding_side = 'left'

        model = create_model(config, tokenizer)
        model.projector.load_state_dict(checkpoint_pt)


        for remove_graph in [False, ]:
            for remove_ts in [False, ]:
        # for remove_graph in [False, True]:
        #     for remove_ts in [False, True]:
                train_loader, test_loader = create_dataloader(config, tokenizer,
                                                              remove_graph=remove_graph,
                                                              remove_ts=remove_ts)

                # save_name = f"{checkpoint_path.stem}"+f"_dg{remove_graph}_dt{remove_ts}"+".csv"
                save_name = f"{checkpoint_path.stem}" + f"_dg{remove_graph}_dt{remove_ts}" + f"_logit.csv"
                if Path('/data/bwyin/project/MultiviewLLM/evaluation_results', save_name).exists():
                    print(f"[Eval] Skip existing file: {save_name}")
                    continue
                generate_on_loader_no_accel(
                    projector=model.projector,
                    language_model=model.language_model,
                    tokenizer=tokenizer,
                    test_loader=test_loader,
                    save_path=Path('/data/bwyin/project/MultiviewLLM/evaluation_results', save_name),
                    device=config['device'],
                    use_amp=None,
                    n_samples=n_samples,
                )