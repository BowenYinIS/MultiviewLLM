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


@torch.no_grad()
def generate_on_loader_no_accel(projector,
                                language_model,
                                tokenizer,
                                test_loader,
                                save_path: Path,
                                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                gen_kwargs: dict = None,
                                use_amp: bool = True):
    """
    使用原生 PyTorch，在 test_loader 上进行生成并保存 JSONL。
    要求 batch 至少包含：
      input_ids, attn_mask, is_graph, is_ts, graph_x, graph_x_pad, ts_x, ts_x_pad
    可选：sample_id（若无则使用递增id）
    """
    language_model.to(device).eval()
    projector.to(device).eval()

    df = test_loader.dataset.data
    label_dict = df[['original_tag', 'target_delinquency']].set_index('original_tag')['target_delinquency'].to_dict()


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

    results = []
    sid_counter = 0

    autocast_dtype = torch.bfloat16 if use_amp and torch.cuda.is_available() else None

    for batch in tqdm(test_loader, desc="[Eval] Generating"):
        # 移到设备；过滤无关键
        original_tags = batch.get("original_tags", None)
        labels = [label_dict[tag] for tag in original_tags] if original_tags is not None else None
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

        # 2) LLM.generate
        gen_ids = language_model.generate(
            inputs_embeds=embeds,
            attention_mask=batch['attn_mask'],
            **gen_kwargs
        )

        # 3) 解码
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # 4) 组装输出
        result_batch = list(zip(texts, labels))
        results.extend(result_batch)

    # 5) 写盘
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results, columns=['generated_text', 'true_label'])
    df_results.to_csv(save_path.with_suffix('.csv'), index=False, encoding='utf-8')

    print(f"[Eval] Saved {len(results)} generations to: {save_path}")
    return results


def main(config, save_name, remove_graph=False, remove_ts=False):
    tokenizer = create_tokenizer(config)
    train_loader, test_loader = create_dataloader(config, tokenizer, remove_graph=remove_graph, remove_ts=remove_ts)
    for batch in test_loader:
        print(f"Test batch keys: {batch.keys()}")
        break

    model, optimizer, warmup_scheduler = create_model_and_optimizer(config, tokenizer, train_loader)

    model.projector.load_state_dict(torch.load(config['load_checkpoint_path'], map_location='cpu'))
    print(f"[Load] Loaded projector checkpoint from {config['load_checkpoint_path']}")

    # 使用原生 PyTorch 进行生成
    generate_on_loader_no_accel(
        projector=model.projector,
        language_model=model.language_model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        save_path=Path('/data/bwyin/project/MultiviewLLM/evaluation_results', save_name),
        device=config['device'],
        use_amp=None,
    )


if __name__ == '__main__':
    from src.config.MultiviewLLM.Instruction.config import train_delinquency_config as config

    match_only = Path('/data/bwyin/project/MultiviewLLM/checkpoint/MultiviewLLM/Instruction/Match/projector_match_12mo_fixed_final_step18068.pt')
    sft_w_match = Path('/data/bwyin/project/MultiviewLLM/checkpoint/MultiviewLLM/Instruction/Delinquency_Prediction/projector_delinquency_prediction_12mo_fixed_m_final_step1076.pt')
    sft_wo_match = Path('/data/bwyin/project/MultiviewLLM/checkpoint/MultiviewLLM/Instruction/Delinquency_Prediction/projector_delinquency_prediction_12mo_fixed_nm_final_step1076.pt')
    config['batch_size'] = 256

    used_ckpt = 'match_only'  # options: 'match_only', 'sft_w_match', 'sft_wo_match'
    remove_graph = True
    remove_ts = False

    used_ckpt_tag = {'match_only': match_only, 'sft_w_match': sft_w_match, 'sft_wo_match': sft_wo_match}
    config['load_checkpoint_path'] = used_ckpt_tag[used_ckpt]

    save_name = f'{used_ckpt}_12mo_fixed_evaluation_results_dg{remove_graph}_dt{remove_ts}.csv'
    print(f"{used_ckpt}  |  dg:{remove_graph}  |  dt:{remove_ts}")
    main(config, save_name, remove_graph=remove_graph, remove_ts=remove_ts)