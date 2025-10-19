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
            max_new_tokens=64,
            do_sample=False,          # 评测通常不采样
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
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


def main(config):
    tokenizer = create_tokenizer(config)
    train_loader, test_loader = create_dataloader(config, tokenizer)
    projector, language_model, optimizer, warmup_scheduler = create_model_and_optimizer(config, tokenizer, train_loader)

    projector.load_state_dict(torch.load(config['load_checkpoint_path'], map_location='cpu'))
    print(f"[Load] Loaded projector checkpoint from {config['load_checkpoint_path']}")

    # 使用原生 PyTorch 进行生成
    generate_on_loader_no_accel(
        projector=projector,
        language_model=language_model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        save_path=Path(config['save_dir'], 'evaluation_results.jsonl'),
        device=config['device'],
        use_amp=(config.get('mixed_precision','no') in ['fp16','bf16']),
    )


if __name__ == '__main__':
    from src.config.MultiviewLLM.Instruction.config import train_delinquency_config as config

    # 使用match后的权重直接训练
    # main(config)

    # 使用fine-tune后的权重训练
    config['load_checkpoint_path'] = Path('/data/bwyin/project/MultiviewLLM/checkpoint/MultiviewLLM/Instruction/Delinquency_Prediction/projector_final_step7515.pt')
    config['batch_size'] = 512
    main(config)