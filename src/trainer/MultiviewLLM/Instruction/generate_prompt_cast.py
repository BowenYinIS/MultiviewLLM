import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm


def generate_for_df(df, batch_size=8, temperature=0.7, max_new_tokens=16):
    """
    df: 包含 sys_msg, user_msg 两列
    返回：在 df 上加一列 'model_output'
    """
    outputs = []

    # 按 batch 处理，避免一次性太大
    for i in tqdm(range(0, len(df), batch_size), desc="Generating"):
        # print(f"Processing samples {i} to {min(i + batch_size, len(df))}...")
        batch = df.iloc[i:i + batch_size]

        # 2. 把每一行转成 chat 格式，然后用 apply_chat_template
        conversations = []
        for _, row in batch.iterrows():
            sys_msg = str(row["sys_msg"])
            user_msg = str(row["user_msg"])
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ]
            conversations.append(messages)

        # 3. 使用 chat_template 生成文本输入
        #   tokenize=True + return_tensors="pt" => 直接得到 input_ids
        inputs = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,   # 让模型知道后面要生成assistant回复
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,              # 视情况调整
        ).to(model.device)
        inputs_embeds = model.get_input_embeddings()(inputs)
        attention_mask = (inputs != tokenizer.pad_token_id).long()

        # 4. 生成
        # gen_ids = model.generate(
        #     inputs,
        #     do_sample=True,
        #     temperature=temperature,
        #     top_p=0.8,
        #     repetition_penalty=1.05,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.pad_token_id,
        # )
        gen_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=0.8,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 5. 只取新生成的部分
        #    原始输入长度可能不同，用 input_ids.shape[1] 来截取
        input_len = inputs.shape[1]
        # new_tokens = gen_ids[:, input_len:]
        new_tokens = gen_ids

        batch_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend([t.strip() for t in batch_texts])

    df = df.copy()
    df["model_output"] = outputs
    return df


def get_original_index(df_pc):
    '''To retrieve the original sample index from PromptCast output dataframe.'''
    sample_index = pd.read_feather(Path(paths.processed_data_dir, 'sample_index', 'samples_min12mo_fixed_2test.feather'))
    sample_index['index'] = sample_index.index

    df_pc['pc_tag'] = df_pc.apply(lambda row: f"{row['act_idn_sky']}_{row['billing_dates'][0]}", axis=1)
    sample_index['pc_tag'] = sample_index.apply(lambda row: f"{row['act_idn_sky']}_{row['billing_dates'][0]}", axis=1)

    pc_tag_to_index = dict(zip(sample_index['pc_tag'], sample_index['index']))

    df_pc['original_tag'] = df_pc['pc_tag'].map(pc_tag_to_index)

    df_pc = df_pc.drop(columns=['pc_tag'])
    return df_pc


if __name__ == '__main__':
    import pandas as pd
    from src.config.paths import paths
    from src.config.MultiviewLLM.Instruction.config_with_encoder import train_config as config

    df_promptcast = pd.read_feather(r'/home/bwyin/project/Agent/MultiviewLLM/src/temp/preds_promptcast_summary_1_20251204_181833.feather')
    df_promptcast = get_original_index(df_promptcast)

    # 1. 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(config['backbone'], cache_dir=config['backbone_cache_dir'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        config['backbone'],
        cache_dir=config['backbone_cache_dir'],
        dtype=torch.bfloat16,
        use_cache=False,
    ).to(config['device'])

    df_with_answer = generate_for_df(df_promptcast, batch_size=128, temperature=0.7, max_new_tokens=16)
    df_with_answer.to_feather(r'/home/bwyin/project/Agent/MultiviewLLM/src/temp/my_transformer_promptcast_em.feather')