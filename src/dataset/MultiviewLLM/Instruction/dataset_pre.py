from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import json


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, config, data, g_bank=None, g_pad=None, g_pad_index=None,
                 ts_bank=None, ts_pad=None, ts_pad_index=None):
        self.tokenizer = tokenizer
        self.config = config
        self.data = data
        self.g_bank, self.g_pad, self.g_pad_index = g_bank, g_pad, g_pad_index
        self.ts_bank, self.ts_pad, self.ts_pad_index = ts_bank, ts_pad, ts_pad_index

        # Expand placeholder tokens in prompt
        self.g_expand_tokens = " ".join([f"<G_PLACEHOLDER {i}>" for i in range (1, 1 + self.config['query_num'])])
        self.ts_expand_tokens = " ".join([f"<TS_PLACEHOLDER {i}>" for i in range (1, 1 + self.config['query_num'])])

        # Get placeholder token ids
        self.g_token_ids, self.ts_token_ids = self.get_placeholder_id()

        # Pre tokenization for speed up
        self.row2input_ids, self.row2labels, self.row2original_tag, self.row2graph_index, self.row2ts_index = self.pre_tokenization()

    def get_placeholder_id(self):
        g_special = [f"<G_PLACEHOLDER {i}>" for i in range(1, 1 + self.config['placeholder_num'])]
        ts_special = [f"<TS_PLACEHOLDER {i}>" for i in range(1, 1 + self.config['placeholder_num'])]
        g_token_ids = self.tokenizer.convert_tokens_to_ids(g_special)
        ts_token_ids = self.tokenizer.convert_tokens_to_ids(ts_special)
        return torch.tensor(g_token_ids), torch.tensor(ts_token_ids)

    def pre_tokenization(self):
        row2input_ids = {}
        row2labels = {}
        row2original_tag = {}
        row2graph_index = {}
        row2ts_index = {}
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc="Pre-tokenizing dataset"):
            system_prompt = row.get('system_prompt', '')
            prompt = row.get('prompt', '')
            response = row.get('output', '')

            # Replace placeholder tokens in prompt with actual token ids
            prompt = prompt.replace("<G_PLACEHOLDER>", self.g_expand_tokens)
            prompt = prompt.replace("<TS_PLACEHOLDER>", self.ts_expand_tokens)

            # Get token ids and labels
            instruction_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            instruction_ids = self.tokenizer.apply_chat_template(instruction_messages,
                                                                 tokenize=True,
                                                                 add_generation_prompt=True)
            instruction_length = len(instruction_ids)

            if self.config['task_mode'] == 'Match':
                task_type = row['original_tag'].split('_')[1]
                if task_type != 'TGX':
                    response = json.dumps({'is_match': response}, ensure_ascii=False, indent=4)
                    full_messages = instruction_messages + [
                        {"role": "assistant", "content": response}]
                else:
                    response = json.dumps({'match_type': response}, ensure_ascii=False, indent=4)
                    full_messages = instruction_messages + [
                        {"role": "assistant", "content": response}]
            else:
                response = json.dumps({'is_delinquent': response}, ensure_ascii=False, indent=4)
                full_messages = instruction_messages + [
                    {"role": "assistant", "content": response}]
            full_ids = self.tokenizer.apply_chat_template(full_messages,
                                                          tokenize=True,
                                                          add_generation_prompt=False,
                                                          return_tensors='pt',
                                                          padding='max_length',
                                                          max_length=self.config['padding_length'],
                                                          truncation=True)

            labels = full_ids.clone()
            labels[0, :instruction_length] = -100  # Mask instruction part in labels
            labels[0, labels[0] == self.tokenizer.pad_token_id] = -100  # Mask padding part in labels

            row2input_ids[idx] = full_ids
            row2labels[idx] = labels
            row2original_tag[idx] = row['original_tag']
            row2graph_index[idx] = row['graph_index']
            row2ts_index[idx] = row['ts_index']
        return row2input_ids, row2labels, row2original_tag, row2graph_index, row2ts_index

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.row2input_ids[idx],
                'labels': self.row2labels[idx],
                'original_tag': self.row2original_tag[idx],
                'graph_sample_index': self.row2graph_index[idx],
                'ts_sample_index': self.row2ts_index[idx],
                }

    def collate_fn(self, batch):
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        labels = torch.cat([item['labels'] for item in batch], dim=0)
        original_tags = [item['original_tag'] for item in batch]
        is_graph = torch.isin(input_ids, self.g_token_ids)
        is_ts = torch.isin(input_ids, self.ts_token_ids)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        graph_indices = torch.tensor([item['graph_sample_index'] for item in batch])
        graph_indices[graph_indices < 0] = self.g_pad_index
        graph_x = self.g_bank[graph_indices]
        graph_x_pad = self.g_pad[graph_indices]

        ts_indices = torch.tensor([item['ts_sample_index'] for item in batch])
        ts_indices[ts_indices < 0] = self.ts_pad_index
        ts_x = self.ts_bank[ts_indices]
        ts_x_pad = self.ts_pad[ts_indices]

        return {'input_ids': input_ids,
                'labels': labels,
                'original_tags': original_tags,
                'is_graph': is_graph,
                'is_ts': is_ts,
                'attn_mask': attn_mask,
                'graph_x': graph_x,
                'graph_x_pad': graph_x_pad,
                'ts_x': ts_x,
                'ts_x_pad': ts_x_pad,
                }


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.utils.MultiviewLLM.Instruction.utils import load_and_split_data, create_tokenizer, load_graph_and_ts_embed
    from src.config.MultiviewLLM.Instruction.config import train_match_config as config

    tokenizer = create_tokenizer(config)

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

    # for i in range(len(train_dataset)):
    #     _ = train_dataset[i]
    # print(train_dataset.max_length)

