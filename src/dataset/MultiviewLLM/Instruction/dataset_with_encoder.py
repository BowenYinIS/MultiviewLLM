from torch.utils.data import Dataset
import pandas as pd
import torch
import json
from torch_geometric.data import Batch


class InstructionDataset(Dataset):
    def __init__(self, data_dict, config, tokenizer, is_test=True):
        self.data_dict = data_dict
        self.config = config
        self.tokenizer = tokenizer
        self.is_test = is_test

        # Extract data components
        self.profile_data = data_dict['profile_data']
        self.transaction_data = data_dict['transaction_data']
        self.index_data = data_dict['index_data']
        self.graph_data = data_dict['graph_data']
        self.ts_data = data_dict['ts_data']

        # Get system prompt and profile prompts
        self.system_prompt = self.get_system_prompt()
        self.profile_prompts = self.get_profile_prompt()
        # Get transaction summary
        self.transaction_summary = self.get_transaction_summary()

        # Expand placeholder tokens in prompt
        self.g_expand_tokens = " ".join([f"<G_PLACEHOLDER {i}>" for i in range (1, 1 + self.config['graph_query_num'])])
        self.ts_expand_tokens = " ".join([f"<TS_PLACEHOLDER {i}>" for i in range (1, 1 + self.config['ts_query_num'])])
        # Get placeholder token ids
        self.g_token_ids, self.ts_token_ids = self.get_placeholder_id()

        # prepaer gt token ids
        self.gt_tokens = ["true", "false"]
        self.gt_token_ids = torch.tensor([self.tokenizer.encode(tok, add_special_tokens=False)[0] for tok in self.gt_tokens])

        # Get valid data index
        if self.is_test:
            self.valid_index = self.index_data[self.index_data['split'] == 'test']['index'].tolist()
        else:
            self.valid_index = self.index_data[self.index_data['split'] == 'train']['index'].tolist()

    def get_system_prompt(self):
        prompt = (
            "角色：你是资深金融风控建模专家，熟悉信用评分、逾期定义、卡账行为、稳定性检验。\n\n"
            "任务：基于给定用户的个人信息、信用卡历史数据、时间序列表征和图表征，预测其在最新一个账单周期是否能按时还款。只使用提供的数据，不要臆测缺失值。\n\n"
            "请输出合法JSON格式（不包含 markdown code block），包含以下字段：\n"
            "{\n"
            '    "is_delinquent": boolean\n'
            "}\n"
        )
        if ('Graph' in self.config['keep_views']) and ('TS' in self.config['keep_views']):
            return prompt
        elif 'Graph' in self.config['keep_views']:
            prompt = prompt.replace("时间序列表征和图表征", "图表征")
            return prompt
        elif 'TS' in self.config['keep_views']:
            prompt = prompt.replace("、时间序列表征和图表征", "和时间序列表征")
            return prompt
        else:
            prompt = prompt.replace("、信用卡历史数据、时间序列表征和图表征", "和信用卡历史数据")
            return prompt

    def get_profile_prompt(self):
        cols = ["act_idn_sky", "lvl_4_bch_nam", "residence", "industry", "education"]
        work_act_info = self.profile_data[cols]

        profile_prompts = {}
        for _, row in work_act_info.iterrows():
            prompt = (
                f"【用户信息】:\n"
                f"- 居住地: 南京\n"
                f"- 所属分支行为: {row['lvl_4_bch_nam']}\n"
                f"- 居住情况: {row['residence']}\n"
                f"- 行业: {row['industry']}\n"
                f"- 学历: {row['education']}\n\n"
            )
            profile_prompts[row['act_idn_sky']] = prompt
        return profile_prompts

    def get_transaction_summary(self):
        grouped = (self.transaction_data.groupby(['act_idn_sky', 'billing_date'])
                   .agg(txn_number=('txn_amt', 'count'),
                        txn_amount=('txn_amt', 'sum'),
                        bank_delinquency_label=('bank_delinquency_label', 'first'))
                   )
        result = grouped.to_dict(orient='index')
        return result

    def get_placeholder_id(self):
        g_special = [f"<G_PLACEHOLDER {i}>" for i in range(1, 1 + self.config['placeholder_num'])]
        ts_special = [f"<TS_PLACEHOLDER {i}>" for i in range(1, 1 + self.config['placeholder_num'])]
        g_token_ids = self.tokenizer.convert_tokens_to_ids(g_special)
        ts_token_ids = self.tokenizer.convert_tokens_to_ids(ts_special)
        return torch.tensor(g_token_ids), torch.tensor(ts_token_ids)

    def __len__(self):
        return len(self.valid_index)

    def generate_transaction_prompt(self, act_idn_sky, billing_dates):
        prompt = "【消费与历史违约情况】:\n"
        temp = ""
        delinquency_lis = []
        transaction_num_lis = []

        for date in billing_dates:
            key = (act_idn_sky, date)
            txn_info = self.transaction_summary[key]
            # 是否是最后一个月
            if date == billing_dates[-1]:
                temp += "（最新一个账单周期）"
            # 构造描述
            if ('Graph' in self.config['keep_views']) and ('TS' in self.config['keep_views']):
                temp += f"在{date.year}年{date.month}月的账单周期，总支出金额为{txn_info['txn_amount']:.1f}元，共发生{txn_info['txn_number']}笔交易，图表征为<G_PLACEHOLDER>，时间序列表征为<TS_PLACEHOLDER>。\n"
            elif 'Graph' in self.config['keep_views']:
                temp += f"在{date.year}年{date.month}月的账单周期，总支出金额为{txn_info['txn_amount']:.1f}元，共发生{txn_info['txn_number']}笔交易，图表征为<G_PLACEHOLDER>。\n"
            elif 'TS' in self.config['keep_views']:
                temp += f"在{date.year}年{date.month}月的账单周期，总支出金额为{txn_info['txn_amount']:.1f}元，共发生{txn_info['txn_number']}笔交易，时间序列表征为<TS_PLACEHOLDER>。\n"
            else:
                temp += f"在{date.year}年{date.month}月的账单周期，总支出金额为{txn_info['txn_amount']:.1f}元，共发生{txn_info['txn_number']}笔交易。\n"
            delinquency_lis.append(txn_info['bank_delinquency_label'])
            transaction_num_lis.append(txn_info['txn_number'])
        delinquency_sum_number = sum(delinquency_lis[:-1])  # Exclude the latest month

        prompt = prompt + f"该用户在过去12个账单周期中，共发生{delinquency_sum_number}次违约。\n\n" + temp
        prompt = prompt.strip() + "\n\n"
        prompt = prompt.replace("<G_PLACEHOLDER>", self.g_expand_tokens)
        prompt = prompt.replace("<TS_PLACEHOLDER>", self.ts_expand_tokens)

        return prompt, transaction_num_lis

    def __getitem__(self, idx):
        index = self.valid_index[idx]
        act_idn_sky = self.index_data.loc[index, 'act_idn_sky']
        billing_dates = self.index_data.loc[index, 'billing_dates']

        # Construct prompt
        ## profile prompt
        profile_prompt = self.profile_prompts.get(self.index_data.loc[index, 'act_idn_sky'], '')
        ## transaction summary prompt
        transaction_prompt, transaction_number_lis = self.generate_transaction_prompt(act_idn_sky, billing_dates)

        # Construct response
        gt = self.index_data.loc[index, 'target_delinquency']
        response = "true" if gt else "false"
        response = json.dumps({'target_delinquency': response}, ensure_ascii=False, indent=4)

        # messages
        input_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": profile_prompt + transaction_prompt},
        ]
        full_messages = input_messages + [
            {"role": "assistant", "content": response}
        ]
        # In test mode, we only use input messages for generation
        if self.is_test:
            temp_message = self.tokenizer.apply_chat_template(input_messages,
                                                              tokenize=False,
                                                              add_generation_prompt=True)
            # test_suffix = '根据提供的信息，我预测该账户是否存在逾期风险的结果是：\n\n{\n    "is_delinquent": "'
            test_suffix = ''
            full_messages = temp_message + test_suffix

        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(input_messages,
                                                       tokenize=True,
                                                       add_generation_prompt=True)
        full_ids = self.tokenizer.apply_chat_template(full_messages,
                                                      tokenize=True,
                                                      add_generation_prompt=False,
                                                      return_tensors='pt',
                                                      padding='max_length',
                                                      max_length=self.config['padding_length'],
                                                      truncation=True)
        labels = full_ids.clone()

        labels[0, :len(input_ids)] = -100  # Mask instruction part in labels
        labels[0, labels[0] == self.tokenizer.pad_token_id] = -100  # Mask padding part in labels
        if self.config['sft_mode_strict']:
            labels[0, ~torch.isin(labels[0], self.gt_token_ids)] = -100  # Mask non-gt tokens in labels

        return {'input_ids': full_ids,
                'graph_data': self.graph_data[index],
                'ts_data': self.ts_data[index],
                'labels': labels,
                'index': index,
                'transaction_number_lis': transaction_number_lis,
                'gt': gt,
                }

    def txn_amt_transform(self, txn_amt):
        # Apply log(1+x) transformation
        txn_amt = torch.log(1 + txn_amt)
        txn_amt = (txn_amt - self.config['txn_amt_mean']) / self.config['txn_amt_std']
        return txn_amt

    def collate_fn(self, batch):
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        labels = torch.cat([item['labels'] for item in batch], dim=0)
        graph_data = Batch.from_data_list([item['graph_data'] for item in batch])
        indexs = torch.tensor([item['index'] for item in batch], dtype=torch.long)
        transaction_number_lis = torch.tensor([item['transaction_number_lis'] for item in batch], dtype=torch.long)
        gts = torch.tensor([item['gt'] for item in batch], dtype=torch.long)

        # ts data
        ts_data = [item['ts_data']['time_series'] for item in batch]
        mcc_cde = [torch.tensor(ts['mcc_cde'], dtype=torch.long) for ts in ts_data]
        hod = [torch.tensor(ts['hod'], dtype=torch.long) for ts in ts_data]
        dow = [torch.tensor(ts['dow'], dtype=torch.long) for ts in ts_data]
        wom = [torch.tensor(ts['wom'], dtype=torch.long) for ts in ts_data]
        moy = [torch.tensor(ts['moy'], dtype=torch.long) for ts in ts_data]
        billing_cycle_id = [torch.tensor(ts['billing_cycle_id'], dtype=torch.long) for ts in ts_data]
        txn_amt = [torch.tensor(ts['txn_amt'], dtype=torch.float32) for ts in ts_data]
        txn_amt = [self.txn_amt_transform(amt) for amt in txn_amt]
        ts_data = {
            'mcc_cde': mcc_cde,
            'hod': hod,
            'dow': dow,
            'wom': wom,
            'moy': moy,
            'billing_cycle_id': billing_cycle_id,
            'txn_amt': txn_amt,
        }

        is_graph = torch.isin(input_ids, self.g_token_ids)
        is_ts = torch.isin(input_ids, self.ts_token_ids)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {'input_ids': input_ids,
                'labels': labels,
                'graph_data': graph_data,
                'is_graph': is_graph,
                'ts_data': ts_data,
                'is_ts': is_ts,
                'attn_mask': attn_mask,
                'indexs': indexs,
                'transaction_number_lis': transaction_number_lis,
                'gts': gts,
                }

