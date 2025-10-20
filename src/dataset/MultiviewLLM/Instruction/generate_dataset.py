'''
生成指令数据集
'''


import random
import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.config.paths import paths
from torch_geometric.data import Data
from src.utils.seed_everything import seed_everything


class GenerateInstructDataset:
    def __init__(self, config: dict):
        """
        Initialize the dataset builder with configuration and load source data.

        Args:
            config (dict): Configuration dictionary containing:
                - sample_index_path (str): Path to the sample index file
                - output_data_dir (str): Directory path for saving the dataset
        """
        # store configuration
        self.config = config
        self.task_mode = config['task_mode']
        self.negative_mode_ratio = config['negative_mode_ratio']
        self.negative_ratio = config['negative_ratio']
        self.text_graph_ts_ratio = config.get('text_graph_ts_ratio', None)
        self.graph_ts_ratio = config.get('graph_ts_ratio', None)
        self.sample_index_path = config["sample_index_path"]
        self.output_data_dir = config["output_data_dir"]

        # load source data
        self.transaction = pd.read_feather(paths.sample_transaction)
        self.sample_index = pd.read_feather(self.sample_index_path)
        self.act_info = pd.read_feather(paths.act_info)

        # set random seed
        seed_everything(config['seed'])

        # create template for different task modes
        self._create_prompt_template()
        self._create_system_prompt()

        # preprocess transaction data
        self._preprocess_data()

        # construct profile prompt
        self.construct_profile_prompt()

        # construct recent transaction prompt
        self.construct_recent_transaction_prompt()

        # create act_idn_sky for negative sampling
        self.act_idx_map = self.sample_index.groupby('act_idn_sky')['index'].apply(list).to_dict()
        self.all_idx_list = self.sample_index['index'].tolist()

    def _create_prompt_template(self):
        '''
        Create prompt templates for different task modes.
        '''
        self.text_graph_prompt_template = (
            "【图表征描述】:\n"
            "<G_PLACEHOLDER>\n\n"
            "{profile_prompt}\n\n"
            "{recent_transaction_prompt}\n"
        )
        self.text_ts_prompt_template = (
            "【时间序列表征描述】:\n"
            "<TS_PLACEHOLDER>\n\n"
            "{profile_prompt}\n\n"
            "{recent_transaction_prompt}\n"
        )
        self.graph_ts_prompt_template = (
            "【图表征描述】:\n"
            "<G_PLACEHOLDER>\n\n"
            "【时间序列表征描述】:\n"
            "<TS_PLACEHOLDER>\n"
        )
        self.text_graph_ts_prompt_template = (
            "【图表征描述】:\n"
            "<G_PLACEHOLDER>\n\n"
            "【时间序列表征描述】:\n"
            "<TS_PLACEHOLDER>\n\n"
            "{profile_prompt}\n\n"
            "{recent_transaction_prompt}\n"
        )
        self.delinquency_prediction_prompt_template = (
            "【图表征描述】:\n"
            "<G_PLACEHOLDER>\n\n"
            "【时间序列表征描述】:\n"
            "<TS_PLACEHOLDER>\n\n"
            "{profile_prompt}\n\n"
            "{recent_transaction_prompt}\n"
        )

    def _create_system_prompt(self):
        self.text_graph_system_prompt = (
            "角色：你是一名资深金融风控与多模态表征匹配专家，熟悉信用卡消费行为、文本表征与图表征的语义一致性评估。\n\n"
            "任务：给定某用户某一时间段的信用卡消费“文本概览”与这些消费的“图表征”，"
            "判断两者是否匹配（是否描述同一用户/同一消费集合）。仅依据提供的数据进行判断，不得臆测缺失值或引入外部信息。\n\n"
            "输出：仅输出合法 JSON 对象（不包含 markdown 代码块或任何解释性文本），格式如下：\n"
            "{\n"
            '    "is_match": "true" 或 "false"\n'
            "}\n"
        )

        self.text_ts_system_prompt = (
            "角色：你是一名资深金融风控与多模态表征匹配专家，熟悉信用卡消费行为、文本表征与时间序列表征的语义一致性评估。\n\n"
            "任务：给定某用户某一时间段的信用卡消费“文本概览”与这些消费的“时序表征”，"
            "判断两者是否匹配（是否描述同一用户/同一消费集合）。仅依据提供的数据进行判断，不得臆测缺失值或引入外部信息。\n\n"
            "输出：仅输出合法 JSON 对象（不包含 markdown 代码块或任何解释性文本），格式如下：\n"
            "{\n"
            '    "is_match": "true" 或 "false"\n'
            "}\n"
        )

        self.graph_ts_system_prompt = (
            "角色：你是一名资深金融风控与多模态表征匹配专家，熟悉信用卡消费行为、图表征与时间序列表征的一致性评估。\n\n"
            "任务：给定某用户某一时间段消费的“图表征”与“时序表征”，"
            "判断两者是否匹配（是否描述同一用户/同一消费集合）。仅依据提供的数据进行判断，不得臆测缺失值或引入外部信息。\n\n"
            "输出：仅输出合法 JSON 对象（不包含 markdown 代码块或任何解释性文本），格式如下：\n"
            "{\n"
            '    "is_match": "true" 或 "false"\n'
            "}\n"
        )

        self.text_graph_ts_system_prompt = (
            "角色：你是一名资深金融风控与多模态表征匹配专家，熟悉信用卡消费行为在文本、图结构与时间序列三种表征间的一致性建模。\n\n"
            "任务：给定某用户某一时间段的信用卡消费“文本概览”、“图表征”与"
            "“时序表征”，判断三者之间的匹配关系。仅依据提供的数据进行判断，不得臆测缺失值或引入外部信息。\n\n"
            "分类定义：\n"
            " - “全部匹配”：三种表征均描述同一用户/同一消费集合；\n"
            " - “图不匹配”：文本与时序匹配，但图不一致；\n"
            " - “时序不匹配”：文本与图匹配，但时序不一致；\n"
            " - “全部不匹配”：三者之间均不一致。\n\n"
            "输出：仅输出合法 JSON 对象（不包含 markdown 代码块或任何解释性文本），格式如下：\n"
            "{\n"
            '    "match_type": "全部匹配" 或 "图不匹配" 或 "时序不匹配" 或 "全部不匹配"\n'
            "}\n"
        )

        self.delinquency_prediction_system_prompt = (
            "角色：你是资深金融风控建模专家，熟悉信用评分、逾期定义、卡账行为、稳定性检验。\n\n"
            "任务：给定同一用户的“用户信息”、某一时间段的信用卡消费“文本概览”、“图表征”与“时序表征”，预测其在最新一个账单周期是否能按时还款。仅依据提供的数据进行判断，不得臆测缺失值或引入外部信息。\n\n"
            "输出要求：仅输出合法 JSON 对象（不包含 markdown 代码块或任何解释性文本），格式如下：\n"
            "{\n"
            '    "is_delinquent": "true" 或 "false"\n'
            "}\n"
        )

    def _preprocess_data(self):
        '''
        Preprocess the transaction data by handling missing values, generating datetime columns,
        sorting.

        Preprocess the sample index data by adding an index column.
        '''
        # fill missing mcc_cde with 9999 and mcc_13cat with 'Insurance'
        self.transaction['mcc_cde'] = self.transaction['mcc_cde'].fillna(9999)
        self.transaction['mcc_13cat'] = self.transaction['mcc_13cat'].fillna('Insurance')
        self.transaction.loc[self.transaction['mcc_cde'] == 9999, 'mcc_desc'] = '保险扣款'
        self.transaction['mcc_cde'] = self.transaction['mcc_cde'].astype(str)

        # generate txn_dte_tme and sort
        self.transaction['txn_dte_tme'] = self.transaction['txn_dte'].astype(str) + ' ' + self.transaction[
            'txn_tme'].astype(str)
        self.transaction['txn_dte_tme'] = pd.to_datetime(self.transaction['txn_dte_tme'], format='%Y-%m-%d %H:%M:%S')
        self.transaction = self.transaction.sort_values(by=['act_idn_sky', 'txn_dte_tme'])

        # generate mcc_transition
        self.transaction['next_mcc_desc'] = self.transaction.groupby("act_idn_sky")['mcc_desc'].shift(-1)
        self.transaction = self.transaction.dropna(subset=['next_mcc_desc'])
        self.transaction['mcc_transition'] = self.transaction['mcc_desc'] + '->' + self.transaction['next_mcc_desc']

        # generate hour and map to bucket
        self.transaction['hour'] = self.transaction['txn_dte_tme'].dt.hour
        bins = [-1, 6, 12, 18, 24]
        labels = ['夜间', '上午', '下午', '晚上']
        self.transaction['tod'] = pd.cut(self.transaction['hour'], bins=bins, labels=labels)

        # add index column to sample_index
        self.sample_index['index'] = self.sample_index.index

    def construct_profile_prompt(self):
        cols = ["act_idn_sky", "lvl_4_bch_nam", "residence", "industry", "education"]
        work_act_info = self.act_info[cols]

        profile_prompts = {}
        for _, row in work_act_info.iterrows():
            prompt = (
                f"【用户信息】:\n"
                f"- 居住地: 南京\n"
                f"- 所属分支行为: {row['lvl_4_bch_nam']}\n"
                f"- 居住情况: {row['residence']}\n"
                f"- 行业: {row['industry']}\n"
                f"- 学历: {row['education']}"
            )
            profile_prompts[row['act_idn_sky']] = prompt
        self.profile_prompts = profile_prompts

    def construct_recent_transaction_prompt(self):
        save_path = self.output_data_dir / self.sample_index_path.name.replace('.feather', f'_recent_transaction_prompts.json')
        if save_path.exists():
            with open(save_path, 'r', encoding='utf-8') as f:
                self.recent_transaction_prompts = json.load(f)
            return
        recent_transaction_prompts = {}
        for _, row in tqdm(self.sample_index.iterrows(), total=self.sample_index.shape[0], desc="Constructing recent transaction prompts"):
            act_idn_sky = row['act_idn_sky']
            billing_dates = row['billing_dates']
            idx = row['index']
            working_data = self.transaction[
                (self.transaction['act_idn_sky'] == act_idn_sky) &
                (self.transaction['billing_date'].isin(billing_dates))
            ].copy()

            start_str = working_data['txn_dte_tme'].min().strftime('%Y-%m-%d')
            end_str = working_data['txn_dte_tme'].max().strftime('%Y-%m-%d')

            n_txn = working_data.shape[0]
            active_days = working_data['txn_dte'].nunique()
            txn_per_day = n_txn / active_days if active_days > 0 else 0

            total_amt = working_data['txn_amt'].sum()
            mean_amt = working_data['txn_amt'].mean() if n_txn > 0 else 0
            med_amt = working_data['txn_amt'].median() if n_txn > 0 else 0
            p90_amt = working_data['txn_amt'].quantile(0.9) if n_txn > 0 else 0

            vc = working_data['mcc_desc'].value_counts(normalize=True).head(self.config['mcc_top_k'])
            cat_str = ', '.join([f"{cat} {pct:.1%}" for cat, pct in zip(vc.index, vc.values)])

            tod_vc = working_data['tod'].value_counts(normalize=True)
            tod_str = ', '.join([f"{tod} {pct:.1%}" for tod, pct in zip(tod_vc.index, tod_vc.values)])

            trans_vc = working_data['mcc_transition'].value_counts(normalize=True).head(self.config['transition_top_k'])
            trans_str = ', '.join([f"{trans} {pct:.1%}" for trans, pct in zip(trans_vc.index, trans_vc.values)])

            bill_delinquency = working_data[['billing_date', 'bank_delinquency_label']].drop_duplicates().sort_values(by='billing_date')
            bill_delinquency = bill_delinquency.iloc[:-1]  # exclude the most recent billing cycle
            delinquency_num = bill_delinquency['bank_delinquency_label'].sum()

            prompt = (
                f"【近期交易概览】（{start_str} 至 {end_str}）:\n"
                f"- 交易笔数：{n_txn}（活跃天数 {active_days} 天，日均 {txn_per_day:.2f} 笔）\n"
                f"- 金额：总额 ¥{total_amt:,.0f}；客单价 ¥{mean_amt:,.0f}（中位 ¥{med_amt:,.0f}；P90 ¥{p90_amt:,.0f}）\n"
                f"- 类目占比（Top{vc.shape[0]}）：{cat_str}\n"
                f"- 时段分布：{tod_str}\n"
                f"- 常见转移（Top{trans_vc.shape[0]}）：{trans_str}\n"
                f"- 除去最近一个账单周期，近{bill_delinquency.shape[0]}个账单周期内有{delinquency_num}次逾期"
            )

            recent_transaction_prompts[str(idx)] = prompt
        self.recent_transaction_prompts = recent_transaction_prompts

        # save to json
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(recent_transaction_prompts, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _assign_negative_match_index(row, sample_mode_col, act_idx_map, all_idx_list):
        ''' Assign a match_index for a negative sample based on its label. '''
        mode = row[sample_mode_col]
        index = row['index']
        act_idn_sky = row['act_idn_sky']
        if mode == 'random':
            match_index = index
            while match_index == index:
                match_index = random.choice(all_idx_list)
            return match_index
        elif mode == 'other_act':
            other_acts = [act for act in act_idx_map.keys() if act != act_idn_sky]
            chosen_act = random.choice(other_acts)
            match_index = random.choice(act_idx_map[chosen_act])
            return match_index
        elif mode == 'same_act':
            same_act_indices = act_idx_map[act_idn_sky]
            if len(same_act_indices) == 1:
                same_act_indices = all_idx_list
            match_index = index
            while match_index == index:
                match_index = random.choice(same_act_indices)
            return match_index

    def negative_sample(self, df, sample_mode_col='sample_mode', sample_index_col='sample_index'):
        '''
        Generate negative samples for the given DataFrame of positive samples.
        '''
        df = df.copy()

        negative_num = df.shape[0]
        other_act_num = int(negative_num * self.negative_mode_ratio['other_act'])
        same_act_num = int(negative_num * self.negative_mode_ratio['same_act'])
        random_num = negative_num - other_act_num - same_act_num
        negative_labels = ['other_act'] * other_act_num + ['same_act'] * same_act_num + ['random'] * random_num
        random.shuffle(negative_labels)

        df[sample_mode_col] = negative_labels
        df[sample_index_col] = df.apply(self._assign_negative_match_index, axis=1, args=(sample_mode_col, self.act_idx_map, self.all_idx_list))
        return df

    def construct_prompt(self, row, task_mode=None):
        ''' Construct the prompt for a given row based on the task mode. '''
        if task_mode == 'Text-Graph':
            profile_prompt = self.profile_prompts[row['act_idn_sky']]
            recent_transaction_prompt = self.recent_transaction_prompts[str(row['index'])]
            return self.text_graph_prompt_template.format(
                profile_prompt=profile_prompt,
                recent_transaction_prompt=recent_transaction_prompt
            )
        elif task_mode == 'Text-TimeSeries':
            profile_prompt = self.profile_prompts[row['act_idn_sky']]
            recent_transaction_prompt = self.recent_transaction_prompts[str(row['index'])]
            return self.text_ts_prompt_template.format(
                profile_prompt=profile_prompt,
                recent_transaction_prompt=recent_transaction_prompt
            )
        elif task_mode == 'Graph-TimeSeries':
            return self.graph_ts_prompt_template
        elif task_mode == 'Text-Graph-TimeSeries':
            profile_prompt = self.profile_prompts[row['act_idn_sky']]
            recent_transaction_prompt = self.recent_transaction_prompts[str(row['index'])]
            return self.text_graph_ts_prompt_template.format(
                profile_prompt=profile_prompt,
                recent_transaction_prompt=recent_transaction_prompt
            )
        elif task_mode == 'Delinquency-Prediction':
            profile_prompt = self.profile_prompts[row['act_idn_sky']]
            recent_transaction_prompt = self.recent_transaction_prompts[str(row['index'])]
            return self.delinquency_prediction_prompt_template.format(
                profile_prompt=profile_prompt,
                recent_transaction_prompt=recent_transaction_prompt
            )
        else:
            raise ValueError(f"Unsupported task mode: {task_mode}")

    def build(self):
        '''
        Build the entire dataset.
        '''
        if self.task_mode == 'Text-Graph':
            # construct positive samples
            positive_samples = self.sample_index.copy()
            positive_samples['graph_index'] = positive_samples['index']
            positive_samples['graph_sample_mode'] = 'anchor'
            positive_samples['output'] = "true"

            # construct negative samples
            negative_samples = pd.concat([self.sample_index.copy() for _ in range(self.negative_ratio)], ignore_index=True)
            negative_samples = self.negative_sample(negative_samples, sample_mode_col='graph_sample_mode', sample_index_col='graph_index')
            negative_samples['output'] = "false"

            # combine positive and negative samples
            all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
            all_samples = all_samples.sample(frac=1).reset_index(drop=True)

            # construct prompt
            all_samples['prompt'] = all_samples.apply(self.construct_prompt, axis=1, args=(self.task_mode,))
            all_samples['system_prompt'] = self.text_graph_system_prompt
        elif self.task_mode == 'Text-TimeSeries':
            # construct positive samples
            positive_samples = self.sample_index.copy()
            positive_samples['ts_index'] = positive_samples['index']
            positive_samples['ts_sample_mode'] = 'anchor'
            positive_samples['output'] = "true"

            # construct negative samples
            negative_samples = pd.concat([self.sample_index.copy() for _ in range(self.negative_ratio)], ignore_index=True)
            negative_samples = self.negative_sample(negative_samples, sample_mode_col='ts_sample_mode', sample_index_col='ts_index')
            negative_samples['output'] = "false"

            # combine positive and negative samples
            all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
            all_samples = all_samples.sample(frac=1).reset_index(drop=True)

            # construct prompt
            all_samples['prompt'] = all_samples.apply(self.construct_prompt, axis=1, args=(self.task_mode,))
            all_samples['system_prompt'] = self.text_ts_system_prompt
        elif self.task_mode == 'Graph-TimeSeries':
            # construct positive samples
            positive_samples = self.sample_index.copy()
            positive_samples['graph_index'] = positive_samples['index']
            positive_samples['ts_index'] = positive_samples['index']
            positive_samples['graph_sample_mode'] = 'anchor'
            positive_samples['ts_sample_mode'] = 'anchor'
            positive_samples['output'] = "true"
            positive_samples['describe'] = '匹配'

            # construct negative samples
            negative_samples = pd.concat([self.sample_index.copy() for _ in range(self.negative_ratio)], ignore_index=True)

            g_unmatched_num= int(negative_samples.shape[0] * self.graph_ts_ratio['graph_unmatched'])
            ts_unmatched_num = negative_samples.shape[0] - g_unmatched_num
            describe_labels = (['图表征不匹配'] * g_unmatched_num +
                             ['时间序列表征不匹配'] * ts_unmatched_num)
            random.shuffle(describe_labels)
            negative_samples['describe'] = describe_labels

            temp = []
            for mode, group in negative_samples.groupby('describe'):
                if mode == '图表征不匹配':
                    group = self.negative_sample(group, sample_mode_col='graph_sample_mode', sample_index_col='graph_index')
                    group['ts_index'] = group['index']
                    group['ts_sample_mode'] = 'anchor'
                    group['output'] = "false"
                elif mode == '时间序列表征不匹配':
                    group = self.negative_sample(group, sample_mode_col='ts_sample_mode', sample_index_col='ts_index')
                    group['graph_index'] = group['index']
                    group['graph_sample_mode'] = 'anchor'
                    group['output'] = "false"
                temp.append(group)
            negative_samples = pd.concat(temp, ignore_index=True)

            # combine positive and negative samples
            all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
            all_samples = all_samples.sample(frac=1).reset_index(drop=True)

            # construct prompt
            all_samples['prompt'] = all_samples.apply(self.construct_prompt, axis=1, args=(self.task_mode,))
            all_samples['system_prompt'] = self.graph_ts_system_prompt
        elif self.task_mode == 'Text-Graph-TimeSeries':
            # construct positive samples
            positive_samples = self.sample_index.copy()
            positive_samples['graph_index'] = positive_samples['index']
            positive_samples['ts_index'] = positive_samples['index']
            positive_samples['graph_sample_mode'] = 'anchor'
            positive_samples['ts_sample_mode'] = 'anchor'
            positive_samples['output'] = '全部匹配'
            positive_samples['describe'] = '匹配'

            # construct negative samples
            negative_samples = pd.concat([self.sample_index.copy() for _ in range(self.negative_ratio)], ignore_index=True)
            tg_unmatched_num = int(negative_samples.shape[0] * self.text_graph_ts_ratio['both_unmatched'])
            g_unmatched_num = int(negative_samples.shape[0] * self.text_graph_ts_ratio['graph_unmatched'])
            ts_unmatched_num = negative_samples.shape[0] - tg_unmatched_num - g_unmatched_num
            describe_labels = (['文本描述与两个表征均不匹配'] * tg_unmatched_num +
                             ['图表征不匹配'] * g_unmatched_num +
                             ['时间序列表征不匹配'] * ts_unmatched_num)
            random.shuffle(describe_labels)
            negative_samples['describe'] = describe_labels

            temp = []
            for mode, group in negative_samples.groupby('describe'):
                if mode == '文本描述与两个表征均不匹配':
                    group = self.negative_sample(group, sample_mode_col='graph_sample_mode', sample_index_col='graph_index')
                    group = self.negative_sample(group, sample_mode_col='ts_sample_mode', sample_index_col='ts_index')
                    group['output'] = '全部不匹配'
                elif mode == '图表征不匹配':
                    group = self.negative_sample(group, sample_mode_col='graph_sample_mode', sample_index_col='graph_index')
                    group['ts_index'] = group['index']
                    group['ts_sample_mode'] = 'anchor'
                    group['output'] = '图不匹配'
                elif mode == '时间序列表征不匹配':
                    group = self.negative_sample(group, sample_mode_col='ts_sample_mode', sample_index_col='ts_index')
                    group['graph_index'] = group['index']
                    group['graph_sample_mode'] = 'anchor'
                    group['output'] = '时序不匹配'
                temp.append(group)
            negative_samples = pd.concat(temp, ignore_index=True)

            # combine positive and negative samples
            all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
            all_samples = all_samples.sample(frac=1).reset_index(drop=True)

            # construct prompt
            all_samples['prompt'] = all_samples.apply(self.construct_prompt, axis=1, args=(self.task_mode,))
            all_samples['system_prompt'] = self.text_graph_ts_system_prompt
        elif self.task_mode == 'Delinquency-Prediction':
            all_samples = self.sample_index.copy()
            all_samples['prompt'] = all_samples.apply(self.construct_prompt, axis=1, args=(self.task_mode,))
            all_samples['output'] = all_samples['target_delinquency'].map({True: 'true', False: 'false'})
            all_samples['graph_index'] = all_samples['index']
            all_samples['ts_index'] = all_samples['index']
            all_samples['graph_sample_mode'] = 'anchor'
            all_samples['ts_sample_mode'] = 'anchor'
            all_samples['system_prompt'] = self.delinquency_prediction_system_prompt
        else:
            raise ValueError(f"Unsupported task mode: {self.task_mode}")

        # save dataset
        save_path = self.output_data_dir / self.sample_index_path.name.replace('.feather', f'_{self.task_mode}_dataset.feather')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        all_samples.to_feather(save_path)
        return all_samples


if __name__ == '__main__':
    # Define configuration
    from src.config.MultiviewLLM.Instruction.config import generate_dataset_config as config

    for task in ["Text-Graph", "Text-TimeSeries", "Graph-TimeSeries", "Text-Graph-TimeSeries", "Delinquency-Prediction"]:
        print(f"Generating dataset for task: {task}")
        config['task_mode'] = task
         # Generate dataset
        generator = GenerateInstructDataset(config)
        dataset = generator.build()

    # for file in Path('/data/bwyin/project/MultiviewLLM/processed_data/sample_index').glob('samples_*.feather'):
    #     # Update sample index path in config
    #     config['sample_index_path'] = file
    #     print(f"Processing sample index: {file.name}")
    #     # Generate dataset
    #     generator = GenerateInstructDataset(config)
    #     generator.build()