"""
Purpose:
    Create the dataset for the LLM-related benchmarks

Input:
    `transaction` data: from `data/sample_transaction.feather`


Output:
    Dataset that are used as input for benchmark LLM models
"""

import re

import polars as pl
from polars import col as c
from tqdm import tqdm

from src.config.paths import paths


class LlmDataset:
    def __init__(self, config):
        # get the settings
        self.config = config
        self.sample_index_name = config["sample_index_name"]

        # load profile and transaction data
        # - transaction includes spending, balance, and delinquency label
        self.profile = pl.read_ipc(paths.act_info, memory_map=False)
        self.transaction = pl.read_ipc(paths.sample_transaction, memory_map=False)

        # get the valid act_idn_sky
        self.valid_act_idn_sky = set(self.profile["act_idn_sky"]) & set(
            self.transaction["act_idn_sky"]
        )
        print("N valid act_idn_sky:", len(self.valid_act_idn_sky))

        # ensure profile and transaction have the same act_idn_sky
        self.profile = self.profile.filter(c.act_idn_sky.is_in(self.valid_act_idn_sky))
        self.transaction = self.transaction.filter(c.act_idn_sky.is_in(self.valid_act_idn_sky))

        # load sample index
        self.sample_index = pl.read_ipc(
            paths.processed_data_dir / f"sample_index/{self.sample_index_name}.feather",
            memory_map=False,
        )

    def build_cycle_units(self):
        """
        Construct a list of cycle units

        Each cycle unit contains all information for a single billing cycle.
        A cycle unit is represented as a dictionary, keyed by
        (act_idn_sky, billing_date), with the value being a dictionary
        containing the cycle's information.

        In subsequent steps (e.g., the `assemble_samples` method), multiple
        cycle units will be combined to form a single sample.
        """
        # initialize the list of cycle units
        cycle_units = {}

        # convert data to dict for fast access
        # key: (act_idn_sky, billing_date), value: pl.DataFrame
        transaction_dict = (
            self.transaction
            # key: (act_idn_sky, billing_date), value: pl.DataFrame
            .partition_by(
                "act_idn_sky", "billing_date", maintain_order=True, as_dict=True, include_key=False
            )
        )

        # for each (act_idn_sky, billing_date) pair, we construct an unit of observation
        # - each sample will have a list of such units, depending on how many billing cycles we use
        # - for example, if we use 6 billing cycles, then each sample is a list of 6 units
        for (act_idn_sky, billing_date), transaction_cycle in tqdm(
            transaction_dict.items(), desc="Constructing cycle units"
        ):
            # delinquency label of the cycle
            is_delinquent = transaction_cycle["bank_delinquency_label"].last()

            # balance of the cycle
            balance_cycle = transaction_cycle["balance"].round().cast(pl.Int32).last()

            # transaction text of the cycle
            transaction_text = "\n".join(
                transaction_cycle.select(
                    prompt=pl.format(
                        "时间：{} {}，商户类别：{}，交易描述：{}，金额{}元。",
                        c.txn_dte,
                        c.txn_tme,
                        c.mcc_desc,
                        c.txn_des,
                        c.txn_amt,
                    )
                )["prompt"].to_list()
            )

            # add meta data to transaction text
            transaction_text = (
                f"在{billing_date.year}年{billing_date.month}月的账单周期，"
                f"应付账单金额为{balance_cycle}元，"
                f"该账单周期是否违约：{'是' if is_delinquent else '否'}，"
                f"具体消费如下：\n{transaction_text}"
            )

            # add the current cycle to the unit list
            cycle_units[(act_idn_sky, billing_date)] = {
                "is_delinquent": is_delinquent,
                "balance_cycle": balance_cycle,
                "transaction_text": transaction_text,
            }

        # save as class attribute
        self.cycle_units = cycle_units

    def assemble_samples(self):
        """
        Assemble training and testing samples for LLM benchmarking by combining profile and transaction data
        into rolling windows of billing cycles.

        Input:
            self.cycle_units (dict): Mapping from (act_idn_sky, billing_date) to a dict containing:
                - is_delinquent (bool): Whether the cycle is delinquent.
                - balance_cycle (int): The balance for the cycle.
                - transaction_text (str): The formatted transaction text for the cycle.

        Returns:
            pl.DataFrame: A dataframe where each row is a sample containing:
                - split (str): "train" or "test"
                - profile fields for the account
                - billing_dates (list): List of billing dates in the window
                - transaction_text (str): Concatenated transaction text for all cycles in the window
                - target_delinquency (bool): Delinquency label for the last cycle in the window
        """

        # initialize the results
        samples = []

        for row in tqdm(
            self.sample_index.iter_rows(named=True),
            total=self.sample_index.height,
            desc="Assembling samples",
        ):
            # get the data for the current sample
            act_idn_sky = row["act_idn_sky"]
            billing_dates = row["billing_dates"]

            # get the profile data for this account
            profile_act = self.profile.filter(c.act_idn_sky == act_idn_sky).to_dicts()[0]

            # add profile to the sample
            sample = row | {
                **profile_act,
                "transaction_text": "",
            }

            # add transaction text to the sample
            for idx, billing_date in enumerate(billing_dates):
                # get transaction text of the current cycle
                transaction_text = self.cycle_units[(act_idn_sky, billing_date)]["transaction_text"]

                # if this is the last cycle
                # - remove delinquency lable from the transaction text
                if idx == len(billing_dates) - 1:
                    transaction_text = re.sub(
                        r"该账单周期是否违约.*?，(?=具体消费如下)", "", transaction_text
                    )

                # add transaction text
                sample["transaction_text"] += transaction_text + "\n\n"

            # add the sample to the results
            samples.append(sample)

        # convert the results to a dataframe
        self.samples = pl.DataFrame(samples)

    def build(self):
        """
        Build the complete dataset by constructing cycle units and assembling samples.

        Returns:
            pl.DataFrame: The assembled dataset with train/test splits ready for LLM benchmarking.
        """
        # build cycle units
        self.build_cycle_units()

        # assemble samples
        self.assemble_samples()

        print(f"Dataset built: {len(self.samples)} samples")
        print(
            f"\tTrain: {(self.samples['split'] == 'train').sum()}, Test: {(self.samples['split'] == 'test').sum()}"
        )

        # save the dataset
        save_path = (
            paths.processed_data_dir
            / f"llm_benchmark_samples/{self.sample_index_name.replace('index', 'samples')}.feather"
        )
        self.samples.write_ipc(save_path, compression="lz4")
        print(f"Dataset saved to {save_path.name}")


if __name__ == "__main__":
    configs = [
        {"sample_index_name": "index_min6mo_allprevious_2test"},
        {"sample_index_name": "index_min6mo_fixed_2test"},
        {"sample_index_name": "index_min12mo_allprevious_2test"},
        {"sample_index_name": "index_min12mo_fixed_2test"},
        {"sample_index_name": "index_min12mo_allprevious_1test"},
        {"sample_index_name": "index_min12mo_fixed_1test"},
    ]

    # build and save the dataset
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 80}")
        print(f"Building dataset {i}/{len(configs)}")
        print(f"Config: {config}")
        print(f"{'=' * 80}\n")

        dataset = LlmDataset(config)
        dataset.build()
