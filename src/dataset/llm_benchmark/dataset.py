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
        self.min_train_size = config["min_train_size"]
        self.train_previous_all = config["train_previous_all"]
        self.test_size = config["test_size"]
        self.data_dir = config["data_dir"]

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

        return cycle_units

    def _build_windows(self, billing_dates):
        """
        Input:
            billing_dates: list of billing dates for an account, sorted in order.
            min_train_size: int. Minimum number of billing cycles in a training window.
            train_previous_all: bool. Whether to use all previous billing cycles for each window.
            test_size: int. Maximum number of test samples per account.
        Output:
            windows: list of (window, label), where label is either "train" or "test".
        """
        # build the windows
        if not self.train_previous_all:
            # only a fixed number of billing cycles are used
            windows = [
                billing_dates[i : i + self.min_train_size]
                for i in range(0, len(billing_dates) - self.min_train_size + 1)
            ]
        else:
            # all previous billing cycles are used
            windows = [
                billing_dates[:i] for i in range(self.min_train_size, len(billing_dates) + 1)
            ]

        # label each window as train or test
        # - at least one test sample is created for each account
        # - the max number of test samples is test_size
        # - the number of test samples is determined by the number of billing cycles
        actual_test_size = min(self.test_size, len(windows) - 1)
        train_end = len(windows) - actual_test_size

        for i, window in enumerate(windows):
            if i < train_end:
                windows[i] = (window, "train")
            else:
                windows[i] = (window, "test")

        return windows

    def assemble_samples(self, cycle_units):
        """
        Assemble training and testing samples for LLM benchmarking by combining profile and transaction data
        into rolling windows of billing cycles.

        Args:
            cycle_units (dict): Mapping from (act_idn_sky, billing_date) to a dict containing:
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

        # get dict: {act_idn_sky: [billing_date, ...]}
        transaction_by_act = (
            self.transaction
            # only keep unique (act_idn_sky, billing_date) pairs
            .select(c.act_idn_sky, c.billing_date)
            .unique()
            # partition into {act_idn_sky: [billing_date, ...]}
            .partition_by("act_idn_sky", maintain_order=True, as_dict=True, include_key=False)
        )

        # initialize the results
        samples = []

        for (act_idn_sky,), billing_dates in transaction_by_act.items():
            # convert billing_dates to list
            billing_dates = billing_dates["billing_date"].sort().unique().to_list()

            # ensure at least have train_num billing_dates
            # - if the number of billing_dates is exactly train_num, then this
            #   account is used as a train sample
            if len(billing_dates) < self.min_train_size:
                continue

            # get the profile data for this account
            profile_act = self.profile.filter(c.act_idn_sky == act_idn_sky).to_dicts()[0]

            # assemble the biling dates into rolling windows
            # return: [[201701,201702],[201702,201703],...]
            windows = self._build_windows(billing_dates)

            # using the window_index to construct the samples
            for window, split in windows:
                # a window is like [2017-01 ,2017-02,...,2027-06]

                # initialize the sample with train/testsplit, profile data, billing dates, and empty transaction text
                # - we'll add transaction text later
                sample = (
                    {"split": split}
                    | profile_act
                    | {"billing_dates": window, "transaction_text": ""}
                )

                # assemble all cycles in the window into one sample
                for idx, billing_date in enumerate(window):
                    # get delinquency label and transaction text of the current cycle
                    is_delinquent, balance, transaction_text = cycle_units[
                        (act_idn_sky, billing_date)
                    ].values()

                    # if this is the last cycle
                    # - remove delinquency lable from the transaction text
                    # - add the target delinquency label
                    if idx == len(window) - 1:
                        # add transaction text
                        transaction_text = re.sub(
                            r"该账单周期是否违约.*?，(?=具体消费如下)", "", transaction_text
                        )
                        sample["target_delinquency"] = is_delinquent

                    # add transaction text
                    sample["transaction_text"] += transaction_text + "\n\n"

                # add the sample to the results
                samples.append(sample)

        # convert the results to a dataframe
        return pl.DataFrame(samples)

    def build(self):
        """
        Build the complete dataset by constructing cycle units and assembling samples.

        Returns:
            pl.DataFrame: The assembled dataset with train/test splits ready for LLM benchmarking.
        """
        print("Building cycle units...")
        cycle_units = self.build_cycle_units()

        print("Assembling samples...")
        samples = self.assemble_samples(cycle_units)

        print(f"Dataset built: {len(samples)} samples")
        print(
            f"\tTrain: {(samples['split'] == 'train').sum()}, Test: {(samples['split'] == 'test').sum()}"
        )

        # save the dataset
        save_path = (
            self.data_dir
            + f"/samples_min{self.config['min_train_size']}mo_{'allprevious' if self.config['train_previous_all'] else 'fixed'}_{self.config['test_size']}test.feather"
        )
        samples.write_ipc(save_path, compression="lz4")
        print(f"Dataset saved to {save_path}")

        return samples


if __name__ == "__main__":
    config = {
        "min_train_size": 6,
        "train_previous_all": False,
        "test_size": 2,
        "data_dir": str(paths.processed_data_dir/'sample_index'),
    }

    # build and save the dataset
    dataset = LlmDataset(config)
    samples = dataset.build()
