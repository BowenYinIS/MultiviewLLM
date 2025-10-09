"""
Purpose:
    Create the sample index

Input:
    `transaction` data: from `data/raw_data/sample_transaction.feather`
    `profile` data: from `data/raw_data/act_info.feather`

Output:
    Dataset where each row is a sample, containing the following columns:
    - split: "train" or "test"
    - act_idn_sky: the account id
    - billing_dates: list of billing dates in the window
    - target_delinquency: delinquency label for the last cycle in the window
"""

import polars as pl
from polars import col as c

from src.config.paths import paths


class SampleIndexDataset:
    """
    Dataset builder for creating training and testing samples from transaction data.

    This class creates rolling window samples from account transaction histories,
    where each sample contains a sequence of billing cycles and a delinquency target.
    """

    def __init__(self, config):
        """
        Initialize the dataset builder with configuration and load source data.

        Args:
            config (dict): Configuration dictionary containing:
                - min_train_size (int): Minimum number of billing cycles per window
                - train_previous_all (bool): Use expanding vs. fixed-size windows
                - test_size (int): Maximum number of test samples per account
                - data_dir (str): Directory path for saving the dataset
        """
        # store configuration
        self.config = config
        self.min_train_size = config["min_train_size"]
        self.train_previous_all = config["train_previous_all"]
        self.test_size = config["test_size"]
        self.data_dir = config["data_dir"]

        # load source data
        self.profile = pl.read_ipc(paths.act_info, memory_map=False)
        self.transaction = pl.read_ipc(paths.sample_transaction, memory_map=False)

        # find accounts that exist in both datasets
        self.valid_act_idn_sky = set(self.profile["act_idn_sky"]) & set(self.transaction["act_idn_sky"])
        print("N valid act_idn_sky:", len(self.valid_act_idn_sky))

        # filter both datasets to only include valid accounts
        self.profile = self.profile.filter(c.act_idn_sky.is_in(self.valid_act_idn_sky))
        self.transaction = self.transaction.filter(c.act_idn_sky.is_in(self.valid_act_idn_sky))

    def _build_windows(self, sequence, min_train_size, train_previous_all, test_size):
        """Convert a sequence (e.g., billing dates or delinquency labels) into rolling windows

        Args:
            sequence: List of values for an account, sorted in chronological order.
                Can be billing dates, delinquency labels, or any other sequential data.
            min_train_size: Minimum number of elements required in a window.
            train_previous_all: If True, use expanding windows (all previous elements).
                If False, use fixed-size rolling windows.
            test_size: Maximum number of windows to label as "test". The rest are labeled "train".

        Returns:
            List of tuples (window, split), where:
                - window: List of elements from the sequence
                - split: Either "train" or "test"
        """
        # build the windows
        if not train_previous_all:
            # fixed-size rolling windows: each window contains exactly min_train_size elements
            windows = [sequence[i : i + min_train_size] for i in range(0, len(sequence) - min_train_size + 1)]
        else:
            # expanding windows: each window includes all previous elements up to current position
            windows = [sequence[:i] for i in range(min_train_size, len(sequence) + 1)]

        # label each window as train or test
        # - the last `actual_test_size` windows are labeled as "test"
        # - all earlier windows are labeled as "train"
        # - ensure at least one window remains for training
        actual_test_size = min(test_size, len(windows) - 1)
        train_end = len(windows) - actual_test_size

        for i, window in enumerate(windows):
            if i < train_end:
                windows[i] = (window, "train")
            else:
                windows[i] = (window, "test")

        return windows

    def assemble_samples(self):
        """
        Assemble training and testing samples by creating rolling windows of billing cycles
        for each account.

        For each account, this method:
        1. Extracts the billing dates and delinquency labels
        2. Creates rolling windows from these sequences
        3. Labels each window as "train" or "test"
        4. Creates a sample for each window with the last cycle's delinquency as the target

        Returns:
            pl.DataFrame: A dataframe where each row is a sample containing:
                - split (str): "train" or "test"
                - act_idn_sky: Account ID
                - billing_dates (list): List of billing dates in the window
                - target_delinquency (bool): Delinquency label for the last cycle in the window
        """

        # partition the transaction dataset by account
        transaction_by_act = (
            self.transaction
            # select relevant columns and remove duplicates
            .select(c.act_idn_sky, c.billing_date, c.bank_delinquency_label)
            .sort(c.act_idn_sky, c.billing_date)
            .unique(["act_idn_sky", "billing_date"])
            # partition into dict: {(act_idn_sky,): DataFrame(billing_date, bank_delinquency_label)}
            .partition_by("act_idn_sky", maintain_order=True, as_dict=True, include_key=False)
        )

        # initialize the results
        samples = []

        for (act_idn_sky,), (billing_dates, bank_delinquency_labels) in transaction_by_act.items():
            billing_dates = billing_dates.to_list()
            bank_delinquency_labels = bank_delinquency_labels.to_list()

            # skip accounts with insufficient billing history
            if len(billing_dates) < self.min_train_size:
                continue

            # create rolling windows from both billing dates and delinquency labels
            # each window is a tuple: (list_of_values, split_label)
            billing_dates_windows = self._build_windows(
                billing_dates, self.min_train_size, self.train_previous_all, self.test_size
            )
            delinquency_windows = self._build_windows(
                bank_delinquency_labels, self.min_train_size, self.train_previous_all, self.test_size
            )

            # iterate over windows to construct samples
            for (billing_date_window, split), (delinquency_window, _) in zip(
                billing_dates_windows, delinquency_windows
            ):
                # billing_date_window contains dates like [2017-01, 2017-02, ..., 2017-06]
                # delinquency_window contains corresponding labels like [False, False, ..., True]

                # initialize the sample with split label, account ID, and billing dates
                sample = {"split": split, "act_idn_sky": act_idn_sky, "billing_dates": billing_date_window}

                # get the delinquency status of the last cycle as the target delinquency label
                sample["target_delinquency"] = delinquency_window[-1]

                # add the sample to the results
                samples.append(sample)

        # convert the results to a dataframe
        return pl.DataFrame(samples)

    def build(self):
        """
        Build the complete dataset by assembling samples from transaction data.

        This method orchestrates the dataset creation process:
        1. Calls assemble_samples() to create rolling window samples
        2. Reports dataset statistics (total samples, train/test split)
        3. Saves the dataset to disk in Feather format

        Returns:
            pl.DataFrame: The assembled dataset with train/test splits.
        """
        print("Assembling samples...")
        samples = self.assemble_samples()

        print(f"Dataset built: {len(samples)} samples")
        print(f"\tTrain: {(samples['split'] == 'train').sum()}, Test: {(samples['split'] == 'test').sum()}")

        # save the dataset
        save_path = (
            self.data_dir
            + f"/samples_min{self.config['min_train_size']}mo_{'allprevious' if self.config['train_previous_all'] else 'fixed'}_{self.config['test_size']}test.feather"
        )
        samples.write_ipc(save_path, compression="lz4")
        print(f"Dataset saved to {save_path}")

        return samples


if __name__ == "__main__":
    # Common settings
    data_dir = str(paths.processed_data_dir / "sample_index")

    # Define varying parameters for each configuration
    configs = [
        {"min_train_size": 6, "train_previous_all": False, "test_size": 2},
        {"min_train_size": 6, "train_previous_all": True, "test_size": 2},
        {"min_train_size": 12, "train_previous_all": False, "test_size": 2},
        {"min_train_size": 12, "train_previous_all": True, "test_size": 2},
        {"min_train_size": 12, "train_previous_all": False, "test_size": 1},
        {"min_train_size": 12, "train_previous_all": True, "test_size": 1},
    ]

    # Build and save datasets for all configurations
    for i, params in enumerate(configs, 1):
        config = params | {"data_dir": data_dir}

        print(f"\n{'=' * 80}")
        print(f"Building dataset {i}/{len(configs)}")
        print(
            f"Config: min_train_size={config['min_train_size']}, "
            f"train_previous_all={config['train_previous_all']}, "
            f"test_size={config['test_size']}"
        )
        print(f"{'=' * 80}\n")

        dataset = SampleIndexDataset(config)
        samples = dataset.build()
