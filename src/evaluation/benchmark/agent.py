import functools
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import tiktoken
from openai import OpenAI
from polars import col as c
from polars import lit
from src.config.paths import paths
from src.utils.templates import templates
from tqdm import tqdm


class Agent:
    def __init__(self, config):
        self.config = config
        self.sample_path = config["sample_path"]
        self.split = config["split"]
        self.max_prompts = config["max_prompts"]

    def run(self):
        a1_output = self.a1()
        a2_output = self.a2(a1_output)
        a3_output = self.a3(a1_output, a2_output)
        a4_output = self.a4(a1_output, a2_output, a3_output)
        a5_output = self.a5(a1_output, a2_output, a3_output, a4_output)

        return a5_output

    def a1(self):
        """Construct the output of A1

        To save time, we manually construct the output, skipping the LLM
        """
        # load the samples
        samples = pl.read_ipc(self.sample_path, memory_map=False)

        # select the split
        samples = samples.filter(c.split.is_in(self.split))

        # limit the number of samples
        if self.max_prompts:
            samples = samples[:self.max_prompts]

        # calcualte per-cycle summaries

    def a2(self):
        pass

    def a3(self):
        pass

    def a4(self):
        pass

    def a5(self):
        pass

    def _calculate_per_cycle_summaries(self, df):
        df