import functools
import json5
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time
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
        self.transaction_path = paths.raw_data_dir / config["transaction_path"]
        self.sample_path = paths.processed_data_dir / config["sample_path"]
        self.transaction_text_type = config["transaction_text_type"]
        self.llm_name = config["llm_name"]

        # load the transactions and samples
        self.load_data()

        # initialize the LLM
        self.initialize_llm()

    def run(self):
        """Run the agent for all samples"""
        predictions = []

        for sample in self.samples.iter_rows(named=True):
            pred = self.infer_one_sample(sample)
            predictions.append(pred)

        # save the predictions
        self.save_predictions(predictions)

    def load_data(self):
        """Load the transactions and samples"""
        # load the samples
        samples = pl.read_ipc(self.sample_path, memory_map=False)

        # select the split
        samples = samples.filter(c.split.is_in(self.split))

        # limit the number of samples
        if self.max_prompts:
            samples = samples[: self.max_prompts]

        # save samples as class attribute
        self.samples = samples

        # load the transactions
        self.transactions = pl.read_ipc(self.transaction_path, memory_map=False)

    def initialize_llm(self):
        """
        Initialize the language model based on configuration
        """
        self.llm = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8080/v1",
        )

        # Verify the model running on llama.cpp matches the configured model
        self._verify_model_match()

        if "qwen3" in self.llm_name.lower():
            if "instruct" in self.llm_name.lower():
                self.generation_params = {
                    "model": self.llm_name,
                    "temperature": 0.7,
                    "top_p": 0.80,
                    "presence_penalty": 1.0,
                    "response_format": {"type": "json_object"},
                }
            elif "thinking" in self.llm_name.lower():
                self.generation_params = {
                    "model": self.llm_name,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "presence_penalty": 1.0,
                }
        elif "qwen2" in self.llm_name.lower():  # including qwen2.5
            self.generation_params_logprobs = {
                "model": self.llm_name,
                "logprobs": True,
                "top_logprobs": 10,
            }
            self.generation_params = {
                "model": self.llm_name,
            }
        else:
            raise ValueError(f"Unsupported model: {self.llm_name}")

    def infer_one_sample(self, sample):
        """
        Args:
            sample (dict): A dict containing user profile & billing dates for a single sample
        """
        profile, transaction_text = self.prepare_raw_input(sample)
        a1_output, a1_sys_msg, a1_user_msg = self.a1(transaction_text)  # extract transaction features
        a2_output, a2_sys_msg, a2_user_msg = self.a2(transaction_text, a1_output)  # analyze behavior patterns
        a3_output, a3_sys_msg, a3_user_msg = self.a3(
            transaction_text, a1_output, a2_output
        )  # counterfactual stress tests
        a4_output, a4_sys_msg, a4_user_msg = self.a4(
            profile, transaction_text, a1_output, a2_output, a3_output
        )  # final decision

        return {
            "a1_output": a1_output,
            "a1_sys_msg": a1_sys_msg,
            "a1_user_msg": a1_user_msg,
            "a2_output": a2_output,
            "a2_sys_msg": a2_sys_msg,
            "a2_user_msg": a2_user_msg,
            "a3_output": a3_output,
            "a3_sys_msg": a3_sys_msg,
            "a3_user_msg": a3_user_msg,
            "a4_output": a4_output,
            "a4_sys_msg": a4_sys_msg,
            "a4_user_msg": a4_user_msg,
        }

    def prepare_raw_input(self, sample):
        """Prepare the raw input for the sample
        Args:
            sample (dict): A dict containing user profile & billing dates for a single sample
                - split (str): "train" or "test"
                - act_idn_sky (int): The account ID
                - billing_dates (list): List of billing dates in the window
                - target_delinquency (bool): Delinquency label for the last cycle in the window
                - lvl_4_bch_nam (str): The name of the branch
                - residence (str): The residence of the user
                - industry (str): The industry of the user
                - education (str): The education of the user
                - birth_year (int): The birth year of the user
                - sex (str): The sex of the user
                - marriage_status (str): The marriage status of the user
                - transaction_text (str): Concatenated transaction text for all cycles in the window
                - transaction_summary (str): Summary of the transaction text
        """
        profile = templates._promptcast_user_msg_1.format(
            lvl_4_bch_nam=sample["lvl_4_bch_nam"],
            residence=sample["residence"],
            industry=sample["industry"],
            education=sample["education"],
        )
        transaction_text = sample[f"transaction_text_{self.transaction_text_type}"]
        return profile, transaction_text

    def a1(self, transaction_text):
        """Construct the output of A1

        Extract transaction features from the transaction text.
        Args:
            transaction_text (str): The transaction text
        Returns:
            dict: The extracted transaction features
        """

        # construct the messages
        sys_msg = templates.get_agent_sys_msg("a1")
        user_msg = templates.get_agent_user_msg("a1", transaction_text=transaction_text)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        # call the LLM
        response = self.llm.chat.completions.create(messages=messages, **self.generation_params).choices[0]
        a1_output = response.message.content

        return a1_output, sys_msg, user_msg

    def a2(self, transaction_text, a1_output):
        """Construct the output of A2
        Args:
            transaction_text (str): The transaction text
            a1_output (dict): The output of A1
        Returns:
            dict: The output of A2
        """
        # construct the messages
        sys_msg = templates.get_agent_sys_msg("a2")
        user_msg = templates.get_agent_user_msg("a2", transaction_text=transaction_text, agent_a1_output=a1_output)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        # call the LLM
        response = self.llm.chat.completions.create(messages=messages, **self.generation_params).choices[0]
        a2_output = response.message.content

        # return the response
        return a2_output, sys_msg, user_msg

    def a3(self, transaction_text, a1_output, a2_output):
        """Construct the output of A3"""
        # construct the messages
        sys_msg = templates.get_agent_sys_msg("a3")
        user_msg = templates.get_agent_user_msg(
            "a3", transaction_text=transaction_text, agent_a1_output=a1_output, agent_a2_output=a2_output
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        # call the LLM
        response = self.llm.chat.completions.create(messages=messages, **self.generation_params).choices[0]
        a3_output = response.message.content

        # return the response
        return a3_output, sys_msg, user_msg

    def a4(self, profile, transaction_text, a1_output, a2_output, a3_output):
        """Construct the output of A4"""
        # construct the messages
        sys_msg = templates.get_agent_sys_msg("a4")
        user_msg = templates.get_agent_user_msg(
            "a4",
            user_profile=profile,
            transaction_text=transaction_text,
            agent_a1_output=a1_output,
            agent_a2_output=a2_output,
            agent_a3_output=a3_output,
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        # call the LLM
        response = self.llm.chat.completions.create(messages=messages, **self.generation_params_logprobs).choices[0]
        a4_output = self._parse_response_as_json(response.message.content)

        # extract logits (if available)
        logprobs_data = []
        if getattr(response, "logprobs", None) and response.logprobs.content:
            for token_logprob in response.logprobs.content:
                token_info = {
                    "token": token_logprob.token,
                    "logprob": token_logprob.logprob,
                    "top_candidates": [
                        {"token": candidate.token, "logprob": candidate.logprob}
                        for candidate in token_logprob.top_logprobs
                    ],
                }
                logprobs_data.append(token_info)

        # add logprobs to the output
        a4_output["logprobs"] = logprobs_data

        return a4_output, sys_msg, user_msg

    def save_predictions(self, predictions):
        """
        Save the predictions to a feather file
        """
        # verify the number of predictions is equal to the number of samples
        if len(predictions) != len(self.samples):
            raise ValueError(f"Number of predictions ({len(predictions)}) is not equal to the number of samples ({len(self.samples)})")

        # convert the responses to a dataframe
        preds = pl.DataFrame(predictions)
        preds = preds.rename({col: f"pred_{col}" for col in preds.columns})

        # add sample to the output table
        preds = pl.concat([self.samples, preds], how="horizontal")

        # add config to the output table (use lit() to avoid column reference issues)
        config_cols = {f"config_{key}": lit(value) for key, value in self.config.items() if key != "split"}
        preds = preds.with_columns(**config_cols)

        # save the preds table
        save_path = (
            self.sample_path.parent
            / f"preds_agent_{datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')}.feather"
        )
        preds.write_ipc(str(save_path), compression="lz4")
        print("-" * 30)
        print(f"Saved predictions to {save_path.name}\n\n\n")

    def _parse_response_as_json(self, response):
        """Parse the response as JSON
        Remove markdown code block and not including the "```json" and "```" tags
        Args:
            response (str): The response from the LLM
        Returns:
            dict: The parsed response
        """
        response = response.replace("```json", "").replace("```", "")
        return json5.loads(response)

    def _verify_model_match(self):
        """
        Verify that the model name in config matches the model running on llama.cpp server.
        Raises ValueError if they don't match.
        """
        try:
            models_response = self.llm.models.list()
            server_models = [model.id for model in models_response.data]
            server_model = server_models[0]  # llama.cpp typically serves one model

            # Check if the configured model matches the server model
            if self.llm_name.split("/")[-1].split(":")[0] not in server_model:
                raise ValueError(
                    f"Model mismatch!\n"
                    f"  Configured model: {self.llm_name}\n"
                    f"  Server model:     {server_model}\n"
                    f"Please update the config or restart llama.cpp with the correct model."
                )

            print(f"✓ Model verification passed: {self.llm_name}")

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to verify model on llama.cpp server: {e}")


def main():
    config = {
        "transaction_path": "sample_transaction.feather",
        "sample_path": "llm_benchmark/samples_min6mo_fixed_2test.feather",
        "transaction_text_type": "detail_3",
        "split": ["test"],
        "max_prompts": 5,
        "llm_name": "qwen2.5-7b-instruct",
    }
    # initialize the agent
    agent = Agent(config)

    # run the agent
    agent.run()


if __name__ == "__main__":
    main()
