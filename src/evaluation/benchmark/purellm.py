import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import tiktoken
from openai import OpenAI
from polars import col as c, lit
from src.config.paths import paths
from src.utils.templates import templates
import socket
from tqdm import tqdm


class PureLLM:
    def __init__(self, config):
        """
        Initialize the inference pipeline with configuration
        """
        # get configurations
        self.config = config
        self.max_prompts = config["max_prompts"]
        self.max_workers = config["max_workers"]
        self.sample_path = paths.processed_data_dir / config["sample_path"]
        self.has_protected_attributes = config["has_protected_attributes"]
        self.llm_name = config["llm_name"]
        self.split = config["split"]
        self.is_cot_prompt = config["is_cot_prompt"]
        self.max_transaction_tokens = config["max_transaction_tokens"]
        self.host = socket.gethostbyname(socket.gethostname())

        # Initialize tiktoken encoder for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def run_inference(self):
        """
        Run LLM inference on the data with comprehensive error handling
        Uses ThreadPoolExecutor to send concurrent requests to llama.cpp server
        """
        # build prompts
        self.build_prompts()

        # initialize the llm
        self.initialize_llm()

        self.responses = []
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.infer_single_sample, prompt): idx for idx, prompt in enumerate(self.prompts)
            }

            # Collect results as they complete with progress bar
            with tqdm(total=len(self.prompts), desc="Inference Progress", unit="sample") as pbar:
                # Store results with their original index to maintain order
                results = [None] * len(self.prompts)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    response = future.result()  # Will raise exception if task failed
                    results[idx] = response
                    pbar.update(1)

                self.responses = results

        # save the results
        self.save_predictions()

    def build_prompts(self):
        """Build the prompts from samples and templates.

        Loads samples from the sample path, applies the prompt templates,
        and creates formatted prompts for each sample.

        Attributes:
            prompts (list[dict]): List of prompt dictionaries, each containing
                'system' and 'user' message keys. Set by this method.
        """
        # load the samples
        samples = pl.read_ipc(self.sample_path, memory_map=False)

        # select the split
        samples = samples.filter(c.split.is_in(self.split))

        # load the prompt template
        self.sys_msg_template = templates.get_purellm_sys_msg(self.is_cot_prompt)
        self.user_msg_template = templates.get_purellm_user_msg(self.has_protected_attributes)

        # build the prompts
        prompts = []
        truncation_count = 0
        max_tokens_seen = 0

        for sample in samples.iter_rows(named=True):
            # build the system message
            sys_msg = self.sys_msg_template

            # Truncate transaction history if needed
            transaction_text, was_truncated = self._truncate_transaction_history(
                sample["transaction_text"], max_tokens=self.max_transaction_tokens
            )

            if was_truncated:
                truncation_count += 1

            # build the user message
            user_msg = self.user_msg_template.format(
                lvl_4_bch_nam=sample["lvl_4_bch_nam"],
                residence=sample["residence"],
                industry=sample["industry"],
                education=sample["education"],
                birth_year=sample["birth_year"],
                sex=sample["sex"],
                marriage_status=sample["marriage_status"],
                transaction_history=transaction_text,
            )

            # Track max tokens for reporting
            prompt_tokens = self._count_tokens(sys_msg + user_msg)
            max_tokens_seen = max(max_tokens_seen, prompt_tokens)

            # assemble the final prompt
            prompt = {
                "sys_msg": sys_msg,
                "user_msg": user_msg,
            }
            prompts.append(prompt)

        # limit the number of prompts
        if self.max_prompts:
            prompts = prompts[: self.max_prompts]
            samples = samples[: self.max_prompts]

        # store the prompts and samples as class attribute
        self.prompts = prompts
        self.samples = samples

        # print one prompt for debugging
        self._print_prompt_example(prompt, truncation_count, len(prompts), max_tokens_seen)

    def initialize_llm(self):
        """
        Initialize the language model based on configuration
        """
        self.llm = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8080/v1",
        )
        if self.llm_name.startswith("unsloth/Qwen3-30B-A3B-Instruct"):
            # Store generation parameters separately
            self.generation_params = {
                "model": self.llm_name,
                "temperature": 0.7,
                "top_p": 0.80,
                "presence_penalty": 1.0,
                "response_format": {"type": "json_object"},
                # Note: llama.cpp server handles top_k and min_p from the server config
            }
        elif self.llm_name.startswith("unsloth/Qwen3-30B-A3B-Thinking"):
            # For Thinking models, DO NOT use response_format to preserve reasoning_content
            # The model will output thinking in reasoning_content, but we need to extract JSON from it
            self.generation_params = {
                "model": self.llm_name,
                "temperature": 0.6,
                "top_p": 0.95,
                "presence_penalty": 1.0,
            }
        else:
            raise ValueError(f"Unsupported model: {self.llm_name}")

    def infer_single_sample(self, prompt):
        """
        Generate response for a single prompt using OpenAI SDK

        Args:
            prompt (dict): Dictionary with 'system' and 'user' keys

        Returns:
            dict: Parsed JSON response dict with reasoning (if available)
        """
        messages = [
            {"role": "system", "content": prompt["sys_msg"]},
            {"role": "user", "content": prompt["user_msg"]},
        ]

        # generate response and message
        response = self.llm.chat.completions.create(messages=messages, **self.generation_params).choices[0]
        message = response.message

        # Extract both reasoning and final answer
        # For Thinking models: reasoning_content has the thinking process, content has final answer
        # For Instruct models: only content is populated
        prediction = message.content
        reasoning_process = getattr(message, "reasoning_content", None)

        # parse response into dict
        parsed_response = json.loads(prediction)
        parsed_response["reasoning_process"] = reasoning_process

        return parsed_response

    def save_predictions(self):
        """
        Save the predictions to a feather file
        """
        # convert the responses to a dataframe
        preds = pl.DataFrame(self.responses)
        preds = preds.rename({col: f"pred_{col}" for col in preds.columns})

        # add prompt to the output table
        prompt_df = pl.DataFrame(self.prompts)
        preds = pl.concat([prompt_df, preds], how="horizontal")

        # add sample to the output table
        preds = pl.concat([self.samples, preds], how="horizontal")

        # add config to the output table (use lit() to avoid column reference issues)
        config_cols = {f"config_{key}": lit(value) for key, value in self.config.items() if key != "split"}
        preds = preds.with_columns(**config_cols)

        # save the preds table
        save_path = (
            self.sample_path.parent
            / f"preds_{datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')}.feather"
        )
        preds.write_ipc(str(save_path), compression="lz4")
        print("-" * 60)
        print(f"Saved predictions to {save_path.name}")

    def _print_prompt_example(self, prompt, truncation_count, total_prompts, max_tokens_seen):
        """
        Print the prompt example with truncation statistics
        """
        print("=" * 60)
        print(f"Building Prompts...({datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 60)
        print()
        print(f"[USING LLM]: {self.llm_name}\n")
        print(f"[USING SAMPLES]: {self.sample_path.stem} (N={len(self.samples)})\n")
        print(f"[TOKEN LIMITS]:")
        print(f"  - Max transaction tokens: {self.max_transaction_tokens:,}")
        print(f"  - Max prompt tokens seen: {max_tokens_seen:,}")

        if truncation_count > 0:
            print(f"\n⚠️  TRUNCATION: {truncation_count}/{total_prompts} prompts had transaction history truncated")
        else:
            print(f"\n✓ No truncation needed for {total_prompts} prompts")

        print()
        print(f"[SYSTEM MESSAGE]:\n\n{self.sys_msg_template}")
        print("-" * 60)
        print()
        print(f"[USER MESSAGE Example]:\n\n{'\n'.join(prompt['user_msg'].splitlines()[:15])}")
        print("...")

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text to count tokens

        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _truncate_transaction_history(self, transaction_text: str, max_tokens: int) -> tuple[str, bool]:
        """
        Truncate transaction history to fit within token limit.
        Keeps most recent transactions (end of text) since they're more relevant.

        Args:
            transaction_text: Full transaction history text
            max_tokens: Maximum tokens allowed for this field

        Returns:
            tuple: (truncated_text, was_truncated)
        """
        current_tokens = self._count_tokens(transaction_text)

        if current_tokens <= max_tokens:
            return transaction_text, False

        # Split by lines to preserve transaction boundaries
        lines = transaction_text.strip().split("\n")

        # Keep from the end (most recent transactions are more relevant)
        truncation_msg = "[... Earlier transactions truncated due to length limit ...]\n\n"
        truncation_msg_tokens = self._count_tokens(truncation_msg)

        # Binary search for the right number of lines
        available_tokens = max_tokens - truncation_msg_tokens

        for i in range(len(lines), 0, -1):
            candidate_text = "\n".join(lines[-i:])
            if self._count_tokens(candidate_text) <= available_tokens:
                return truncation_msg + candidate_text, True

        # If even one line is too long, truncate by tokens directly
        tokens = self.tokenizer.encode(transaction_text)
        truncated_tokens = tokens[-(available_tokens):]
        truncated_text = self.tokenizer.decode(truncated_tokens)

        return truncation_msg + truncated_text, True


def main():
    """
    Main function to run inference with different configurations
    """
    # base inference configuration
    shared_config = {
        "max_prompts": None,
        "max_workers": 16,  # Number of concurrent threads (should match server's -np value - optimal)
        "split": ["test"],
        "llm_name": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL",
        "max_transaction_tokens": 28500,  # Maximum tokens for transaction history
    }

    configs = [
        {
            "is_cot_prompt": True,
            "has_protected_attributes": False,
            "sample_path": "llm_benchmark/samples_min6mo_fixed_2test.feather",
        },
    ]
    for config in configs:
        # build the config
        config = shared_config | config

        # create inference pipeline
        inference_pipeline = PureLLM(config)

        # run inference
        inference_pipeline.run_inference()


if __name__ == "__main__":
    # run the main inference pipeline - fail fast on any error
    main()
