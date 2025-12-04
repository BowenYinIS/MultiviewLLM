# ----------------------------------
# Impliment FinPT
# - inputs: profile + transaction summary (12 cycles) + accumulated N delinquency
# - Qwen2.5 as feature encoder
# - add a MLP layer for classification
# ----------------------------------


import functools
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI
from polars import col as c
from polars import lit
from src.config.paths import paths
from src.utils.templates import templates
from tqdm import tqdm


def timeit(func):
    """Decorator to time function execution and print results"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"✓ {func.__name__} completed in {elapsed_time:.2f} seconds")
        return result

    return wrapper


class FinPT:
    def __init__(self, config):
        """
        Initialize the inference pipeline with configuration
        """
        # get configurations
        self.config = config
        self.max_prompts = config["max_prompts"]

        # Determine max_workers based on hostname if not provided
        if "max_workers" in config:
            self.max_workers = config["max_workers"]
        else:
            hostname = socket.gethostname()
            if hostname == "yu-lerner":
                self.max_workers = 8
            elif hostname == "hopper":
                self.max_workers = 16
            else:
                self.max_workers = 16  # default

        self.sample_path = paths.processed_data_dir / config["sample_path"]

        # set the embeddings filename, either from config or generate a new one
        if config.get("embeddings_filename", None) is None:
            self.embedding_filepath = (
                self.sample_path.parent
                / f"embeds_finpt_{datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')}.pt"
            )
        else:
            self.embedding_filepath = (self.sample_path.parent / config["embeddings_filename"]).with_suffix(".pt")

        # set the other configurations
        self.llm_name = config["llm_name"]
        self.max_transaction_tokens = config["max_transaction_tokens"]
        self.host = socket.gethostbyname(socket.gethostname())
        self.transaction_text_type = config["transaction_text_type"]

        # Initialize tiktoken encoder for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def run(self):
        """
        Run the FinPT pipeline
        """
        # build prompts
        self.build_prompts()

        # initialize the llm
        self.initialize_llm()

        # encode the inputs or load existing encoded inputs
        if self.config.get("embeddings_filename", None) is None:
            self.encode_inputs()
        else:
            print(f"Loading existing embeddings from {self.embedding_filepath.name}")
            self.embeddings = torch.load(self.embedding_filepath)

        # train the classifier
        # self.train_classifier()

    def encode_inputs(self):
        """
        Encode the inputs using the LLM
        Uses ThreadPoolExecutor to send concurrent requests to llama.cpp server
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.infer_single_sample, prompt): idx for idx, prompt in enumerate(self.prompts)
            }

            # Collect results as they complete with progress bar
            with tqdm(total=len(self.prompts), desc="Inference Progress", unit="sample") as pbar:
                # Store results with their original index to maintain order
                embeddings = [None] * len(self.prompts)
                splits = [None] * len(self.prompts)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    response = future.result()  # Will raise exception if task failed
                    splits[idx] = response[0]
                    embeddings[idx] = response[1]
                    pbar.update(1)

        self.embeddings = [{"split": split, "embedding": embedding} for split, embedding in zip(splits, embeddings)]

        # save the encoded inputs
        torch.save(self.embeddings, self.embedding_filepath)

    def train_classifier(self):
        """
        Train a small MLP classifier on the embeddings
        """
        print(f"Starting training classifier...")
        
        # load the embeddings
        embeddings = torch.load(self.embedding_filepath)

        # get the target labels
        targets = torch.tensor(self.samples["target_delinquency"].to_list(), dtype=torch.float32).unsqueeze(1)

        # define the model
        input_dim = embeddings.shape[1]
        output_dim = 1
        model = nn.Sequential(nn.Linear(input_dim, output_dim))

        # training configuration
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # increased lr for simple linear model
        epochs = 100
        batch_size = 128

        # create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(embeddings, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # training loop
        print(f"\nTraining Logistic Regression (Input: {input_dim} -> Output: {output_dim})...")
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_embeddings, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        print("Training completed.")

        # generate the predictions and probabilities
        model.eval()
        with torch.no_grad():
            logits = model(embeddings)
            self.probs = torch.sigmoid(logits).numpy().flatten()
            self.preds = (self.probs > 0.5)

        # save the predictions
        self.save_predictions()

    @timeit
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

        # limit the number of samples
        if self.max_prompts:
            samples = samples[: self.max_prompts]

        # load the prompt template
        self.user_msg_template = templates.get_finpt_user_msg()

        # build the prompts
        prompts = []
        truncation_count = 0
        max_tokens_seen = 0

        for sample in samples.iter_rows(named=True):
            split = sample["split"]

            # get transaction history
            if "summary" in self.transaction_text_type:
                transaction_history = sample[f"transaction_text_{self.transaction_text_type}"]
            else:
                transaction_history, was_truncated = self._truncate_transaction_history(
                    sample[f"transaction_text_{self.transaction_text_type}"], max_tokens=self.max_transaction_tokens
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
                transaction_history=transaction_history,
            )

            # Track max tokens for reporting
            prompt_tokens = self._count_tokens(user_msg)
            max_tokens_seen = max(max_tokens_seen, prompt_tokens)

            # assemble the final prompt
            prompt = {
                "split": split,
                "user_msg": user_msg,
            }
            prompts.append(prompt)

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
            base_url="http://127.0.0.1:8080/v1",
        )

    def infer_single_sample(self, prompt):
        """
        Generate response for a single prompt using OpenAI SDK

        Args:
            prompt (dict): Dictionary with 'system' and 'user' keys

        Returns:
            list: Embedding vector
        """
        # generate response
        response = self.llm.embeddings.create(
            input=prompt["user_msg"],
            model=self.llm_name,
        )

        # get the embedding
        embedding = torch.tensor(response.data[0].embedding)

        return prompt["split"], embedding

    def save_predictions(self):
        """
        Save the embeddings to a feather file
        """
        # convert the emb, prob, and preds to a dataframe
        preds = pl.DataFrame({"pred_embed": self.embeddings.numpy(), "pred_prob": self.probs, "pred_is_delinquency": self.preds})

        # ensure preds and prompt has same number of rows
        if preds.height != len(self.prompts):
            raise ValueError(f"Number of predictions ({preds.height}) is not equal to the number of prompts ({len(self.prompts)})")

        # add prompt to the output table
        prompt_df = pl.DataFrame(self.prompts)
        preds = pl.concat([prompt_df, preds], how="horizontal")

        # add sample to the output table
        preds = pl.concat([self.samples, preds], how="horizontal").select(pl.all().exclude("^transaction_text.*$"))

        # add config to the output table (use lit() to avoid column reference issues)
        config_cols = {f"config_{key}": lit(value) for key, value in self.config.items() if key != "split"}
        preds = preds.with_columns(**config_cols)

        # save the preds table
        save_path = (
            self.sample_path.parent
            / f"preds_finpt_{datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')}.feather"
        )

        preds.write_ipc(str(save_path), compression="lz4")
        print("-" * 30)
        print(f"Saved predictions to {save_path.name}\n\n\n")

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
        "llm_name": "qwen2.5-7b-instruct",
        "max_transaction_tokens": 28500,  # Maximum tokens for transaction history
    }

    configs = [
        {
            "transaction_text_type": "summary_2",  # detail_1, detail_2, summary_1, summary_2
            "sample_path": "llm_benchmark/samples_min12mo_fixed_2test.feather",
            "embeddings_filename": None,  # e.g., "embeds_finpt_20251204_130655", or None if you want to encode from scratch
        },
    ]
    for config in configs:
        # build the config
        config = shared_config | config

        # create inference pipeline
        inference_pipeline = FinPT(config)

        # run inference
        inference_pipeline.run()


if __name__ == "__main__":
    # run the main inference pipeline - fail fast on any error
    main()
