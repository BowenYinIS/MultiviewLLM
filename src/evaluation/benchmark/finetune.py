from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
from datasets import Dataset
from polars import col as c
from src.config.paths import paths
from src.utils.templates import templates
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer


class Finetune:
    def __init__(self, config):
        self.config = config
        self.llm_name = config["llm_name"]
        self.sample_path = paths.processed_data_dir / config["sample_path"]
        self.split = config["split"]
        self.has_protected_attributes = config["has_protected_attributes"]
        self.is_cot_prompt = config["is_cot_prompt"]
        self.max_transaction_tokens = config["max_transaction_tokens"]
        self.max_prompts = config["max_prompts"]

        # create LLM-specific config
        if "qwen3" in self.llm_name.lower() and "instruct" in self.llm_name.lower():
            self.chat_template = "qwen3-instruct"
        else:
            raise ValueError(f"Unsupported LLM: {self.llm_name}")

    def finetune(self):
        # initialize model and tokenizer
        self.init_model()

        # build dataset
        self.build_dataset()

        # initialize trainer
        self.init_trainer()

        # start training
        trainer_stats = self.trainer.train()

    def init_model(self):
        # load model and tokenizer
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.llm_name,
            max_seq_length=32768,  # Choose any for long context!
            load_in_4bit=True,  # 4 bit quantization to reduce memory
            load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning=False,  # [NEW!] We have full finetuning now!
        )

        # add LoRA adapters to model
        self.model = FastModel.get_peft_model(
            self.model,
            r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        # add chat template to tokenizer
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=self.chat_template,
        )

    def build_dataset(self):
        # load samples
        samples = pl.read_ipc(self.sample_path, memory_map=False)

        # select the split
        samples = samples.filter(c.split.is_in(self.split))

        # limit the number of prompts
        if self.max_prompts:
            samples = samples[: self.max_prompts]

        # load the prompt template
        self.sys_msg_template = templates.get_purellm_sys_msg(self.is_cot_prompt)
        self.user_msg_template = templates.get_purellm_user_msg(self.has_protected_attributes)

        # build the prompts
        prompts = []
        truncation_count = 0
        max_tokens_seen = 0

        for sample in samples.iter_rows(named=True):
            # get the ground truth
            target_delinquency = sample["target_delinquency"]

            # build the system message
            sys_msg = self.sys_msg_template

            # Truncate transaction history if needed
            transaction_text = sample["transaction_text"]
            # transaction_text, was_truncated = self._truncate_transaction_history(
            #     sample["transaction_text"], max_tokens=self.max_transaction_tokens
            # )

            # if was_truncated:
            #     truncation_count += 1

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
            # prompt_tokens = self._count_tokens(sys_msg + user_msg)
            # max_tokens_seen = max(max_tokens_seen, prompt_tokens)

            # assemble the final prompt
            prompt = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{{'is_delinquent': {target_delinquency}}}"},
            ]
            prompts.append(prompt)

        # convert the prompts to dataset
        dataset = Dataset.from_dict({"conversations": prompts})

        # apply chat template to the prompts
        dataset = dataset.map(self._formatting_prompts_func, batched=True)

        # store the prompts and samples as class attribute
        self.dataset = dataset
        self.prompts = prompts
        self.samples = samples

        # print one prompt for debugging
        self._print_sample_info(truncation_count, len(prompts), max_tokens_seen)

    def init_trainer(self):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 60,
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
            ),
        )

        # ensure the it's only trained on the completion part
        self.trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )

    def _print_prompt_masking(self):
        # print an example to check if instruction masking is done
        # let's select sample #2
        print(f"Sample #2:\n{self.tokenizer.decode(self.trainer.train_dataset[2]["input_ids"])}")
        print(f"Sample #2 (after masking):\n{self.tokenizer.decode([self.tokenizer.pad_token_id if x == -100 else x for x in self.trainer.train_dataset[2]["labels"]]).replace(self.tokenizer.pad_token, " ")}")

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

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Input text to count tokens

        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _print_sample_info(self, truncation_count, total_prompts, max_tokens_seen):
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
            print(f"\n✓ No truncation needed for {total_prompts} prompts\n\n")

    def _formatting_prompts_func(self, examples):
        convos = examples["conversations"]
        texts = [
            self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos
        ]
        return {"text": texts}


def main():
    config = {
        "is_cot_prompt": False,
        "has_protected_attributes": False,
        "llm_name": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",  # "unsloth/Qwen3-30B-A3B-Instruct-2507",
        "max_transaction_tokens": 28500,
        "max_prompts": 50,
        "sample_path": "llm_benchmark/samples_min12mo_fixed_2test.feather",
        "split": ["test"],
    }

    finetune = Finetune(config)
    finetune.finetune()


if __name__ == "__main__":
    main()
