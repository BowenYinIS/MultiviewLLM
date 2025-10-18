import functools
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
from datasets import Dataset
from polars import col as c
from src.config.paths import paths
from src.utils.templates import templates
from tqdm import tqdm
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


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


def _count_tokens(tokenizer, text):
    """Count tokens in text using the tokenizer"""
    return len(tokenizer.encode(text))


def _print_sample_info(
    llm_name, sample_path, samples, max_transaction_tokens, max_tokens_seen, truncation_count, total_prompts
):
    """Print the prompt example with truncation statistics"""
    print("=" * 60)
    print(f"Building Prompts...({datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)
    print()
    print(f"[USING LLM]: {llm_name}\n")
    print(f"[USING SAMPLES]: {sample_path.stem} (N={len(samples)})\n")
    print(f"[TOKEN LIMITS]:")
    print(f"  - Max transaction tokens: {max_transaction_tokens:,}")
    print(f"  - Max prompt tokens seen: {max_tokens_seen:,}")

    if truncation_count > 0:
        print(f"\n⚠️  TRUNCATION: {truncation_count}/{total_prompts} prompts had transaction history truncated")
    else:
        print(f"\n✓ No truncation needed for {total_prompts} prompts\n\n")


@timeit
def init_model(llm_name):
    """Initialize model and tokenizer with LoRA adapters"""
    # Determine chat template based on model name
    if "qwen3" in llm_name.lower() and "instruct" in llm_name.lower():
        chat_template = "qwen3-instruct"
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")

    # Load model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name=llm_name,
        max_seq_length=32768,
        device_map="cuda:0",
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Add LoRA adapters to model
    model = FastModel.get_peft_model(
        model,
        r=32,
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
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Add chat template to tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    return model, tokenizer


@timeit
def build_dataset(
    tokenizer,
    sample_path,
    split,
    max_prompts,
    has_protected_attributes,
    is_cot_prompt,
    max_transaction_tokens,
    llm_name,
):
    """Build dataset from samples"""
    # Load samples
    samples = pl.read_ipc(sample_path, memory_map=False)

    # Select the split
    samples = samples.filter(c.split.is_in(split))

    # Limit the number of prompts
    if max_prompts:
        samples = samples[:max_prompts]

    # Load the prompt template
    sys_msg_template = templates.get_purellm_sys_msg(is_cot_prompt)
    user_msg_template = templates.get_purellm_user_msg(has_protected_attributes)

    # Build the prompts
    prompts = []
    truncation_count = 0
    max_tokens_seen = 0

    for sample in tqdm(samples.iter_rows(named=True), total=len(samples), desc="Building prompts"):
        # Get the ground truth
        target_delinquency = sample["target_delinquency"]

        # Build the system message
        sys_msg = sys_msg_template

        # Truncate transaction history if needed
        transaction_text = sample["transaction_text"]

        # truncate the transaction text
        transaction_text = transaction_text[: (max_transaction_tokens * 2)]

        # Build the user message
        user_msg = user_msg_template.format(
            lvl_4_bch_nam=sample["lvl_4_bch_nam"],
            residence=sample["residence"],
            industry=sample["industry"],
            education=sample["education"],
            birth_year=sample["birth_year"],
            sex=sample["sex"],
            marriage_status=sample["marriage_status"],
            transaction_history=transaction_text,
        )

        # Assemble the final prompt
        prompt = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"{{'is_delinquent': {target_delinquency}}}"},
        ]
        prompts.append(prompt)

    # Convert the prompts to dataset
    dataset = Dataset.from_dict({"conversations": prompts})

    # Format the prompts
    def format_prompts(examples):
        """Format conversations into text using chat template"""
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    # Print sample info
    _print_sample_info(
        llm_name, sample_path, samples, max_transaction_tokens, max_tokens_seen, truncation_count, len(prompts)
    )

    return dataset, prompts, samples


@timeit
def init_trainer(model, tokenizer, dataset):
    """Initialize SFT trainer"""
    # Create training arguments
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=1e-5,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",
            max_seq_length=32768,
            max_grad_norm=1.0,  # Gradient norm clipping
        ),
    )

    # Ensure to only train on completion
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    return trainer


def finetune(config):
    """Main finetuning pipeline"""
    # Extract config values
    llm_name = config["llm_name"]
    sample_path = paths.processed_data_dir / config["sample_path"]
    split = config["split"]
    has_protected_attributes = config["has_protected_attributes"]
    is_cot_prompt = config["is_cot_prompt"]
    max_transaction_tokens = config["max_transaction_tokens"]
    max_prompts = config["max_prompts"]

    # Initialize model and tokenizer
    model, tokenizer = init_model(llm_name)

    # Build dataset
    dataset, prompts, samples = build_dataset(
        tokenizer,
        sample_path,
        split,
        max_prompts,
        has_protected_attributes,
        is_cot_prompt,
        max_transaction_tokens,
        llm_name,
    )

    # Initialize trainer
    trainer = init_trainer(model, tokenizer, dataset)

    # Start training
    trainer_stats = trainer.train()

    # save the checkpoint
    model.save_pretrained_gguf(paths.checkpoint_dir / "benchmark/finetune", tokenizer)

    return trainer_stats


def main():
    config = {
        "is_cot_prompt": False,
        "has_protected_attributes": False,
        "llm_name": "unsloth/Qwen3-4B-Instruct-2507",
        "max_transaction_tokens": 28500,
        "max_prompts": None,
        "sample_path": "llm_benchmark/samples_min6mo_fixed_2test.feather",
        "split": ["test"],
    }

    finetune(config)


if __name__ == "__main__":
    main()
