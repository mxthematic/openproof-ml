"""Supervised fine-tuning for tactic prediction.

Usage:
    python -m openproof_ml.training.sft --config configs/sft_qwen35_2b.yaml
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from ..utils.config import load_config

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(cfg: dict):
    """Load base model and tokenizer, optionally with LoRA."""
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    dtype = model_cfg.get("dtype", "bfloat16")

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if model_cfg.get("use_lora", False):
        lora_config = LoraConfig(
            r=model_cfg.get("lora_rank", 64),
            lora_alpha=model_cfg.get("lora_alpha", 128),
            target_modules=model_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # Setup wandb
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg:
        import wandb
        wandb.init(project=wandb_cfg.get("project"), name=wandb_cfg.get("name"))

    model, tokenizer = build_model_and_tokenizer(cfg)
    max_length = data_cfg.get("max_seq_length", 2048)
    prompt_field = data_cfg.get("prompt_field", "prompt")
    completion_field = data_cfg.get("completion_field", "completion")

    # Load dataset
    data_files = {"train": data_cfg["train_file"]}
    val_file = data_cfg.get("val_file")
    if val_file and Path(val_file).exists():
        data_files["validation"] = val_file
    dataset = load_dataset("json", data_files=data_files)

    # Tokenize: concat prompt + completion, create labels with prompt masked
    def tokenize(example):
        text = example[prompt_field] + example[completion_field] + tokenizer.eos_token
        tokenized = tokenizer(text, truncation=True, max_length=max_length)

        # Mask prompt tokens in labels (set to -100)
        prompt_tokenized = tokenizer(example[prompt_field], add_special_tokens=False)
        prompt_len = len(prompt_tokenized["input_ids"])

        labels = tokenized["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels

        return tokenized

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy="steps" if "eval_steps" in train_cfg else "no",
        eval_steps=train_cfg.get("eval_steps"),
        seed=train_cfg.get("seed", 42),
        report_to="wandb" if wandb_cfg else "none",
        dataloader_num_workers=4,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Resume from checkpoint if one exists
    output_dir = Path(train_cfg["output_dir"])
    last_checkpoint = None
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting SFT training")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info(f"Model saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
