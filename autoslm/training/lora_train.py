import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# =========================
# CONFIG
# =========================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "artifacts/dataset.jsonl"
OUTPUT_DIR = "artifacts/adapter"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Base model (GPU + FP16)
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # IMPORTANT for Phi-3 stability
    )

    # REQUIRED: disable KV cache for training
    model.config.use_cache = False

    # -------------------------
    # LoRA config (Phi-3 compatible)
    # -------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # Dataset
    # -------------------------
    dataset = load_dataset(
        "json",
        data_files={"train": DATASET_PATH},
    )

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset["train"].map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -------------------------
    # Training arguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,          # increase later
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        optim="adamw_torch",
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    # -------------------------
    # Save adapter
    # -------------------------
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
