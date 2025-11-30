import pandas as pd
import torch
import os
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset
from rouge_score import rouge_scorer
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../data"))
SPLIT_DIR = os.path.join(DATA_DIR, "split-data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_RUN_DIR = os.path.join(MODELS_DIR, "t5-summarizer")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tunable defaults
MAX_INPUT_TOKENS = int(os.getenv("T5_MAX_INPUT", "512"))
MAX_TARGET_TOKENS = int(os.getenv("T5_MAX_TARGET", "160"))
NUM_EPOCHS = int(os.getenv("T5_EPOCHS", "3"))  # Back to 3 epochs
GEN_BEAMS = int(os.getenv("T5_NUM_BEAMS", "2"))
EARLY_STOP_PATIENCE = int(os.getenv("T5_EARLY_STOP", "1"))


def load_data():
    print("\n[STEP 1/3] Loading data splits...")
    train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv")
    val_df = pd.read_csv(f"{SPLIT_DIR}/val.csv")
    test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")

    print(f"✓ Train: {len(train_df)} examples")
    print(f"✓ Val:   {len(val_df)} examples")
    print(f"✓ Test:  {len(test_df)} examples")

    return train_df, val_df, test_df


def prepare_datasets(train_df, val_df, test_df, tokenizer):
    print("\n[STEP 2/3] Tokenizing datasets...")

    def tokenize_function(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            padding=False
        )

        labels = tokenizer(
            batch["target_summary"],
            max_length=MAX_TARGET_TOKENS,
            truncation=True,
            padding=False
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    num_proc = 2  

    def tokenize_split(ds, desc):
        return ds.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            desc=desc
        )

    train_dataset = tokenize_split(train_dataset, "Tokenizing train")
    val_dataset = tokenize_split(val_dataset, "Tokenizing val")
    # Don't need to tokenize test for training-only
    # test_dataset = tokenize_split(test_dataset, "Tokenizing test")

    print("✓ Tokenization complete")

    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset, model, tokenizer):
    print("\n[STEP 3/3] Training model...")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_RUN_DIR,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_safetensors=True,
        label_smoothing_factor=0.1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=2,
        optim="adafactor",
        logging_strategy="steps",  # Enable progress tracking
        logging_steps=50,  # Update every 50 steps (minimal overhead)
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # No validation dataset needed
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=None,  # Removed early stopping since no eval
    )

    print("Starting training...")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Training examples: {len(train_dataset)}")
    print("Progress bar will update every 50 steps")
    print()
    
    trainer.train()

    # Save final model
    final_path = os.path.join(MODEL_RUN_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n✓ Model saved to {final_path}")

    return trainer


def main():
    print("=" * 60)
    print("T5 SUMMARIZER - TRAINING ONLY")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_RUN_DIR, exist_ok=True)

    train_df, val_df, test_df = load_data()

    print("\nInitializing model from google/flan-t5-base...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

    train_dataset, val_dataset = prepare_datasets(
        train_df, val_df, test_df, tokenizer
    )

    trainer = train_model(train_dataset, val_dataset, model, tokenizer)

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTrained model saved in: {MODEL_RUN_DIR}/final/")
    print("\nTo evaluate the model, use the evaluation script.")
    print("=" * 60)


if __name__ == "__main__":
    main()