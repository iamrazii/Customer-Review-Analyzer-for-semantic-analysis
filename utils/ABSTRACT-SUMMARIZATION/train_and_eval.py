import pandas as pd
import torch
import os
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from rouge_score import rouge_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    base_path = "../../data/split-data"
    train_df = pd.read_csv(f"{base_path}/train.csv")
    val_df = pd.read_csv(f"{base_path}/val.csv")
    test_df = pd.read_csv(f"{base_path}/test.csv")
    return train_df, val_df, test_df

def prepare_datasets(train_df, val_df, test_df, tokenizer, max_input=256, max_target=64):
    def tokenize_function(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=max_input,
            truncation=True,
            padding=False
        )
        labels = tokenizer(
            batch["target_summary"],
            max_length=max_target,
            truncation=True,
            padding=False
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    num_proc = min(os.cpu_count() or 1, 4)

    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=num_proc,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=num_proc,
        remove_columns=val_dataset.column_names
    )
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=num_proc,
        remove_columns=test_dataset.column_names
    )

    return train_dataset, val_dataset, test_dataset

def train_model(train_dataset, val_dataset, model, tokenizer):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir="../../models/t5-summarizer",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="../../logs",
        logging_steps=20,
        fp16=True,
        gradient_accumulation_steps=2,
        save_total_limit=2,
        report_to="none",
        save_safetensors=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("../../models/t5-summarizer/final")
    tokenizer.save_pretrained("../../models/t5-summarizer/final")
    return trainer

def evaluate_model(test_df, model, tokenizer):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    texts = test_df["input_text"].tolist()
    refs = test_df["target_summary"].tolist()
    predictions = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True,
            padding=True, 
            max_length=256
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_length=64,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    rouge1, rouge2, rougel = [], [], []
    for pred, ref in zip(predictions, refs):
        score = scorer.score(ref, pred)
        rouge1.append(score["rouge1"].fmeasure)
        rouge2.append(score["rouge2"].fmeasure)
        rougel.append(score["rougeL"].fmeasure)

    results = {
        "ROUGE-1": float(np.mean(rouge1)),
        "ROUGE-2": float(np.mean(rouge2)),
        "ROUGE-L": float(np.mean(rougel))
    }

    print("\nTest Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results, predictions, refs

def save_results(results, predictions, references, test_df):
    os.makedirs("../../results", exist_ok=True)
    pd.DataFrame([results]).to_csv("../../results/evaluation_metrics.csv", index=False)
    out_df = test_df.copy()
    out_df["predicted_summary"] = predictions
    out_df.to_csv("../../results/test_predictions.csv", index=False)

def main():
    os.makedirs("../../models/t5-summarizer", exist_ok=True)
    os.makedirs("../../logs", exist_ok=True)

    train_df, val_df, test_df = load_data()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, val_df, test_df, tokenizer,
        max_input=256,
        max_target=64
    )

    trainer = train_model(train_dataset, val_dataset, model, tokenizer)
    results, predictions, references = evaluate_model(test_df, model, tokenizer)
    save_results(results, predictions, references, test_df)

if __name__ == "__main__":
    main()