import pandas as pd
import torch
import os
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
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
final_model_path = os.path.join(MODEL_RUN_DIR, "final")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Settings
MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 160
GEN_BEAMS = 2


def load_test_data():
    """Load test dataset"""
    print("\n[STEP 1/3] Loading test data...")
    test_df = pd.read_csv(f"{SPLIT_DIR}/test.csv")  # reuse pre-split CSV from prepare_data
    print(f"✓ Test examples: {len(test_df)}")
    return test_df


def prepare_test_dataset(test_df, tokenizer):
    print("\n[STEP 2/3] Tokenizing test data...")
    # map raw text columns into token IDs + labels so evaluation mirrors training format
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
    
    test_dataset = Dataset.from_pandas(test_df)
    
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,            # modest parallelism keeps CPU usage manageable
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test set"
    )
    
    print("✓ Tokenization complete")
    return test_dataset


def evaluate_model(model, test_dataset, tokenizer, test_df):
    """Evaluate model and generate predictions"""
    print("\n[STEP 3/3] Evaluating model...")
    model.eval()
    
    decoded_preds = []
    decoded_refs = []
    
    batch_size = 8
    total_examples = len(test_dataset)
    
    print(f"Total test examples: {total_examples}")
    print(f"Batch size: {batch_size}")
    print("Starting evaluation...\n")
    
    with torch.no_grad():
        for i in range(0, total_examples, batch_size):
            # Progress tracking
            progress_pct = (i / total_examples) * 100
            print(f"Progress: {i}/{total_examples} ({progress_pct:.1f}%) - Batch {i//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size}")
            
            # Get batch
            batch_end = min(i + batch_size, total_examples)
            batch_slice = test_dataset[i:batch_end]
            
            # Convert dict of lists to list of dicts
            batch_list = [
                {key: batch_slice[key][j] for key in batch_slice.keys()}
                for j in range(len(batch_slice['input_ids']))
            ]
            
            # Prepare batch inputs
            max_len = max(len(ex["input_ids"]) for ex in batch_list)
            input_ids = []
            attention_masks = []
            
            for ex in batch_list:
                ids = ex["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(ex["input_ids"]))
                mask = ex["attention_mask"] + [0] * (max_len - len(ex["attention_mask"]))
                input_ids.append(ids)
                attention_masks.append(mask)
            
            # Convert to tensors
            input_ids = torch.tensor(input_ids).to(model.device)
            attention_masks = torch.tensor(attention_masks).to(model.device)
            
            # Generate summaries
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=MAX_TARGET_TOKENS,
                num_beams=GEN_BEAMS,
                length_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Decode predictions
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_preds.extend(batch_preds)
            
            # Decode references
            for ex in batch_list:
                labels = [tokenizer.pad_token_id if l == -100 else l for l in ex["labels"]]
                decoded_refs.append(tokenizer.decode(labels, skip_special_tokens=True))
    
    print(f"\n✓ Generation complete! Processed {len(decoded_preds)} examples")
    
    # Calculate ROUGE scores
    print("\nCalculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []
    
    for idx, (pred, ref) in enumerate(zip(decoded_preds, decoded_refs)):
        if idx % 50 == 0 and idx > 0:
            print(f"  Calculated ROUGE for {idx}/{len(decoded_preds)} examples...")
        
        score = scorer.score(ref, pred)
        r1_scores.append(score["rouge1"].fmeasure)
        r2_scores.append(score["rouge2"].fmeasure)
        rl_scores.append(score["rougeL"].fmeasure)
    
    rouge_metrics = {
        "test_ROUGE-1": float(np.mean(r1_scores)),
        "test_ROUGE-2": float(np.mean(r2_scores)),
        "test_ROUGE-L": float(np.mean(rl_scores))
    }
    
    print(f"\n✓ ROUGE scores calculated!")
    print(f"  ROUGE-1: {rouge_metrics['test_ROUGE-1']:.4f}")
    print(f"  ROUGE-2: {rouge_metrics['test_ROUGE-2']:.4f}")
    print(f"  ROUGE-L: {rouge_metrics['test_ROUGE-L']:.4f}")
    
    return rouge_metrics, decoded_preds, decoded_refs


def save_results(results, predictions, references, test_df):
    """Save evaluation results and predictions"""
    print("\nSaving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    pd.DataFrame([results]).to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Save predictions
    output_df = test_df.copy()
    output_df["predicted_summary"] = predictions
    output_df["reference_summary"] = references
    
    preds_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    output_df.to_csv(preds_path, index=False)
    print(f"✓ Predictions saved to: {preds_path}")


def main():
    print("=" * 60)
    print("T5 SUMMARIZER - EVALUATION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(final_model_path):
        print(f"\n❌ ERROR: Model not found at {final_model_path}")
        print("Please train the model first using train.py")
        return
    
    # Load test data
    test_df = load_test_data()
    
    # Load trained model
    print("\nLoading trained model...")
    tokenizer = T5Tokenizer.from_pretrained(final_model_path)
    model = T5ForConditionalGeneration.from_pretrained(final_model_path).to(device)
    print(f"✓ Model loaded from: {final_model_path}")
    
    # Prepare test dataset
    test_dataset = prepare_test_dataset(test_df, tokenizer)
    
    # Evaluate
    results, predictions, references = evaluate_model(model, test_dataset, tokenizer, test_df)
    
    # Save results
    save_results(results, predictions, references, test_df)
    
    print("\n" + "=" * 60)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print(f"  - evaluation_metrics.csv (ROUGE scores)")
    print(f"  - test_predictions.csv (all predictions)")
    print("=" * 60)


if __name__ == "__main__":
    main()