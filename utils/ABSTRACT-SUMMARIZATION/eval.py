import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load
from tqdm import tqdm
import json
import os

class ModelEvaluator:
    """
    Evaluate trained T5 model on test set
    Calculate ROUGE and BLEU scores
    """
    
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to trained model directory
            device: 'cuda' or 'cpu'
        """
        print("="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)
        
        print(f"\n[1/5] Loading trained model from: {model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded on: {self.device}")
        
        # Load metrics
        print(f"\n[2/5] Loading evaluation metrics...")
        self.rouge = load("rouge")
        self.bleu = load("bleu")
        print("✓ ROUGE and BLEU metrics loaded")
    
    def generate_summary(self, input_text, max_length=150, num_beams=4):
        """
        Generate summary for given input text
        
        Args:
            input_text: Text to summarize (with prefix like "summarize pros:")
            max_length: Maximum length of generated summary
            num_beams: Beam search parameter (higher = better quality, slower)
        
        Returns:
            Generated summary string
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2  # Avoid repeating 2-word phrases
            )
        
        # Decode to text
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary
    
    def evaluate_on_test_set(self, test_csv, output_dir="results"):
        """
        Evaluate model on test set
        
        Args:
            test_csv: Path to test data CSV
            output_dir: Directory to save results
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n[3/5] Loading test data from: {test_csv}")
        test_df = pd.read_csv(test_csv)
        print(f"✓ Test set: {len(test_df)} examples")
        
        # Lists to store predictions and references
        predictions = []
        references = []
        results_data = []
        
        print(f"\n[4/5] Generating summaries on test set...")
        print("This may take a few minutes...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            input_text = row['input_text']
            target_summary = row['target_summary']
            
            # Generate prediction
            predicted_summary = self.generate_summary(input_text)
            
            # Store for metrics calculation
            predictions.append(predicted_summary)
            references.append(target_summary)
            
            # Store detailed results
            results_data.append({
                'product_id': row['product_id'],
                'type': row['type'],
                'input_text': input_text[:100] + "...",  # Truncate for readability
                'target_summary': target_summary,
                'predicted_summary': predicted_summary,
                'num_reviews': row['num_reviews']
            })
        
        print("✓ All summaries generated!")
        
        # Calculate metrics
        print(f"\n[5/5] Calculating evaluation metrics...")
        
        # ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # BLEU score
        # BLEU expects references as list of lists
        references_bleu = [[ref] for ref in references]
        bleu_score = self.bleu.compute(
            predictions=predictions,
            references=references_bleu
        )
        
        # Compile results
        metrics = {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bleu': bleu_score['bleu'],
            'num_test_examples': len(test_df)
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nTest Examples: {len(test_df)}")
        print(f"\nROUGE-1: {metrics['rouge1']:.4f}")
        print(f"ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"BLEU:    {metrics['bleu']:.4f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✓ Metrics saved to: {metrics_path}")
        
        # Save detailed results as CSV
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(output_dir, "test_predictions.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✓ Detailed results saved to: {results_path}")
        
        # Save sample predictions for inspection
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS (First 3)")
        print("="*60)
        for i in range(min(3, len(results_data))):
            sample = results_data[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Product ID: {sample['product_id']}")
            print(f"Type: {sample['type']}")
            print(f"Target:    {sample['target_summary']}")
            print(f"Predicted: {sample['predicted_summary']}")
        
        print("\n" + "="*60)
        print("✓ STEP 4 COMPLETE!")
        print("="*60)
        
        return metrics, results_df


def main():
    """Run evaluation"""
    
    # Paths (relative to ABSTRACT-SUMMARIZATION folder)
    trained_model_path = "../../models/t5_summarizer/final_model"
    test_csv = "../../data/split_data/test_data.csv"
    output_dir = "../../results"
    
    # Check if model exists
    if not os.path.exists(trained_model_path):
        print(f"❌ Error: Trained model not found at {trained_model_path}")
        print("➡️  Please run train.py first!")
        return
    
    # Check if test data exists
    if not os.path.exists(test_csv):
        print(f"❌ Error: Test data not found at {test_csv}")
        print("➡️  Please run split-data.py first!")
        return
    
    # Evaluate trained model
    evaluator = ModelEvaluator(trained_model_path)
    metrics, results_df = evaluator.evaluate_on_test_set(test_csv, output_dir)
    
    print("\n🎉 Evaluation complete!")
    print(f"📊 Results saved in: {output_dir}")
    print(f"\n📄 Files created:")
    print(f"   - {output_dir}/evaluation_metrics.json")
    print(f"   - {output_dir}/test_predictions.csv")


if __name__ == "__main__":
    main()
