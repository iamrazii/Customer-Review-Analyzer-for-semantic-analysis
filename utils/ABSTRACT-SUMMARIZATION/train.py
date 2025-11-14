import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

# ============================================================
# PART 1: Custom Dataset Class
# ============================================================

class SummaryDataset(Dataset):
    """
    Custom PyTorch Dataset for T5 summarization
    Converts CSV data into format T5 can understand
    """
    def __init__(self, csv_file, tokenizer, max_input_length=512, max_target_length=150):
        """
        Args:
            csv_file: Path to train/val/test CSV
            tokenizer: T5 tokenizer
            max_input_length: Max length of input text
            max_target_length: Max length of summary
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get one training example"""
        row = self.df.iloc[idx]
        
        # Input text (already has "summarize pros:" or "summarize cons:" prefix)
        input_text = row['input_text']
        
        # Target summary (what the model should learn to generate)
        target_text = row['target_summary']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension (squeeze)
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }


# ============================================================
# PART 2: Training Function
# ============================================================

def train_t5_model(
    train_csv,
    val_csv,
    output_dir="models/t5_summarizer",
    model_name="t5-small",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5
):
    """
    Fine-tune T5 model on summarization task
    
    Args:
        train_csv: Path to training data
        val_csv: Path to validation data
        output_dir: Where to save trained model
        model_name: T5 variant (t5-small, t5-base, etc.)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    
    print("="*60)
    print("STEP 3: FINE-TUNING T5 MODEL")
    print("="*60)
    
    # 1. Load tokenizer and model
    print(f"\n[1/6] Loading {model_name} model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Model loaded on: {device}")
    
    # 2. Create datasets
    print(f"\n[2/6] Creating datasets...")
    train_dataset = SummaryDataset(train_csv, tokenizer)
    val_dataset = SummaryDataset(val_csv, tokenizer)
    
    print(f"✓ Train dataset: {len(train_dataset)} examples")
    print(f"✓ Validation dataset: {len(val_dataset)} examples")
    
    # 3. Set up training arguments
    print(f"\n[3/6] Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        
        # Evaluation settings
        eval_strategy="epoch",  # Changed back from evaluation_strategy
        save_strategy="epoch",  # Save model after each epoch
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        
        # Performance
        warmup_steps=100,
        weight_decay=0.01,
        
        # Save best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Other
        save_total_limit=2,  # Only keep 2 best models
        report_to="none"  # Don't use wandb/tensorboard
    )
    
    print(f"✓ Training for {num_epochs} epochs")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Learning rate: {learning_rate}")
    
    # 4. Create Trainer
    print(f"\n[4/6] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # 5. Train!
    print(f"\n[5/6] Starting training...")
    print("="*60)
    print("This may take 10-30 minutes depending on your hardware...")
    print("="*60)
    
    trainer.train()
    
    print("\n✓ Training complete!")
    
    # 6. Save final model
    print(f"\n[6/6] Saving model...")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    print(f"✓ Model saved to: {output_dir}/final_model")
    
    print("\n" + "="*60)
    print("✓ STEP 3 COMPLETE!")
    print("="*60)
    
    return trainer


# ============================================================
# PART 3: Main Function
# ============================================================

def main():
    """Run the training process"""
    
    # Paths to your split data (relative to where script is run from)
    train_csv = "../../data/split_data/train_data.csv"
    val_csv = "../../data/split_data/val_data.csv"
    
    # Output directory for saved model
    output_dir = "../../models/t5_summarizer"
    
    # Check if CSV files exist
    if not os.path.exists(train_csv):
        print(f"❌ Error: Training data not found at {train_csv}")
        print("➡️  Run driver.py and split-data.py first!")
        return
    
    if not os.path.exists(val_csv):
        print(f"❌ Error: Validation data not found at {val_csv}")
        print("➡️  Run split-data.py first!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the model
    trainer = train_t5_model(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=output_dir,
        model_name="t5-small",  # Change to "t5-base" for better quality
        num_epochs=3,  # Increase to 5-10 for better results
        batch_size=4,  # Reduce to 2 if you get memory errors
        learning_rate=5e-5
    )
    
    print("\n🎉 Training finished successfully!")
    print(f"📁 Trained model saved in: {output_dir}/final_model")
    print("\n➡️  Next: Run evaluation on test set (Step 4)")


if __name__ == "__main__":
    main()