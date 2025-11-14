import pandas as pd
from LoadData import loadfile
from summaryDataset import create_summary_training_data, save_training_data

# Import your REAL sentiment analysis
import sys
sys.path.append('..')  # Go up one directory
from modules.ABSA import aspect_sentiment_analysis  # Or wherever your sentiment code is


def main_create_dataset():
    print("--- Step 1: Creating Summarization Dataset ---")
    
    # 1. Load data
    print("Loading data...")
    df = loadfile("../data/data.csv")
    print(f"✓ Loaded {len(df)} reviews")
    
    # 2. Run REAL sentiment analysis
    print("\nRunning sentiment analysis...")
    
    # Check if 'Sentiment' column already exists in data.csv
    if 'Sentiment' not in df.columns:
        # Option A: If you have a sentiment function, use it
        # df = your_sentiment_function(df)
        
        # Option B: If data.csv already has 'Sentiment' column, it's already there
        print("⚠️ Warning: 'Sentiment' column not found!")
        print("Please make sure your data.csv has a 'Sentiment' column")
        print("OR run sentiment analysis first")
        return
    
    print(f"✓ Sentiment column found")
    print(f"  Positive: {len(df[df['Sentiment']=='positive'])}")
    print(f"  Negative: {len(df[df['Sentiment']=='negative'])}")
    print(f"  Neutral: {len(df[df['Sentiment']=='neutral'])}")
    
    # 3. Create the summary training data
    training_df = create_summary_training_data(df, text_col="Summary", min_reviews=3)
    
    # 4. Save the new dataset
    if not training_df.empty:
        save_training_data(training_df, "../data/summary_training_data.csv")
        
        # Show sample
        print("\n" + "="*60)
        print("SAMPLE OUTPUT:")
        print("="*60)
        if len(training_df) > 0:
            sample = training_df.iloc[0]
            print(f"\nProduct ID: {sample['product_id']}")
            print(f"Type: {sample['type']}")
            print(f"Num Reviews: {sample['num_reviews']}")
            print(f"\nInput Text (first 100 chars):\n{sample['input_text'][:100]}...")
            print(f"\nTarget Summary:\n{sample['target_summary']}")
    else:
        print("❌ No training data created. Check 'min_reviews' or data source.")
    
    print("\n--- Step 1 Complete ---")


if __name__ == "__main__":
    main_create_dataset()