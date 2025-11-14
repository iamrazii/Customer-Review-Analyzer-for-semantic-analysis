import pandas as pd
from sklearn.model_selection import train_test_split
import os


def create_summary_data(df, min_reviews=3, max_reviews=10):
    """
    Create training examples: group reviews by product + sentiment
    Limit input reviews to avoid exceeding token limits
    """
    print("\nCreating training examples...")
    
    # Create product_id from product_name
    df['product_id'] = df['product_name'].astype('category').cat.codes
    
    training_data = []
    
    for prod_id in df['product_id'].unique():
        product_df = df[df['product_id'] == prod_id]
        
        # POSITIVE reviews (PROS)
        pos_reviews = product_df[product_df['Sentiment'] == 'positive']['Summary'].dropna().astype(str).tolist()
        if len(pos_reviews) >= min_reviews:
            # Limit to max_reviews to avoid token limit issues
            selected_reviews = pos_reviews[:max_reviews]
            combined_text = " ".join(selected_reviews)
            # Use first review as target (simpler summary)
            summary = pos_reviews[0]
            
            training_data.append({
                'product_id': prod_id,
                'type': 'pros',
                'input_text': f"summarize pros: {combined_text}",
                'target_summary': summary
            })
        
        # NEGATIVE reviews (CONS)
        neg_reviews = product_df[product_df['Sentiment'] == 'negative']['Summary'].dropna().astype(str).tolist()
        if len(neg_reviews) >= min_reviews:
            selected_reviews = neg_reviews[:max_reviews]
            combined_text = " ".join(selected_reviews)
            summary = neg_reviews[0]
            
            training_data.append({
                'product_id': prod_id,
                'type': 'cons',
                'input_text': f"summarize cons: {combined_text}",
                'target_summary': summary
            })
    
    return pd.DataFrame(training_data)


def main():
    print("="*60)
    print("DATA PREPARATION FOR SUMMARIZATION")
    print("="*60)
    
    # Load data
    print("\n[STEP 1/3] Loading data...")
    data_path = "../../data/data.csv"
    
    if not os.path.exists(data_path):
        data_path = "data/data.csv"
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} reviews")
    
    # Create training examples
    print("\n[STEP 2/3] Creating training examples...")
    summary_df = create_summary_data(df, min_reviews=3, max_reviews=10)
    print(f"✓ Created {len(summary_df)} examples")
    print(f"  - Pros: {len(summary_df[summary_df['type']=='pros'])}")
    print(f"  - Cons: {len(summary_df[summary_df['type']=='cons'])}")
    
    # Check if we have data
    if len(summary_df) == 0:
        print("❌ Error: No training examples created!")
        return
    
    # Split: 80% train, 10% val, 10% test
    print("\n[STEP 3/3] Splitting data (80/10/10)...")
    train_df, temp_df = train_test_split(summary_df, test_size=0.2, random_state=42, stratify=summary_df['type'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['type'])
    
    print(f"✓ Train: {len(train_df)} ({len(train_df)/len(summary_df)*100:.0f}%)")
    print(f"✓ Val:   {len(val_df)} ({len(val_df)/len(summary_df)*100:.0f}%)")
    print(f"✓ Test:  {len(test_df)} ({len(test_df)/len(summary_df)*100:.0f}%)")
    
    # Save files
    output_dir = "../../data/split-data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"\n✓ Saved to {output_dir}/")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLE:")
    print("="*60)
    sample = train_df.iloc[0]
    print(f"\nType: {sample['type']}")
    print(f"Input: {sample['input_text'][:200]}...")
    print(f"Target: {sample['target_summary']}")
    
    print("\n" + "="*60)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\n➡️  Next: Run train_and_eval.py")


if __name__ == "__main__":
    main()