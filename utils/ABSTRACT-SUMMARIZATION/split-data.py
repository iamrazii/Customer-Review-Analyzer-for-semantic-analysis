import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_csv, output_dir="data/split_data", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split summary training data into train/val/ta sets
    
    Args:
        input_csv: Path to summary_training_data.csv
        output_dir: Directory to save split files (default: data/split_data)
        train_ratio: Proportion for training (default 0.8 = 80%)
        val_ratio: Proportion for validation (default 0.1 = 10%)
        test_ratio: Proportion for test (default 0.1 = 10%)
        random_state: For reproducibility
    
    Returns:
        Paths to train, val, test CSVs
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    print("="*60)
    print("STEP 2: SPLITTING DATA INTO TRAIN/VAL/TEST")
    print("="*60)
    
    # Load data
    print(f"\n[1/4] Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} examples")
    
    # Check data
    print(f"  - Pros examples: {len(df[df['type']=='pros'])}")
    print(f"  - Cons examples: {len(df[df['type']=='cons'])}")
    
    # First split: separate out test set (10%)
    print(f"\n[2/4] Creating test set ({test_ratio*100}%)...")
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=random_state,
        stratify=df['type']  # Ensures equal ratio of pros/cons in each split
    )
    print(f"✓ Test set: {len(test_df)} examples")
    
    # Second split: split remaining into train and val
    # Calculate val_ratio relative to remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    print(f"\n[3/4] Creating train and validation sets...")
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_df['type']
    )
    print(f"✓ Train set: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Validation set: {len(val_df)} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"✓ Test set: {len(test_df)} examples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits
    print(f"\n[4/4] Saving split datasets...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_data.csv")
    val_path = os.path.join(output_dir, "val_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Train data saved to: {train_path}")
    print(f"✓ Val data saved to: {val_path}")
    print(f"✓ Test data saved to: {test_path}")
    
    print("\n" + "="*60)
    print("✓ STEP 2 COMPLETE!")
    print("="*60)
    
    return train_path, val_path, test_path


def verify_splits(train_path, val_path, test_path):
    """Verify the splits are correct"""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print("\n" + "="*60)
    print("DATA SPLIT VERIFICATION")
    print("="*60)
    
    print(f"\nTrain Set: {len(train_df)} examples")
    print(f"  - Pros: {len(train_df[train_df['type']=='pros'])}")
    print(f"  - Cons: {len(train_df[train_df['type']=='cons'])}")
    
    print(f"\nValidation Set: {len(val_df)} examples")
    print(f"  - Pros: {len(val_df[val_df['type']=='pros'])}")
    print(f"  - Cons: {len(val_df[val_df['type']=='cons'])}")
    
    print(f"\nTest Set: {len(test_df)} examples")
    print(f"  - Pros: {len(test_df[test_df['type']=='pros'])}")
    print(f"  - Cons: {len(test_df[test_df['type']=='cons'])}")
    
    # Check for overlap (should be none!)
    train_products = set(train_df['product_id'])
    val_products = set(val_df['product_id'])
    test_products = set(test_df['product_id'])
    
    overlap_train_val = train_products & val_products
    overlap_train_test = train_products & test_products
    overlap_val_test = val_products & test_products
    
    print(f"\nOverlap Check:")
    print(f"  - Train ∩ Val: {len(overlap_train_val)} products (should be 0)")
    print(f"  - Train ∩ Test: {len(overlap_train_test)} products (should be 0)")
    print(f"  - Val ∩ Test: {len(overlap_val_test)} products (should be 0)")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("\n✅ No overlap detected - splits are valid!")
    else:
        print("\n⚠️ Warning: Some overlap detected!")


def main():
    """Main function to run the data splitting process"""
    # Input file (created by driver.py)
    input_csv = "../data/summary_training_data.csv"
    
    # Output directory
    output_dir = "../data/split_data"
    
    # Split the data
    train_path, val_path, test_path = split_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Verify the splits
    verify_splits(train_path, val_path, test_path)


if __name__ == "__main__":
    main()