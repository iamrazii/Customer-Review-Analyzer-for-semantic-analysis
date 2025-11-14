
import pandas as pd
from summa import summarizer
import warnings
warnings.filterwarnings('ignore')

def create_summary_training_data(df, text_col="Summary", min_reviews=3):
    """
    Create training data for summarization task
    ...
    """
    
    print("Creating training data for summarization...")
    
    training_data = []
    
    # Get unique products
    product_ids = df['product_id'].unique()
    
    for prod_id in product_ids:
        # Get all reviews for this product
        product_df = df[df['product_id'] == prod_id]
        
        # Separate positive and negative reviews
        positive_reviews = product_df[product_df['Sentiment'] == 'positive'][text_col].tolist()
        negative_reviews = product_df[product_df['Sentiment'] == 'negative'][text_col].tolist()
        
        # Process POSITIVE reviews (PROS)
        if len(positive_reviews) >= min_reviews:
            combined_positive = " ".join(positive_reviews)
            
            # Add the T5 prefix
            prefix = "summarize pros: "  # <-- NEW
            input_text = f"{prefix}{combined_positive}" # <-- NEW
            
            try:
                target_summary = summarizer.summarize(
                    combined_positive, 
                    ratio=0.3, 
                    split=False
                )
                
                if not target_summary or len(target_summary.strip()) < 10:
                    target_summary = " ".join(positive_reviews[:2])
                
                training_data.append({
                    'product_id': prod_id,
                    'type': 'pros',
                    'input_text': input_text, # <-- MODIFIED
                    'target_summary': target_summary,
                    'num_reviews': len(positive_reviews)
                })
            except:
                target_summary = " ".join(positive_reviews[:2])
                training_data.append({
                    'product_id': prod_id,
                    'type': 'pros',
                    'input_text': input_text, # <-- MODIFIED
                    'target_summary': target_summary,
                    'num_reviews': len(positive_reviews)
                })
        
        # Process NEGATIVE reviews (CONS)
        if len(negative_reviews) >= min_reviews:
            combined_negative = " ".join(negative_reviews)
            
            # Add the T5 prefix
            prefix = "summarize cons: " # <-- NEW
            input_text = f"{prefix}{combined_negative}" # <-- NEW
            
            try:
                target_summary = summarizer.summarize(
                    combined_negative,
                    ratio=0.3,
                    split=False
                )
                
                if not target_summary or len(target_summary.strip()) < 10:
                    target_summary = " ".join(negative_reviews[:2])
                
                training_data.append({
                    'product_id': prod_id,
                    'type': 'cons',
                    'input_text': input_text, # <-- MODIFIED
                    'target_summary': target_summary,
                    'num_reviews': len(negative_reviews)
                })
            except:
                target_summary = " ".join(negative_reviews[:2])
                training_data.append({
                    'product_id': prod_id,
                    'type': 'cons',
                    'input_text': input_text, # <-- MODIFIED
                    'target_summary': target_summary,
                    'num_reviews': len(negative_reviews)
                })
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)
    
    print(f"✓ Created {len(training_df)} training examples")
    print(f"  - Pros examples: {len(training_df[training_df['type']=='pros'])}")
    print(f"  - Cons examples: {len(training_df[training_df['type']=='cons'])}")
    
    return training_df


def save_training_data(training_df, output_path="data/summary_training_data.csv"):
    """Save training data to CSV"""
    training_df.to_csv(output_path, index=False)
    print(f"✓ Training data saved to: {output_path}")
    return output_path