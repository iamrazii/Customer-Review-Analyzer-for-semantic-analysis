from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class ReviewSummarizer:
    def __init__(self, model_name="t5-small"):
        """
        Initialize T5 model for summarization
        model_name: can be 't5-small', 't5-base', 't5-large'
        """
        print(f"Loading {model_name} model...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def generate_summary(self, text, max_length=150, min_length=30):
        """
        Generate summary for given text using T5
        """
        # T5 expects input in format: "summarize: <text>"
        input_text = "summarize: " + text
        
        # Tokenize input
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and return
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def summarize_product_reviews(df, product_id, summarizer):
    """
    Generate Pros and Cons summaries for a specific product
    
    Args:
        df: DataFrame with columns ['product_id', 'Summary', 'Sentiment']
        product_id: The product to summarize
        summarizer: ReviewSummarizer instance
    
    Returns:
        dict with 'pros_summary' and 'cons_summary'
    """
    # Filter reviews for this product
    product_reviews = df[df['product_id'] == product_id]
    
    if len(product_reviews) == 0:
        return {
            'product_id': product_id,
            'pros_summary': "No reviews available",
            'cons_summary': "No reviews available"
        }
    
    # Separate positive and negative reviews
    positive_reviews = product_reviews[product_reviews['Sentiment'] == 'positive']['Summary'].tolist()
    negative_reviews = product_reviews[product_reviews['Sentiment'] == 'negative']['Summary'].tolist()
    
    # Combine reviews into text chunks
    pros_text = " ".join(positive_reviews) if positive_reviews else "No positive feedback"
    cons_text = " ".join(negative_reviews) if negative_reviews else "No negative feedback"
    
    # Generate summaries
    pros_summary = summarizer.generate_summary(pros_text) if positive_reviews else "No positive feedback"
    cons_summary = summarizer.generate_summary(cons_text) if negative_reviews else "No negative feedback"
    
    return {
        'product_id': product_id,
        'pros_summary': pros_summary,
        'cons_summary': cons_summary,
        'num_positive_reviews': len(positive_reviews),
        'num_negative_reviews': len(negative_reviews)
    }