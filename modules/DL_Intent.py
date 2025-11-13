from transformers import pipeline
import warnings


warnings.filterwarnings(
    "ignore", 
    message=".*Using a pipeline without specifying a model name.*"
)

try:
    print("\nLoading DL Zero-Shot Intent Model (Model 3)...")
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli"
    )
    print("DL Intent Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading zero-shot pipeline: {e}")
    print("Intent classification will be unavailable.")
    classifier = None

def classify_intent(texts: list):
    """
    Classifies a list of review summaries into one of the candidate labels.
    """
    if classifier is None:
        print("DL Intent Classifier is not available due to loading error.")
        return []
        
    print("\n--- Classifying Review Intents (DL Zero-Shot) ---")
    

    candidate_labels = ['praise', 'complaint', 'suggestion', 'question/inquiry']

    try:
        results = classifier(texts, candidate_labels)
        
        for result in results:
            print(f"\nReview: \"{result['sequence'][:100]}...\"")
            print(f"  -> Predicted Intent: {result['labels'][0]} (Score: {result['scores'][0]:.2f})")
            
        return results
        
    except Exception as e:
        print(f"Error during intent classification: {e}")
        return []