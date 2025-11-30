import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", message=".*Using a pipeline without specifying a model name.*")

# --- Configuration ---
LABELING_FILE = "data/intent_labeling_task.csv"
CONFUSION_MATRIX_PLOT = "models/dl_confusion_matrix.png"
CANDIDATE_LABELS = ['praise', 'complaint', 'suggestion', 'inquiry']

# --- Model Loading ---
try:
    print("\nLoading DL Intent Model (Model 3)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("DL Intent Model loaded.")
except Exception as e:
    print(f"Error loading zero-shot pipeline: {e}")
    classifier = None

def classify_intent(texts: list):
    """
    Classifies a list of texts using the Zero-Shot model.
    """
    if classifier is None:
        print("Classifier not loaded.")
        return []
    
    try:
        results = classifier(
            texts, 
            CANDIDATE_LABELS, 
            hypothesis_template="This review expresses a {}."
        )
        return results
    except Exception as e:
        print(f"Error during classification: {e}")
        return []

def refresh_predictions():
    """
    Forces the model to re-predict every row in the CSV.
    This overwrites the 'model_prediction' column with the AI's current output.
    """
    if not os.path.exists(LABELING_FILE):
        print(f"Error: {LABELING_FILE} not found.")
        return

    df = pd.read_csv(LABELING_FILE)
    
    if df.empty:
        print("CSV is empty.")
        return

    
    
    # Run the classification again on the text column
    results = classify_intent(df['text'].tolist())
    
    # Extract new predictions
    new_preds = []
    new_scores = []
    
    for res in results:
        new_preds.append(res['labels'][0])
        new_scores.append(res['scores'][0])
        
    df['model_prediction'] = new_preds
    df['prediction_score'] = new_scores
    
    df.to_csv(LABELING_FILE, index=False)
    print("Predictions refreshed and saved to CSV.")

def evaluate_intent_model():
    """
    Reads the labeled file, calculates metrics, and prints 20 sample comparisons.
    """
    print("\n--- Evaluating DL Intent Model Performance ---")
    
    if not os.path.exists(LABELING_FILE):
        print(f"Error: {LABELING_FILE} not found.")
        print("Please ensure your labeled data file is in the 'data' folder.")
        return

    df = pd.read_csv(LABELING_FILE)
    
    df = df[df['true_label'].notna() & (df['true_label'].astype(str).str.strip() != "")]
    
    if df.empty:
        print("No labeled data found in the CSV. Please fill in the 'true_label' column!")
        return
    
    df['true_label'] = df['true_label'].astype(str).str.strip().str.lower()
    df['model_prediction'] = df['model_prediction'].astype(str).str.strip().str.lower()
    
    y_true = df['true_label']
    y_pred = df['model_prediction']
    
    valid_labels = [l.lower() for l in CANDIDATE_LABELS]
    
    mask = y_true.isin(valid_labels)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        print("No valid labels found. Make sure you used: praise, complaint, suggestion, inquiry")
        return

    #Calculate Metrics 
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=valid_labels, zero_division=0))
    
    #  Plot Confusion Matrix 
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_labels, yticklabels=valid_labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual (True)')
    plt.title('DL Intent Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT)
    print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PLOT}")
    plt.close()

    print("\n" + "="*70)
    print(f"                SAMPLE RESULTS (20 Reviews)")
    print("="*70)
    print(f"{'PREDICTION':<12} | {'STATUS':<20} | {'REVIEW SNIPPET'}")
    print("-" * 70)
    
    n_samples = min(20, len(df))
    sample_df = df.sample(n_samples, random_state=42)
    
    for i, row in sample_df.iterrows():
        text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
        pred = row['model_prediction']
        actual = row['true_label']
        
        
        if pred == actual:
            status = "✅"
        else:
            status = f"❌ (Exp: {actual})"
        
        print(f"{pred:<12} | {status:<20} | {text_preview}")
    
    print("-" * 70)