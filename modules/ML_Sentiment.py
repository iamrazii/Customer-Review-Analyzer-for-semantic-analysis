import pandas as pd
import numpy as np
import joblib
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns  # <-- Added seaborn for heatmap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    top_k_accuracy_score, 
    log_loss
)
from transformers import DistilBertTokenizer, DistilBertModel

# Define paths
MODEL_PATH = "models/ml_sentiment_model.pkl"
DATA_PATH = "data/aspect_data.csv"
PLOT_LOSS_PATH = "models/training_loss_curve.png"
PLOT_ACC_PATH = "models/training_accuracy_curve.png"
PLOT_CM_PATH = "models/ml_confusion_matrix.png"

class BertVectorizer(BaseEstimator, TransformerMixin):
    """
    Uses the Hugging Face 'DistilBERT' model to generate 
    context-aware text embeddings.
    """
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def fit(self, X, y=None):
        print("Loading DistilBERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        return self

    def transform(self, X):
        print(f"Generating BERT embeddings for {len(X)} phrases...")
        embeddings = []
        for text in X:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            sentence_vector = last_hidden_states.mean(dim=1).squeeze().numpy()
            embeddings.append(sentence_vector)
        return np.array(embeddings)

def plot_learning_curves(clf, X_train, y_train, X_val, y_val):
    """
    Calculates and plots Loss and Accuracy over boosting iterations.
    Handles Early Stopping dynamically.
    """
    print("\nGenerating Training vs Validation Learning Curves...")
    
    # 1. Get the actual number of iterations trained
    real_iterations = len(clf.train_score_)
    
    # 2. Initialize arrays with the REAL size
    train_score = np.zeros((real_iterations,), dtype=np.float64)
    val_score = np.zeros((real_iterations,), dtype=np.float64)
    val_loss = np.zeros((real_iterations,), dtype=np.float64)
    
    # 3. Calculate metrics for each iteration
    for i, y_pred in enumerate(clf.staged_predict(X_train)):
        if i >= real_iterations: break 
        train_score[i] = accuracy_score(y_train, y_pred)
        
    for i, y_pred_proba in enumerate(clf.staged_predict_proba(X_val)):
        if i >= real_iterations: break
        # Log Loss (Cross Entropy)
        val_loss[i] = log_loss(y_val, y_pred_proba)
        # Accuracy
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred_labels = clf.classes_[y_pred]
        val_score[i] = accuracy_score(y_val, y_pred_labels)

    train_loss = clf.train_score_

    iterations_range = np.arange(real_iterations) + 1

    # --- Plot Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(iterations_range, train_loss, 'b-', label='Training Loss')
    plt.plot(iterations_range, val_loss, 'r-', label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Log Loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(PLOT_LOSS_PATH)
    print(f"Loss curve saved to {PLOT_LOSS_PATH}")
    plt.close()

    # --- Plot Accuracy Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(iterations_range, train_score, 'b-', label='Training Accuracy')
    plt.plot(iterations_range, val_score, 'r-', label='Validation Accuracy')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(PLOT_ACC_PATH)
    print(f"Accuracy curve saved to {PLOT_ACC_PATH}")
    plt.close()

def train_ml_model():
    print("\n--- Training ML Sentiment Model (BERT + Gradient Boosting) ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
    
    try:
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=['phrase', 'sentiment'], inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_label'] = df['sentiment'].map(label_map)
    df.dropna(subset=['sentiment_label'], inplace=True)
    
    # 2. Vectorize Data FIRST
    bert = BertVectorizer()
    bert.fit(df['phrase']) 
    X_vectors = bert.transform(df['phrase'])
    y = df['sentiment_label'].values

    # 3. Split: Train (60%), Validation (20%), Test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(f"Data Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Train Gradient Boosting Classifier
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        subsample=0.8,
        validation_fraction=0.1, 
        n_iter_no_change=10, # This triggers Early Stopping
        random_state=42
    )

    print("Fitting Gradient Boosting Model...")
    clf.fit(X_train, y_train)

    # 5. Plot Training vs Validation Graphs
    plot_learning_curves(clf, X_train, y_train, X_val, y_val)

    # 6. Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    em = acc 
    
    target_names = ['negative', 'neutral', 'positive']
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        auc = 0.0
        print("Could not calculate AUC.")

    top2_acc = top_k_accuracy_score(y_test, y_prob, k=2)

    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Exact Match (EM):{em * 100:.2f}%")
    print(f"Top-2 Accuracy:  {top2_acc * 100:.2f}%")
    print(f"ROC AUC Score:   {auc:.4f}")
    print(f"Precision (W):   {report['weighted avg']['precision']:.4f}")
    print(f"Recall (W):      {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score (W):    {report['weighted avg']['f1-score']:.4f}")

    print("\n--- Generating Confusion Matrix Plot ---")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('ML Sentiment Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(PLOT_CM_PATH)
    print(f"Confusion Matrix saved to {PLOT_CM_PATH}")
    plt.close()

    # 7. Save Model
    joblib.dump(clf, MODEL_PATH) 
    print(f"Model saved to {MODEL_PATH}")
    print("Graphs saved to models/ directory.")

def predict_ml_sentiment(phrases: list):
    pass