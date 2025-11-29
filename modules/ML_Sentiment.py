import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Define paths
MODEL_PATH = "models/ml_sentiment_model.pkl"
DATA_PATH = "data/aspect_data.csv"

def train_ml_model():
    """
    Loads the aspect data, trains a Gradient Boosting model with N-Grams,
    and saves it to the 'models/' directory.
    """
    print("\n--- Training ML Sentiment Model (Gradient Boosting) ---")
    
    # 1. Check if the required data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        print("Please ensure main.py runs `generate_aspect_dataset` first.")
        return

    # 2. Load and clean data
    try:
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=['phrase', 'sentiment'], inplace=True)
    except Exception as e:
        print(f"Error loading {DATA_PATH}: {e}")
        return
        
    if df.empty:
        print("No data found in aspect_data.csv after cleaning. Aborting.")
        return
    
    # Map sentiments to numerical labels
    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_label'] = df['sentiment'].map(label_map)
    
    df.dropna(subset=['sentiment_label'], inplace=True)
    
    if df.empty:
        print("No mappable sentiment labels found. Aborting.")
        return

    X = df['phrase']
    y = df['sentiment_label']

    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Create Pipeline 
    #    - ngram_range=(1, 2): This tells it to look at single words ("good") 
    #      AND pairs of words ("not good"). This drastically improves sentiment accuracy.
    #    - GradientBoostingClassifier: A powerful ensemble model.
    ml_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), 
        ('classifier', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42))
    ])

    # 5. Train the model
    print(f"Fitting Gradient Boosting Pipeline on {len(X_train)} samples...")
    ml_pipeline.fit(X_train, y_train)

    # 6. Evaluate the model
    y_pred = ml_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    print("\n--- Detailed Classification Report ---")
    target_names = ['negative (-1)', 'neutral (0)', 'positive (1)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n--- Confusion Matrix ---")
    print("Rows: Actual / Columns: Predicted")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save the model
    joblib.dump(ml_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def predict_ml_sentiment(phrases: list):
    """
    Loads the trained model and predicts sentiment for a list of phrases.
    """
    if not os.path.exists(MODEL_PATH):
        print("ML model not found. Please train it first by running main.py.")
        return []

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(phrases)
    
    label_map_inv = {1: 'positive', 0: 'neutral', -1: 'negative'}
    text_predictions = [label_map_inv.get(p, 'unknown') for p in predictions]
    
    return text_predictions