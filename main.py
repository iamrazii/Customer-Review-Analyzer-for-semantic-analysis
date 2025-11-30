import pandas as pd
import sys
import warnings
import os

from utils.LoadData import loadfile
from utils.AspectDataset import generate_aspect_dataset

from modules.ABSA import aspect_sentiment_analysis
from modules.ML_Sentiment import train_ml_model
from modules.DL_Intent import evaluate_intent_model, refresh_predictions

try:
    from modules.ABSAMLPrediction import train_pipeline, predict_sentiment
    from modules.ABSATransformer import train_simple_absa, predict_sentiment as transformer_predict_sentiment
except ImportError as e:
    print(f"Warning: Could not import external modules ({e}). Skipping those steps.")

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_and_prepare_data():
    print("\n" + "="*50)
    print("--- STEP 1: Loading Dataset ---")
    print("="*50)
    try:
        df = loadfile("data/data.csv")
    except FileNotFoundError:
        print("Error: 'data/data.csv' not found. Please place it in the 'data' folder.")
        return None
        
    print(f"Loaded {len(df)} reviews.")
    
    if 'Summary' not in df.columns:
        print("Error: 'Summary' column not found in data.csv.")
        return None
        
    df.dropna(subset=["Summary"], inplace=True)
    df["Summary"] = df["Summary"].astype(str)
    return df

def generate_absa_data(df):
    print("\n" + "="*50)
    print("--- STEP 2: Transformer ABSA (Data Generation) ---")
    print("="*50)
    
    if df is None:
        print("Data is empty. Skipping.")
        return None

    df = aspect_sentiment_analysis(df)
    
    print("Generating aspect dataset (aspect_data.csv)...")
    aspect_df = generate_aspect_dataset(df)
    print(f"Generated {len(aspect_df)} aspect-sentiment entries.")
    
    return aspect_df

def run_gradient_boosting_training(aspect_df):
    print("\n" + "="*50)
    print("--- STEP 3: Gradient Boosting Sentiment Training ---")
    print("="*50)
    
    if aspect_df is None or aspect_df.empty:
        print("Skipping: No aspect data provided.")
        return

    train_ml_model()

def run_absa_ml_prediction(aspect_df):
    print("\n" + "="*50)
    print("--- STEP 4: ABSA ML Pipeline Prediction ---")
    print("="*50)

    if aspect_df is None or aspect_df.empty:
        try:
            aspect_df = pd.read_csv("data/aspect_data.csv")
        except:
            print("Skipping: No aspect data found.")
            return

    model, vectorizer = train_pipeline(aspect_df)
    
    test_sentence = "Although processing speed is good, battery is bad."
    print(f"\n[Demo] Testing on: '{test_sentence}'")
    preds = predict_sentiment(model, vectorizer, test_sentence)
    print(f"Prediction: {preds}")

def run_absa_transformer_training(aspect_df):
    print("\n" + "="*50)
    print("--- STEP 5: ABSA Transformer Fine-Tuning ---")
    print("="*50)

    if aspect_df is None or aspect_df.empty:
        try:
            aspect_df = pd.read_csv("data/aspect_data.csv")
        except:
            print("Skipping: No aspect data found.")
            return

    train_simple_absa(aspect_df, model_name="distilbert-base-uncased", output_dir="models/absa_pytorch")
    
    test_sentence = "Although processing speed is good, battery is bad."
    print(f"\n[Demo] Testing on: '{test_sentence}'")
    transres = transformer_predict_sentiment(test_sentence)
    print(f"Prediction: {transres}")

def run_dl_intent_analysis(df):
    print("\n" + "="*50)
    print("--- STEP 6: DL Intent Analysis ---")
    print("="*50)

    label_file = "data/intent_labeling_task.csv"

    if not os.path.exists(label_file):
        print(f"Error: {label_file} not found.")
        print("Please ensure you have generated and labeled the intent data file.")
    else:
        refresh_predictions()
        evaluate_intent_model()

def main():
    print("--- COMBINED NLP PROJECT RUNNER ---")
    
    main_df = load_and_prepare_data()
    
    if main_df is not None:
        aspect_df = generate_absa_data(main_df)
        run_gradient_boosting_training(aspect_df)
        run_absa_ml_prediction(aspect_df)
        run_absa_transformer_training(aspect_df)
        run_dl_intent_analysis(main_df)
    else:
        print("Critical Error: Failed to load data. Exiting.")


if __name__ == "__main__":
    main()