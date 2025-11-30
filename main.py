import pandas as pd
import sys
import warnings
import os
from utils.LoadData import loadfile
from utils.AspectDataset import generate_aspect_dataset
from modules.ABSA import aspect_sentiment_analysis
from modules.ML_Sentiment import train_ml_model
from modules.DL_Intent import evaluate_intent_model, refresh_predictions
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_and_prepare_data():
    """
    Handles loading the main data.csv file and performing initial prep.
    """
    print("\n--- Step 1: Loading and Preparing Data ---")
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


def run_model_1_absa(df):
    print("\n--- Step 2: Running Transformer ABSA (Model 1) ---")
    if df is None:
        print("Cannot run Model 1: Data frame is empty.")
        return None, None
        
    df = aspect_sentiment_analysis(df)
    
    print("Generating aspect dataset from Model 1 results...")
    aspect_df = generate_aspect_dataset(df)
    print(f"Generated {len(aspect_df)} aspect-sentiment entries.")
    
    return df, aspect_df

    # creating aspect dataset 
    # aspect_df = generate_aspect_dataset(aspect_df)

    aspect_df = pd.read_csv("data/aspect_data.csv")
    
    print("-----------------ABSA Machine Learning--------------------")
    model,vectorizer = train_pipeline(aspect_df) # ABSA ML MODEL 
    
    test_sentence  = "Although processing speed is good, battery is bad."
    print(f"Showcasing example of ABSA ML on {test_sentence}")
    preds = predict_sentiment(model, vectorizer, test_sentence)

    print("Predictions:")
    print(preds)
    print("\n\n-----------------ABSA TRANSFORMER--------------------")
    train_simple_absa(aspect_df, model_name="distilbert-base-uncased", output_dir="models/absa_pytorch")  # ABSA TRANFORMER

    transres= transformer_predict_sentiment(test_sentence)
    print(transres)
    

def run_model_2_ml_training(aspect_df):
    print("\n--- Step 3: Training ML Sentiment Model (Model 2) ---")
    if aspect_df is None or aspect_df.empty:
        print("Skipping ML Model training: No aspect data was provided (run Model 1 first).")
        return
        
    train_ml_model()


def run_model_3_dl_intent(df):
    """
    Runs the Deep Learning Intent evaluation (Model 3).
    """
    print("\n--- Step 4: DL Intent Checker (Model 3) ---")
    
    label_file = "data/intent_labeling_task.csv"

    if not os.path.exists(label_file):
        print(f"Error: {label_file} not found.")
        print("Please ensure you have generated and labeled the intent data file.")
    else:
       
        refresh_predictions()
        evaluate_intent_model()


if __name__ == "__main__":
    print("--- NLP Project Start ---")
    
    main_df = load_and_prepare_data()
    
    if main_df is not None:
        
        # --- Model 1: Transformer ABSA ---
        main_df, aspect_df = run_model_1_absa(main_df)
        
        # --- Model 2: ML Sentiment Training ---
        run_model_2_ml_training(aspect_df)
        
        # --- Model 3: DL Intent Checker ---
        #run_model_3_dl_intent(main_df)
        
    else:
        print("Failed to load data. Exiting.")

    print("\n--- Project End ---")