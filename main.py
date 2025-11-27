from utils.LoadData import loadfile
from utils.AspectDataset import generate_aspect_dataset
from modules.ABSA import aspect_sentiment_analysis
from modules.ABSAMLPrediction import train_pipeline,predict_sentiment
from modules.ABSATransformer import train_simple_absa,predict_sentiment as transformer_predict_sentiment 

import pandas as pd
def main():
    print("Loading dataset...")
    df = loadfile("data/data.csv")

    # applying sentiment analysis and creating dataset (commented cuz already done)
    # df = aspect_sentiment_analysis(df)
    # aspect_df= df.loc[df["aspect_sentiments"].apply(lambda x: len(x) > 0), ["product_id","aspect_sentiments","index"]]

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
    

if __name__== "__main__":
    main()