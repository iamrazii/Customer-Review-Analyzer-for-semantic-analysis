from utils.LoadData import loadfile
from modules.ABSA import aspect_sentiment_analysis
def main():
    print("Loading dataset...")
    df = loadfile("data/data.csv")
    print(len(df))
    df = aspect_sentiment_analysis(df)
    aspect_df = df[df["aspect_sentiments"].apply(lambda x: len(x) > 0)] 

    print(len(aspect_df))


if __name__== "__main__":
    main()