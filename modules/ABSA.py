from transformers import pipeline
from utils.DataProcessing import AspectExtraction,CleanText
from nltk import pos_tag


def classify_sentiment(texts, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    pipe = pipeline("sentiment-analysis", model=model_name)
    results = pipe(texts)
    mapped = []
    for r in results:
        l = r["label"].lower()
        if "1" in l or "2" in l or "neg" in l:
            mapped.append("negative")
        elif "3" in l or "neutral" in l:
            mapped.append("neutral")
        else:
            mapped.append("positive")
    return mapped

def map_stars_to_sentiment(label:str):
    label = label.lower()
    if "1" in label or "2" in label:
        return "negative"
    elif "3" in label:
        return "neutral"
    return "positive"

def aspect_sentiment_analysis(df, text_col="Summary"):
    pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    all_aspect_data = []

    for text in df[text_col]:
        pairs = AspectExtraction(text)
        aspects = []

        if pairs:
            phrases = []
            normalized_pairs = []

            for w1, w2 in pairs:
                # Use POS tagging to detect which is noun/adjective
                tags = pos_tag([w1, w2])
                if tags[0][1].startswith("JJ") and tags[1][1].startswith("NN"):
                    # adjective → noun
                    adj, noun = w1, w2
                elif tags[0][1].startswith("NN") and tags[1][1].startswith("JJ"):
                    # noun → adjective
                    noun, adj = w1, w2
                else:
                    # fallback — assume noun-adjective order
                    noun, adj = w1, w2

                normalized_pairs.append((noun, adj))
                phrases.append(f"{adj} {noun}")  # Adjective before noun sounds more natural for sentiment

            # Run transformer on all phrases at once
            results = pipe(phrases)

            for (noun, adj), res in zip(normalized_pairs, results):
                sentiment = map_stars_to_sentiment(res["label"])
                aspects.append({
                    "aspect": noun,
                    "opinion": adj,
                    "sentiment": sentiment
                })

        all_aspect_data.append(aspects)

    df["aspect_sentiments"] = all_aspect_data
    return df
