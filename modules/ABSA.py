from transformers import pipeline
from utils.DataProcessing import AspectExtraction
from nltk import pos_tag




def map_stars_to_sentiment(label:str):
    label = label.lower()
    if "1" in label or "2" in label:
        return "negative"
    elif "3" in label:
        return "neutral"
    return "positive"




def generate_AspectOpinionPairs(pairs,text):
    
    NEGATIONS = {"no", "not", "never", "none", "hardly", "barely", "don’t", "doesn’t", "didn’t", "can’t", "cannot"}
    normalized_pairs = [] # tuple (aspect,opinion,negation)
    contextual_inputs = [] # contextual opinions of aspects
    if pairs:

        for w1, w2 in pairs:
            opinion,aspect = w1, w2
            # build contextual input for model 
            contextual_text = (
                f"Aspect: {aspect}\nOpinion: {opinion}"
            )

            normalized_pairs.append((aspect, opinion))
            contextual_inputs.append(contextual_text)
    
    return normalized_pairs,  contextual_inputs



def aspect_sentiment_analysis(df, text_col="Summary"):
    
    pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    all_aspect_data = []

    for text in df[text_col]:
        pairs = AspectExtraction(text)
        aspects = []
        normalized_pairs,contextual_inputs = generate_AspectOpinionPairs(pairs,text)

        # run transformer model on full contextual inputs
        results = pipe(contextual_inputs)

        # postprocess model output
        for (aspect, opinion), res in zip(normalized_pairs, results):
            sentiment = map_stars_to_sentiment(res["label"])

            aspects.append({
                "aspect": aspect,
                "opinion": opinion,
                "sentiment": sentiment
            })

        all_aspect_data.append(aspects)

    df["aspect_sentiments"] = all_aspect_data
    return df
