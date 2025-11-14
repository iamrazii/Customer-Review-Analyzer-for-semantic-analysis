

import pandas as pd

def generate_aspect_dataset(df: pd.DataFrame):
    data = []
    for _, row in df.iterrows():
        review_id = row["index"]
        product_id = row["product_id"]
        entries = row["aspect_sentiments"]  # list of dicts

        if not entries or not isinstance(entries, list):
            continue

        for entry in entries:
            aspect = entry.get("aspect")
            opinion = entry.get("opinion")
            sentiment = entry.get("sentiment")
            phrase = f"{opinion} {aspect}"

            data.append({
                "review_id":review_id,
                "product_id":product_id,
                "aspect": aspect,
                "opinion": opinion,
                "phrase":phrase,
                "sentiment": sentiment
            })

    new_df = pd.DataFrame(data)
    new_df.to_csv("data/aspect_data.csv",index=False)
    return new_df