import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re
import ftfy
import unicodedata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../data"))
SPLIT_DIR = os.path.join(DATA_DIR, "split-data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ENCODINGS = ("utf-8", "latin1", "cp1252", "iso-8859-1")

REVIEW_COL = "Summary"  # This column has the actual review text
SUMMARY_COL = "Review"  # This column has short sentiment labels
SENTIMENT_COL = "Sentiment"
PRODUCT_COL = "product_name"

REQUIRED_COLUMNS = (
    REVIEW_COL, SUMMARY_COL, SENTIMENT_COL, PRODUCT_COL
)

REQUIRED_DIRS = (MODELS_DIR, RESULTS_DIR, SPLIT_DIR)

def clean_text(text):
    # Normalize and clean text
    if pd.notna(text): 
        text = str(text)
        text = unicodedata.normalize("NFKC", text)         # normalize unicode width
        text = ftfy.fix_text(text)                        # fix mojibake
        text = re.sub(r"\?{2,}", " ", text)               # collapse ??? artifacts
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)                  # squeeze whitespace
        text = text.encode("ascii", "ignore").decode("ascii", "ignore")  # strip non-ascii noise
        return text.strip()
    return ""


def resolve_data_path():
    candidates = (
        os.path.join(DATA_DIR, "data.csv"),
        os.path.join(BASE_DIR, "data", "data.csv"),
    )
    return next((p for p in candidates if os.path.exists(p)), None)


def read_dataset():
    data_path = resolve_data_path()
    if not data_path:
        print("❌ data.csv not found.")
        return None

    df = None

    for coding in ENCODINGS:
        try:
            df = pd.read_csv(data_path, encoding=coding)
            break
        except Exception:
            continue

    if df is None:
        print("❌ Could not read CSV with any encoding")
        return None

    # Clean ALL text columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(clean_text)  # normalize all text columns early

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return None

    # Remove duplicate input/summary pairs
    df.drop_duplicates(subset=[REVIEW_COL, SUMMARY_COL], inplace=True)
    return df

def create_aspect_summary_data(df):
    df[SENTIMENT_COL] = df[SENTIMENT_COL].str.lower()
    df = df[df[SENTIMENT_COL].isin(["positive", "negative"])]

    prompts = {
        "positive": "summarize what people like about",
        "negative": "summarize what people dislike about",
    }

    # Expanded list of generic, useless summaries/labels to filter out
    GENERIC_BAD = [
        "very good product", "very bad product",
        "i am very happy", "i am not happy",
        "good product", "bad product",
        "super", "awesome", "fair", "delightful", "wonderful",
        "terrible", "horrible", "useless", "worthless",
        "good", "bad", "ok", "nice", "poor",
        "best", "worst", "amazing", "fabulous"
    ]

    MAX_REV = 1800
    MAX_SUM = 280

    grouped = df.groupby([PRODUCT_COL, SENTIMENT_COL])
    output = []

    for (product, sentiment), group in grouped:
        if len(group) < 2:
            continue  # need multiple reviews per sentiment bucket

        # Collect reviews (now from Summary column which has actual text)
        reviews_list = group[REVIEW_COL].dropna().tolist()
        
        # Filter out short sentiment labels that got mixed in
        reviews_list = [r for r in reviews_list if len(r) > 20]
        reviews_list = list(dict.fromkeys(reviews_list))  # dedupe
        reviews_list = reviews_list[:5]  # limit

        if len(reviews_list) < 2:
            continue

        reviews_joined = " [SEP] ".join(reviews_list)[:MAX_REV]

        sorted_reviews = sorted(reviews_list, key=len, reverse=True)
        
        # Take the longest review as the target summary (it has most detail)
        target = sorted_reviews[0][:MAX_SUM]  # longest review doubles as target

        target = target.strip()

        if len(target) < 20 or len(reviews_joined) < 30:
            continue

        prefix = prompts[sentiment]

        output.append({
            "product_name": product,
            "sentiment": sentiment,
            "input_text": f"{prefix} {product}: {reviews_joined}",
            "target_summary": target,
        })

    return pd.DataFrame(output)


def split_dataset(df):
    enough = len(df) >= 12

    try:
        if "sentiment" in df.columns and enough:
            train, temp = train_test_split(
                df, test_size=0.2, random_state=42,
                stratify=df["sentiment"]               # keep sentiment balance in train
            )
            val, test = train_test_split(
                temp, test_size=0.5, random_state=42,
                stratify=temp["sentiment"]             # balanced val/test if possible
            )
        else:
            train, temp = train_test_split(df, test_size=0.2, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)
    except Exception:
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, val, test

def save_splits(train, val, test):
    for name, data in zip(["train", "val", "test"], [train, val, test]):
        data.to_csv(os.path.join(SPLIT_DIR, f"{name}.csv"),
                    index=False, encoding="utf-8")
    print(f"\n✓ Saved {len(train)} train, {len(val)} val, {len(test)} test examples\n")



def main():

    for folder in REQUIRED_DIRS:
        os.makedirs(folder, exist_ok=True)

    df = read_dataset()
    if df is None:
        return

   

    summary_df = create_aspect_summary_data(df)
    if summary_df.empty:
        print("❌ No training examples generated.")
        return

    train, val, test = split_dataset(summary_df)
    save_splits(train, val, test)

    print("\n✓ DATA PREPARATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()