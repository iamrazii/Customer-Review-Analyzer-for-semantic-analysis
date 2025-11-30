import pandas as pd
import numpy as np
import re

def clean_text(text):
    text = str(text)
    cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)  # Keep alphanumerics/punctuation
    return re.sub(r"\s+", " ", cleaned).strip()


def map_sentiment_to_numeric(sentiment):
    return {"positive": 1.0, "negative": 0.0, "neutral": 0.5}.get(
        str(sentiment).lower(), 0.5
    )

# Count syllables in a word
def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count, prev = 0, False

    for ch in word:
        is_v = ch in vowels
        if is_v and not prev:
            count += 1
        prev = is_v

    if word.endswith("e"):
        count -= 1

    return max(1, count)


# Calculate Flesch Reading Ease score
def calculate_readability(text, words):
    if len(words) < 3:
        return 50.0

    sentences = max(text.count("."), 1)
    syllables = sum(count_syllables(w) for w in words)

    score = (
        206.835
        - 1.015 * (len(words) / sentences)
        - 84.6 * (syllables / len(words))
    )
    return max(0, min(100, score))

#constants with some jargon
PRODUCT_TERMS = [
    "battery", "screen", "camera", "processor", "memory", "price",
    "quality", "performance", "design", "durability", "sound",
    "display", "storage", "speed", "charging"
]


TECH_TERMS = [
    "mah", "ghz", "gb", "ram", "rom", "megapixel", "mp", "inch",
    "watt", "voltage", "ampere", "hz", "mhz"
]

TIME_TERMS = ["day", "week", "month", "year", "hour", "minute"]

INFO_WORDS = [
    "because", "however", "although", "specifically", "compared",
    "versus", "better", "worse", "recommend", "worth",
    "bought", "used", "tested", "tried"
]

PROS = ["pros", "advantages", "benefits", "good"]
CONS = ["cons", "disadvantages", "drawbacks", "issues", "problems"]
COMPARISONS = ["better than", "worse than", "compared to", "similar to", "versus", "vs"]


EXPERIENCE = [
    "i bought", "i used", "i tried", "i tested", "my experience",
    "i received", "i ordered", "after using", "been using", 
    "purchase", "arrived", "package"
]

WARNING_TERMS = [
    "waste", "worst", "poor", "terrible", "broke", "useless", 
    "return", "disappointed", "garbage", "horrible", "avoid", 
    "stopped working", "don't buy", "do not buy"
]

HELP_KEYWORDS = [
    "quality", "performance", "excellent", "recommend", "worth",
    "battery", "fast", "durable", "bought", "used", "price",
    "value", "features", "specifications", "compared", "better"
]

GENERIC = ["good product", "nice product", "bad product", "okay"]
TECH_INFO = ["mah", "ghz", "gb", "ram", "inch", "watt"]

def extract_enhanced_features(text, sentiment, rating):
    text = str(text)
    text_lower = text.lower()
    words = text.split()

    nums = len(re.findall(r"\b\d+\b", text))  # Numeric mentions

    # Build feature dictionary
    features = {
        "review_length": len(words),
        "char_count": len(text),
        "num_count": nums,
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "sentence_count": max(text.count("."), 1),
        "sentiment": map_sentiment_to_numeric(sentiment),
        "rating": float(rating) if rating is not None else 3.0,
        "readability_score": calculate_readability(text, words),
        "specificity_score": nums * 5 + sum(f in text_lower for f in PRODUCT_TERMS) * 3,
        "tech_term_count": sum(t in text_lower for t in TECH_TERMS),
        "time_reference_count": sum(t in text_lower for t in TIME_TERMS), # time refs
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "caps_ratio": sum(c.isupper() for c in text) / len(text) if len(text) else 0,
        "unique_word_ratio": len(set(words)) / len(words) if words else 0,
        "has_structure": 1 if (text.count(".") > 2 or text.count(",") > 2) else 0,
        "informative_word_count": sum(w in text_lower for w in INFO_WORDS),
        "has_pros": int(any(w in text_lower for w in PROS)),
        "has_cons": int(any(w in text_lower for w in CONS)),
        "has_comparison": int(any(c in text_lower for c in COMPARISONS)),
        "has_personal_experience": int(any(e in text_lower for e in EXPERIENCE)),
        "warning_signal_count": sum(w in text_lower for w in WARNING_TERMS),
        
        "info_density": (nums + sum(f in text_lower for f in PRODUCT_TERMS)) 
        / len(words) 
        if words
        else 0,
    }

    return features

def extract_enhanced_features_from_dataframe(df):
    print("Extracting features from reviews...")
    features = [extract_enhanced_features(r.Summary, r.Sentiment, r.Rate) for r in df.itertuples()]
    features_df = pd.DataFrame(features).apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"✓ Extracted {features_df.shape[1]} features from {len(df)} reviews")
    return features_df

def calculate_helpfulness(text):
    text = str(text).lower()
    words = text.split()
    wc = len(words)

    score = 0  # Composite heuristic score

    # 1. Length (max 25)
    score += [5, 10, 15, 20, 25][
        0 if wc < 5 else 1 if wc < 10 else 2 if wc < 20 else 3 if wc < 40 else 4
    ]

    # 2. Specificity (max 15)
    score += min(len(re.findall(r"\b\d+\b", text)) * 5, 15)

    # 3. Informative keywords (max 20)
    score += min(sum(k in text for k in HELP_KEYWORDS) * 2, 20)

    # 4. Structure (max 15)
    score += min(text.count(".") * 3, 15)

    # 5. Personal experience (max 10) - Now uses updated EXPERIENCE list
    if any(exp in text for exp in EXPERIENCE):
        score += 10

    # 6. Technical (max 5)
    if any(t in text for t in TECH_INFO):
        score += 5

    # 7. Bonus content (max 10)
    if wc > 15 and not any(g in text for g in GENERIC):
        score += 10
        

    # If the review contains strong warning words, it is very helpful
    if any(w in text for w in WARNING_TERMS):
        score += 15

    return float(min(score, 100))

# Define feature order for consistent extraction
FEATURE_ORDER = [
    "review_length", "char_count", "num_count", "avg_word_len",
    "sentence_count", "sentiment", "rating", "readability_score",
    "specificity_score", "tech_term_count", "time_reference_count",
    "question_count", "exclamation_count", "caps_ratio",
    "unique_word_ratio", "has_structure", "informative_word_count",
    "has_pros", "has_cons", "has_comparison",
    "has_personal_experience", "info_density",
    "warning_signal_count" 
]

def prepare_features_for_prediction(text, sentiment, rating, tfidf, scaler):
    clean = clean_text(text)

    num_dict = extract_enhanced_features(clean, sentiment, rating)  # Deterministic order
    num_arr = np.array([[num_dict[f] for f in FEATURE_ORDER]], dtype=np.float64)

    # text features
    tfidf_arr = tfidf.transform([clean]).toarray()

    X = np.hstack([num_arr, tfidf_arr])
    num_feat_count = len(FEATURE_ORDER)
    X[:, :num_feat_count] = scaler.transform(X[:, :num_feat_count])  # Scale numeric slice

    return X