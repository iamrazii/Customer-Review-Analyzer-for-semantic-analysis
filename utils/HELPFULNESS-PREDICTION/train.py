import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from preprocessing import clean_text, calculate_helpfulness, extract_enhanced_features_from_dataframe

warnings.filterwarnings("ignore")


def log(msg):
    print(f"\n[+] {msg}")


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():

    log("Loading dataset...")
    df = pd.read_csv("../../data/data.csv")[["Summary", "Sentiment", "Rate"]].dropna()
    df["Summary"] = df["Summary"].apply(clean_text)

    # Helpful score
    log("Generating helpfulness scores...")
    df["helpfulness"] = df["Summary"].apply(calculate_helpfulness)

    # Fix/validate Rate
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df.dropna(subset=["Rate"])

    log("Extracting numeric features...")
    X_num_df = extract_enhanced_features_from_dataframe(df)
    X_num = X_num_df.values.astype(np.float64)
    num_features = X_num.shape[1]

    log("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )
    X_tfidf = tfidf.fit_transform(df["Summary"]).toarray()

    log("Combining features...")
    X = np.hstack([X_num, X_tfidf])
    y = df["helpfulness"].values

    log("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log("Scaling numeric features...")
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[:, :num_features])
    X_test_scaled_num = scaler.transform(X_test[:, :num_features])

    X_train = np.hstack([X_train_scaled_num, X_train[:, num_features:]])
    X_test = np.hstack([X_test_scaled_num, X_test[:, num_features:]])

    # Save test data
    os.makedirs("../../data/HELPFULNESS-DATA", exist_ok=True)
    np.save("../../data/HELPFULNESS-DATA/X_test.npy", X_test)
    pd.DataFrame(y_test, columns=["helpfulness"]).to_csv(
        "../../data/HELPFULNESS-DATA/y_test.csv", index=False
    )

    log("Training CatBoost model...")
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_state=42,
        early_stopping_rounds=30,
        logging_level="Silent",
        allow_writing_files=False
        
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

    log("Saving model artifacts...")
    os.makedirs("models", exist_ok=True)

    save_pickle(model, "models/catboost.pkl")
    save_pickle(scaler, "models/scaler.pkl")
    save_pickle(tfidf, "models/tfidf.pkl")

    with open("models/model_info.txt", "w") as f:
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Numeric Features: {num_features}\n")
        f.write(f"TF-IDF: {X_tfidf.shape[1]}\n")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
