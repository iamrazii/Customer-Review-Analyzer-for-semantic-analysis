import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import prepare_features_for_prediction

warnings.filterwarnings("ignore")

SECTION_LINE = "=" * 60


def print_section(title):
    print(f"\n{SECTION_LINE}\n{title}\n{SECTION_LINE}")


def load_model():
    print_section("Loading model artifacts")
    model = pickle.load(open("models/catboost.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
    print("✓ Model, scaler, and TF-IDF loaded")
    if os.path.exists("models/model_info.txt"):
        print("Model info:")
        with open("models/model_info.txt", "r") as f:
            for line in f:
                print(f"  {line.strip()}")
    return model, scaler, tfidf


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }, preds


def display_results(results):
    print_section("Evaluation metrics")
    print(f"MAE : {results['MAE']:.2f}/100")
    print(f"RMSE: {results['RMSE']:.2f}/100")
    print(f"R²  : {results['R2']:.4f}")
    rating = (
        "⭐⭐⭐ Excellent" if results['R2'] > 0.85 else
        "⭐⭐ Very good" if results['R2'] > 0.75 else
        "⭐  Good" if results['R2'] > 0.65 else
        "⚠️  Needs work"
    )
    print(f"\n{rating}")
    print(f"Typical error ≈ ±{results['MAE']:.1f} pts")


def show_samples(y_test, preds, n=10):
    print_section(f"Sample predictions (first {min(n, len(y_test))})")
    rows = list(zip(y_test[:n], preds[:n]))
    for idx, (actual, predicted) in enumerate(rows, 1):
        diff = abs(actual - predicted)
        badge = "✓ Excellent" if diff < 5 else "✓ Good" if diff < 10 else "~ Fair" if diff < 15 else "✗ Poor"
        print(f"[{idx:02}] actual={actual:5.1f} | pred={predicted:5.1f} | Δ={diff:4.1f}  {badge}")


def analyze_errors(y_test, preds):
    print_section("Error Analysis")
    errors = np.abs(y_test - preds)

    print(f"Error < 5 pts  : {(errors < 5).mean():.1%} of predictions")
    print(f"Error > 15 pts : {(errors >= 15).mean():.1%} (Outliers)")
    
    worst_idx = np.argmax(errors)
    print(f"\nWorst Prediction Error: {errors[worst_idx]:.1f} pts")
    print(f"Actual: {y_test[worst_idx]:.1f} vs Pred: {preds[worst_idx]:.1f}")

def save_results(results):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv("results/helpfulness_catboost_metrics.csv", index=False)
    print("\n✓ Metrics saved to results/helpfulness_catboost_metrics.csv")


def test_new_reviews(model, scaler, tfidf):
    print_section("Testing hand-crafted examples")
    examples = [
        # (review_text, sentiment, rating, expected_helpfulness_range)
        ("great product", "positive", 4, "LOW"),
        ("This works fine. No complaints.", "neutral", 3, "LOW-MEDIUM"),
        ("Excellent quality and performance. Highly recommend this to everyone!", "positive", 5, "MEDIUM"),
        ("Bought this last week. Battery lasts 48 hours. Fast charging in 2 hours. Worth the 299 dollars. Recommend for sure.", "positive", 5, "HIGH"),
        ("Terrible. Broke after 3 days. Waste of money.", "negative", 1, "LOW-MEDIUM"),
        ("The phone has amazing camera quality with 64MP sensor. Display is 6.7 inches with AMOLED screen. Battery life easily gets through a full day. Processor handles gaming smoothly. Fast charging at 65W takes only 35 minutes. Price is 599 dollars. Excellent value for money. Highly recommend.", "positive", 5, "VERY HIGH"),
        ("Not bad, decent product.", "neutral", 3, "LOW"),
        ("Poor quality materials. Not durable at all. Very disappointed with this purchase.", "negative", 2, "MEDIUM"),
        ("Love this! Amazing performance and excellent build quality. Battery lasts forever. Fast delivery. Worth every penny. Bought 2 more as gifts.", "positive", 5, "HIGH"),
        ("okay", "neutral", 3, "VERY LOW"),
        ("I used this laptop for 3 months. Screen brightness is 400 nits which is perfect for outdoor use. Battery backup is around 8 hours with normal usage. Keyboard feels premium. Weight is only 1.2 kg so very portable. Price 799 dollars. Processor is i5 11th gen. RAM 16GB. Worth buying for students and professionals.", "positive", 5, "VERY HIGH"),
        ("bad", "negative", 1, "VERY LOW")
    ]

    print("\nPredictions:")
    for i, (review_text, sentiment, rating, expected) in enumerate(examples, 1):
        X = prepare_features_for_prediction(
            text=review_text,
            sentiment=sentiment,
            rating=rating,
            tfidf=tfidf,
            scaler=scaler
        )
        helpfulness = model.predict(X)[0]
        
        # Categorize helpfulness
        if helpfulness >= 75:
            category = "VERY HIGH"
        elif helpfulness >= 60:
            category = "HIGH"
        elif helpfulness >= 40:
            category = "MEDIUM"
        elif helpfulness >= 25:
            category = "LOW-MEDIUM"
        elif helpfulness >= 15:
            category = "LOW"
        else:
            category = "VERY LOW"

        print(f"\n[Example {i}] Expected: {expected}")
        display_text = review_text[:80] + "..." if len(review_text) > 80 else review_text
        print(f'Review: "{display_text}"')
        print(f"Sentiment: {sentiment:8s} | Rating: {rating}/5")
        print(f"→ Predicted: {helpfulness:5.1f}/100 ({category})")
    
    print("="*70)


def main():
    print_section("Helpfulness model evaluation")
    model, scaler, tfidf = load_model()
    print_section("Loading cached test data")
    X_test = np.load("../../data/HELPFULNESS-DATA/X_test.npy")
    y_test = pd.read_csv("../../data/HELPFULNESS-DATA/y_test.csv").squeeze()
    print(f"Samples: {len(X_test)} | Shape: {X_test.shape}")
    results, preds = evaluate(model, X_test, y_test)
    display_results(results)
    show_samples(y_test, preds)
    analyze_errors(y_test, preds)
    save_results(results)
    test_new_reviews(model, scaler, tfidf)
    print_section("Evaluation complete")


if __name__ == "__main__":
    main()