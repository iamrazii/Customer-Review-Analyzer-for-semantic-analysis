from utils.DataProcessing import AspectExtraction
from modules.ABSA import generate_AspectOpinionPairs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
     log_loss
)
import matplotlib.pyplot as plt
import os
import joblib

def train_pipeline(df, output_dir="models/absa_ml"):
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = df["phrase"]
    Y = df["sentiment"]
    
    model_path = os.path.join(output_dir, "model.pkl")
    vec_path = os.path.join(output_dir, "vectorizer.pkl")

    if os.path.exists(model_path) and os.path.exists(vec_path):
        print(f"[INFO] Loading existing model and vectorizer from '{output_dir}'...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        
        # Transform data using loaded vectorizer 
        X_vec = vectorizer.transform(X)
        model_loaded = True
    else:
        print("[INFO] No saved model found. Training from scratch...")
        vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
        X_vec = vectorizer.fit_transform(X)
        model_loaded = False

    # --- Train / Val / Test Splits ---
    # We always split to report metrics on the specific dataset provided
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_vec, Y, test_size=0.3, stratify=Y, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42
    )

    if not model_loaded:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')

        # --- Training curve manual epochs ---
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        print("Starting training loop...")
        for i in range(5):  # 5 pseudo-epochs
            model.fit(X_train, Y_train)

            # Predict probabilities
            train_probs = model.predict_proba(X_train)
            val_probs = model.predict_proba(X_val)

            # Compute losses
            train_losses.append(log_loss(Y_train, train_probs))
            val_losses.append(log_loss(Y_val, val_probs))

            # Compute accuracy
            train_accs.append(accuracy_score(Y_train, model.predict(X_train)))
            val_accs.append(accuracy_score(Y_val, model.predict(X_val)))
            print(f"  Epoch {i+1}/5 - Loss: {val_losses[-1]:.4f} - Acc: {val_accs[-1]:.4f}")

        # --- Plots ---
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Validation Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Save plots before showing
        plt.savefig("graphs/ABSA/training_curves.png")
        # plt.show() # Optional: Comment out if running on server without display
        plt.close()

        # --- SAVE MODEL ---
        print(f"Saving model to '{output_dir}'...")
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vec_path)
    else:
        print("Skipping training loop (Model Loaded).")


    # ---- VALIDATION REPORT ----
    val_pred = model.predict(X_val)

    print("\n===== VALIDATION METRICS =====")
    print("Accuracy:", accuracy_score(Y_val, val_pred))
    print("Precision:", precision_score(Y_val, val_pred, average='macro'))
    print("Recall:", recall_score(Y_val, val_pred, average='macro'))
    print("F1-Score:", f1_score(Y_val, val_pred, average='macro'))
    print("\nClassification Report:\n", classification_report(Y_val, val_pred))

    # AUC (One-vs-Rest)
    val_prob = model.predict_proba(X_val)
    try:
        auc = roc_auc_score(Y_val, val_prob, multi_class='ovr')
        print("AUC Score:", auc)
    except:
        print("AUC unavailable (possibly single class in validation set).")


    # Confusion matrix
    print("\nConfusion Matrix:\n", confusion_matrix(Y_val, val_pred))


    # ---- TEST METRICS ----
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)

    print("\n===== TEST METRICS =====")
    print("Accuracy:", accuracy_score(Y_test, test_pred))
    print("Precision:", precision_score(Y_test, test_pred, average='macro'))
    print("Recall:", recall_score(Y_test, test_pred, average='macro'))
    print("F1-Score:", f1_score(Y_test, test_pred, average='macro'))
    print("\nClassification Report:\n", classification_report(Y_test, test_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(Y_test, test_pred))

    # AUC for test
    try:
        auc_test = roc_auc_score(Y_test, test_prob, multi_class='ovr')
        print("AUC Score:", auc_test)
    except:
        print("AUC unavailable.")

    return model, vectorizer


def predict_sentiment(model, vectorizer, text):
    aspect_pairs = AspectExtraction(text)
    nm, _ = generate_AspectOpinionPairs(aspect_pairs, text)

    predictions = []
    for aspect, opinion in nm:
        phrase = f"{opinion} {aspect}"
        X_test = vectorizer.transform([phrase])
        Y_pred = model.predict(X_test)[0]
        predictions.append((phrase, Y_pred))

    return dict(predictions)