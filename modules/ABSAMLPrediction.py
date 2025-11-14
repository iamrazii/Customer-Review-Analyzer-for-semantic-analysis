from utils.DataProcessing import AspectExtraction
from modules.ABSA import generate_AspectOpinionPairs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    top_k_accuracy_score, log_loss
)
import matplotlib.pyplot as plt


def train_pipeline(df):

    X = df["phrase"]
    Y = df["sentiment"]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # --- Train / Val / Test Splits ---
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_vec, Y, test_size=0.3, stratify=Y, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    # --- Training curve manual epochs ---
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for _ in range(5):  # 5 pseudo-epochs
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

    plt.show()

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
        print("AUC unavailable.")


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
