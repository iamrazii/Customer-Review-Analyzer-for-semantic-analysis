import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import os
from utils.DataProcessing import AspectExtraction

class PhraseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_training_history(log_history, output_dir):
    train_loss = []
    eval_loss = []
    eval_acc = []

    for entry in log_history:
        if 'loss' in entry and 'epoch' in entry:
            train_loss.append({'epoch': entry['epoch'], 'loss': entry['loss']})
        if 'eval_loss' in entry and 'epoch' in entry:
            eval_loss.append({'epoch': entry['epoch'], 'loss': entry['eval_loss']})
        if 'eval_accuracy' in entry and 'epoch' in entry:
            eval_acc.append({'epoch': entry['epoch'], 'accuracy': entry['eval_accuracy']})

    df_train_loss = pd.DataFrame(train_loss)
    df_eval_loss = pd.DataFrame(eval_loss)
    df_eval_acc = pd.DataFrame(eval_acc)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    if not df_train_loss.empty:
        plt.plot(df_train_loss['epoch'], df_train_loss['loss'], label='Training Loss')
    if not df_eval_loss.empty:
        plt.plot(df_eval_loss['epoch'], df_eval_loss['loss'], label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_curve.png")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    if not df_eval_acc.empty:
        plt.plot(df_eval_acc['epoch'], df_eval_acc['accuracy'], label='Validation Accuracy', color='green')
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_curve.png")
    plt.close()

def train_simple_absa(df, model_name="distilbert-base-uncased", output_dir="models/absa_pytorch"):
    
    print("Preparing data for evaluation/training...")
    texts = df['phrase'].astype(str).tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df['sentiment'])
    num_classes = len(le.classes_)

    # Split Data (Must use same random_state to match original training)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    model_path = output_dir if os.path.exists(output_dir) and os.path.exists(f"{output_dir}/labels.json") else model_name
    do_train = True

    if model_path == output_dir:
        print(f"[INFO] Loading existing model from '{output_dir}'...")
        do_train = False
    else:
        print(f"[INFO] No saved model found. Initializing '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_classes
    )
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Helper to create dataset
    def tokenize_data(t, l):
        enc = tokenizer(t, truncation=True, padding=True, max_length=64)
        return PhraseDataset(enc, l)

    train_ds = tokenize_data(train_texts, train_labels)
    val_ds = tokenize_data(val_texts, val_labels)
    test_ds = tokenize_data(test_texts, test_labels)

    #  Initialize Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_dir=f"{output_dir}/logs",
        learning_rate=2e-5,
        weight_decay=0.01,
        use_cpu=False if device == "cuda" else True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    if do_train:
        print("Starting training...")
        trainer.train()
        
        # Save artifacts
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        pd.Series(le.classes_).to_json(f"{output_dir}/labels.json")
        
        print("Generating training graphs...")
        plot_training_history(trainer.state.log_history, output_dir)
    else:
        print("Skipping training (Model already exists).")

    print("\n=== Evaluating on Test Set ===")
    preds_output = trainer.predict(test_ds)
    
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = preds_output.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    # AUC
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs, multi_class='ovr')
    except:
        auc = 0.0 # Handle case where only 1 class is present in test set



    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)



    print("\n=== Evaluating on Validation Set ===")
    val_output = trainer.predict(val_ds)
    val_logits = val_output.predictions
    val_preds = np.argmax(val_logits, axis=1)
    val_labels_ids = val_output.label_ids
    
    print("\nValidation Classification Report:")
    print(classification_report(val_labels_ids, val_preds, target_names=le.classes_.astype(str)))

    print("\n--- Test Set Report ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Save Report
    with open(f"{output_dir}/test_report.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nAUC: {auc}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Plot CM
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    return "Process Complete."


def predict_sentiment(phrase, model_path="models/absa_pytorch"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    phrases = AspectExtraction(phrase)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)    
    labels = pd.read_json(f"{model_path}/labels.json", typ='series').values
    inputs = tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    results = []
    for i, phrase in enumerate(phrases):
        label = labels[preds[i]]
        confidence = probs[i][preds[i]].item()
        results.append((phrase, label, f"{confidence:.2%}"))
        
    return results