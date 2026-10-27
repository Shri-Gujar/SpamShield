import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline


# 1. Load the SMS Spam Collection dataset
print("\nLoading dataset...")

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

data = data.dropna(subset=["label", "text"])
data["label"] = data["label"].str.strip().str.lower()
data["text"] = data["text"].astype(str).str.strip()

data = data[data["label"].isin(["ham", "spam"])]
data = data[data["text"] != ""]

print("Total messages:", len(data))
print("Ham messages:", (data["label"] == "ham").sum())
print("Spam messages:", (data["label"] == "spam").sum())


# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)


# 3. Define both models
models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB())
    ]),

    "Balanced Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])
}


# 4. Train and evaluate both models
results = []
trained_models = {}

for name, model in models.items():
    print("\n" + "=" * 55)
    print("MODEL:", name)
    print("=" * 55)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    spam_precision = precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    spam_recall = recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    spam_f1 = f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Spam Precision: {spam_precision:.4f}")
    print(f"Spam Recall:    {spam_recall:.4f}")
    print(f"Spam F1-score:  {spam_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion Matrix (labels: ham, spam):")
    print(confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Spam Precision": spam_precision,
        "Spam Recall": spam_recall,
        "Spam F1-score": spam_f1
    })

    trained_models[name] = model


# 5. Compare models
results_df = pd.DataFrame(results).sort_values(
    by=["Spam F1-score", "Spam Recall", "Spam Precision"],
    ascending=False
)

print("\n\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 6. Save the best model separately
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

with open("spam_model_improved.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nSelected model:", best_name)
print("Saved as: spam_model_improved.pkl")
print("\nYour original spam_model.pkl has NOT been changed.")
print("Model training and evaluation completed successfully!")