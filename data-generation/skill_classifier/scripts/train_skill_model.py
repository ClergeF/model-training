import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = CLASSIFIER_ROOT / "datasets" / "skill_dataset.jsonl"
MODEL_OUTPUT_DIR = CLASSIFIER_ROOT / "saved_models"

MODEL_DISPLAY_NAME = "Technical Development Skill Categorization model"
MODEL_FILENAME = "technical_development_skill_categorization_model.joblib"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return pd.DataFrame(rows)


def clean_dataset(df):
    df = df[["input_text", "skill"]].copy()
    df = df.dropna(subset=["input_text", "skill"])
    df["input_text"] = df["input_text"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df = df[df["input_text"] != ""]
    df = df[df["skill"] != ""]
    df = df.drop_duplicates(subset=["input_text", "skill"])
    return df


def show_class_counts(df):
    print("\nSkill counts:")
    print(df["skill"].value_counts())
    print("\nTotal rows:", len(df))
    print("Total skills:", df["skill"].nunique())


def train_and_evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test):
    print(f"\nTraining {model_name}...")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"\n===== {model_name} Results =====")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return {
        "name": model_name,
        "pipeline": pipeline,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": predictions,
    }


def predict_skill(model, text):
    return {
        "input_text": text,
        "skill": model.predict([text])[0],
    }


def main():
    print("Loading dataset...")
    print(f"Dataset: {DATASET_FILE}")

    df = load_jsonl(DATASET_FILE)
    df = clean_dataset(df)
    show_class_counts(df)

    X = df["input_text"]
    y = df["skill"]

    split_kwargs = {"test_size": 0.20, "random_state": 42}
    min_class_count = int(y.value_counts().min())
    if min_class_count >= 2:
        split_kwargs["stratify"] = y
    else:
        rare_skills = y.value_counts()[y.value_counts() < 2].index.tolist()
        print(
            "\nNote: stratified split skipped — these skills have fewer than 2 rows:",
            rare_skills,
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    linear_svc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
        )),
        ("classifier", LinearSVC(
            class_weight="balanced",
            random_state=42,
        )),
    ])

    logistic_regression_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    results = [
        train_and_evaluate_model(
            "TF-IDF + LinearSVC",
            linear_svc_pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        ),
        train_and_evaluate_model(
            "TF-IDF + Logistic Regression",
            logistic_regression_pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        ),
    ]

    best_result = max(results, key=lambda result: result["macro_f1"])
    best_model = best_result["pipeline"]

    print("\nBest model:", best_result["name"])
    print("Best Macro F1:", round(best_result["macro_f1"], 4))

    model_path = MODEL_OUTPUT_DIR / MODEL_FILENAME
    joblib.dump(best_model, model_path)

    metadata_path = MODEL_OUTPUT_DIR / "technical_development_skill_categorization_model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_DISPLAY_NAME,
                "artifact": MODEL_FILENAME,
                "best_classifier": best_result["name"],
                "macro_f1": round(best_result["macro_f1"], 4),
                "accuracy": round(best_result["accuracy"], 4),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved {MODEL_DISPLAY_NAME} to: {model_path}")
    print(f"Metadata: {metadata_path}")

    test_text = (
        "I fixed the webhook by checking the payload and realizing the message "
        "text was nested inside body.event.text."
    )
    prediction = predict_skill(best_model, test_text)
    print("\nTest Prediction:")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
