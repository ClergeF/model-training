import json
from pathlib import Path

import joblib

CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    CLASSIFIER_ROOT
    / "saved_models"
    / "technical_development_skill_categorization_model.joblib"
)


def main():
    model = joblib.load(MODEL_PATH)
    text = input("Enter story chunk: ")

    result = {
        "input_text": text,
        "skill": model.predict([text])[0],
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
