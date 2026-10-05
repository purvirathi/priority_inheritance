from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "priority_inversion_risk_model.pkl"
)

FEATURE_NAMES = [
    "task_priority",
    "owner_priority",
    "waiting_time",
    "arrival_time",
    "resource_locked",
    "higher_priority_waiting",
    "priority_difference",
]


def calculate_risk_label(row):
    """
    Domain-informed synthetic labeling rule used only
    to generate the training data for this educational demo.
    """
    score = 0

    if row["resource_locked"]:
        score += 2

    if row["priority_difference"] > 0:
        score += 2

    if row["waiting_time"] >= 5:
        score += 2

    if row["higher_priority_waiting"]:
        score += 2

    if row["waiting_time"] >= 10:
        score += 1

    if score >= 6:
        return "HIGH"

    if score >= 3:
        return "MEDIUM"

    return "LOW"


def create_synthetic_dataset():
    rows = []

    for task_priority in [1, 2, 3]:
        for owner_priority in [1, 2, 3]:
            for waiting_time in range(0, 13):
                for arrival_time in range(0, 6):
                    for resource_locked in [0, 1]:
                        for higher_priority_waiting in [0, 1]:

                            priority_difference = (
                                task_priority
                                - owner_priority
                            )

                            row = {
                                "task_priority": task_priority,
                                "owner_priority": owner_priority,
                                "waiting_time": waiting_time,
                                "arrival_time": arrival_time,
                                "resource_locked": resource_locked,
                                "higher_priority_waiting":
                                    higher_priority_waiting,
                                "priority_difference":
                                    priority_difference,
                            }

                            row["risk"] = calculate_risk_label(row)

                            rows.append(row)

    return pd.DataFrame(rows)


def train_model():
    df = create_synthetic_dataset()

    X = df[FEATURE_NAMES]
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save a versioned bundle. This is intentionally a dict
    # so the feature schema travels with the model.
    bundle = {
        "model": model,
        "features": FEATURE_NAMES,
        "model_type": "DecisionTreeClassifier",
        "dataset_type": "synthetic_domain_informed",
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
    )

    return {
        "accuracy": accuracy,
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
        "model_path": str(MODEL_PATH),
    }


def load_model():
    """
    Returns the sklearn estimator.

    Supports both:
    1. the current saved bundle {"model": ...}
    2. an older raw sklearn estimator
    """
    if not MODEL_PATH.exists():
        train_model()

    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict):
        if "model" not in saved:
            raise ValueError(
                "Saved AI model bundle is missing the 'model' key."
            )

        return saved["model"]

    return saved


def predict_risk(
    task_priority,
    owner_priority,
    waiting_time,
    arrival_time,
    resource_locked,
    higher_priority_waiting,
):
    priority_difference = (
        task_priority
        - owner_priority
    )

    model_input = pd.DataFrame(
        [{
            "task_priority": task_priority,
            "owner_priority": owner_priority,
            "waiting_time": waiting_time,
            "arrival_time": arrival_time,
            "resource_locked": resource_locked,
            "higher_priority_waiting": higher_priority_waiting,
            "priority_difference": priority_difference,
        }],
        columns=FEATURE_NAMES,
    )

    model = load_model()

    prediction = model.predict(
        model_input
    )[0]

    probabilities = model.predict_proba(
        model_input
    )[0]

    probability_map = {
        class_name: probability
        for class_name, probability
        in zip(model.classes_, probabilities)
    }

    return {
        "prediction": prediction,
        "probabilities": probability_map,
        "features": model_input.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    result = train_model()

    print("\nMODEL TRAINED")
    print(
        f"Accuracy: {result['accuracy'] * 100:.2f}%"
    )
    print(
        f"Saved to: {result['model_path']}"
    )

    demo = predict_risk(
        task_priority=3,
        owner_priority=1,
        waiting_time=9,
        arrival_time=1,
        resource_locked=1,
        higher_priority_waiting=1,
    )

    print("\nLIVE PRIORITY-INVERSION RISK TEST")
    print(
        f"Predicted Risk: {demo['prediction']}"
    )

    for label, probability in sorted(
        demo["probabilities"].items()
    ):
        print(
            f"{label}: {probability * 100:.2f}%"
        )
