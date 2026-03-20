from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import build_next_action_training_set


def evaluate_next_action(events: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    X, y = build_next_action_training_set(events)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["app", "action"]),
                        ("num", StandardScaler(), ["hour", "duration_sec"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    return {
        "samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
