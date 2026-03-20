from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureArtifacts, build_next_action_training_set, build_user_features
from .propositions import PropositionMemory
from .suggestions import SuggestionEngine


@dataclass
class ModelSummary:
    users: int
    events: int
    clusters: int
    proposition_count: int
    top_features: list[str]


class GeneralUserModel:
    """General-purpose user model inspired by GUM (propositions + profile features)."""

    def __init__(self, n_clusters: int = 4, embedding_dim: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self._feature_artifacts: FeatureArtifacts | None = None
        self._fitted = False

        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._clusterer: KMeans | None = None
        self._anomaly: IsolationForest | None = None

        self._next_action_model: Pipeline | None = None
        self._next_action_fallback: str | None = None

        self._profiles: pd.DataFrame | None = None
        self._events_seen: int = 0

        self._proposition_memory: PropositionMemory = PropositionMemory()
        self._suggestion_engine: SuggestionEngine = SuggestionEngine()

    def fit(self, events: pd.DataFrame) -> "GeneralUserModel":
        if events.empty:
            raise ValueError("events dataframe is empty")

        events = events.copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

        self._events_seen = len(events)
        self._feature_artifacts = build_user_features(events)
        user_features = self._feature_artifacts.user_features.copy()

        numeric_cols = [
            c
            for c in user_features.columns
            if c not in {"user_id", "dominant_app", "dominant_action"}
            and pd.api.types.is_numeric_dtype(user_features[c])
        ]

        X_numeric = user_features[numeric_cols].fillna(0.0)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_numeric)

        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        actual_embedding_dim = max(1, min(self.embedding_dim, max_components))

        self._pca = PCA(n_components=actual_embedding_dim, random_state=self.random_state)
        embeddings = self._pca.fit_transform(X_scaled)

        n_users = len(user_features)
        if n_users <= 1:
            self._clusterer = None
            clusters = np.zeros(n_users, dtype=int)
        else:
            actual_clusters = max(2, min(self.n_clusters, n_users))
            self._clusterer = KMeans(
                n_clusters=actual_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
            clusters = self._clusterer.fit_predict(embeddings)

        if n_users <= 2:
            self._anomaly = None
            anomaly_scores = np.zeros(n_users, dtype=float)
        else:
            self._anomaly = IsolationForest(
                random_state=self.random_state,
                contamination="auto",
                n_jobs=1,
            )
            self._anomaly.fit(X_scaled)
            anomaly_scores = -self._anomaly.score_samples(X_scaled)

        profile_df = user_features.copy()
        for i in range(embeddings.shape[1]):
            profile_df[f"embedding_{i + 1}"] = embeddings[:, i]
        profile_df["cluster"] = clusters
        profile_df["anomaly_score"] = anomaly_scores
        self._profiles = profile_df

        # Next-action predictor from local transition context.
        self._next_action_model = None
        self._next_action_fallback = None
        try:
            X_next, y_next = build_next_action_training_set(events)
            if y_next.nunique() < 2:
                self._next_action_fallback = str(y_next.mode().iloc[0])
            else:
                categorical = ["app", "action"]
                numeric = ["hour", "duration_sec"]

                preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("num", StandardScaler(), numeric),
                    ]
                )
                model = LogisticRegression(max_iter=500)
                self._next_action_model = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
                self._next_action_model.fit(X_next, y_next)
        except ValueError:
            self._next_action_fallback = str(events["action"].astype(str).mode().iloc[0])

        # Proposition memory (GUM-like infer/retrieve/revise).
        self._proposition_memory = PropositionMemory()
        self._proposition_memory.ingest_events(events)

        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(events_df) first.")

    def summary(self) -> ModelSummary:
        self._require_fit()
        assert self._profiles is not None

        numeric_cols = [
            c
            for c in self._profiles.columns
            if c not in {"user_id", "dominant_app", "dominant_action"}
            and pd.api.types.is_numeric_dtype(self._profiles[c])
        ]
        stdev_rank = self._profiles[numeric_cols].std().sort_values(ascending=False).head(6).index.tolist()

        return ModelSummary(
            users=int(self._profiles["user_id"].nunique()),
            events=int(self._events_seen),
            clusters=int(self._profiles["cluster"].nunique()),
            proposition_count=int(self._proposition_memory.proposition_count(status="active")),
            top_features=stdev_rank,
        )

    def get_user_profiles(self) -> pd.DataFrame:
        self._require_fit()
        assert self._profiles is not None
        return self._profiles.copy()

    def get_user_profile(self, user_id: str) -> dict:
        profiles = self.get_user_profiles()
        match = profiles[profiles["user_id"] == user_id]
        if match.empty:
            raise KeyError(f"Unknown user_id: {user_id}")
        row = match.iloc[0]
        return {
            "user_id": str(row["user_id"]),
            "cluster": int(row["cluster"]),
            "event_count": float(row.get("event_count", 0.0)),
            "session_count": float(row.get("session_count", 0.0)),
            "app_entropy": float(row["app_entropy"]),
            "action_entropy": float(row["action_entropy"]),
            "switch_rate": float(row["switch_rate"]),
            "dominant_app": str(row["dominant_app"]),
            "dominant_action": str(row["dominant_action"]),
            "active_hour_mean": float(row.get("active_hour_mean", 0.0)),
            "active_hour_std": float(row.get("active_hour_std", 0.0)),
            "mean_session_duration": float(row.get("mean_session_duration", 0.0)),
            "anomaly_score": float(row["anomaly_score"]),
        }

    def similar_users(self, user_id: str, top_k: int = 5) -> list[dict]:
        self._require_fit()
        assert self._profiles is not None

        embed_cols = [c for c in self._profiles.columns if c.startswith("embedding_")]
        matrix = self._profiles[embed_cols].to_numpy(dtype=float)
        sims = cosine_similarity(matrix)

        idx_lookup = {u: idx for idx, u in enumerate(self._profiles["user_id"].tolist())}
        if user_id not in idx_lookup:
            raise KeyError(f"Unknown user_id: {user_id}")

        src_idx = idx_lookup[user_id]
        scores = sims[src_idx]

        results = []
        for idx in np.argsort(scores)[::-1]:
            other_user = self._profiles.iloc[idx]["user_id"]
            if other_user == user_id:
                continue
            results.append({"user_id": str(other_user), "similarity": float(scores[idx])})
            if len(results) >= top_k:
                break
        return results

    def predict_next_action(
        self,
        app: str,
        action: str,
        hour_of_day: int,
        duration_sec: float = 0.0,
    ) -> dict:
        self._require_fit()

        if self._next_action_model is None:
            fallback = self._next_action_fallback or action
            return {
                "predicted_action": str(fallback),
                "confidence": 0.51,
            }

        X = pd.DataFrame(
            [
                {
                    "app": app,
                    "action": action,
                    "hour": int(hour_of_day),
                    "duration_sec": float(duration_sec),
                }
            ]
        )
        proba = self._next_action_model.predict_proba(X)[0]
        classes = self._next_action_model.classes_
        best_idx = int(np.argmax(proba))
        return {
            "predicted_action": str(classes[best_idx]),
            "confidence": float(proba[best_idx]),
        }

    def list_propositions(
        self,
        user_id: str | None = None,
        status: str | None = "active",
        min_confidence: float = 0.0,
        limit: int = 200,
    ) -> list[dict]:
        self._require_fit()
        return self._proposition_memory.list_propositions(
            user_id=user_id,
            status=status,
            min_confidence=min_confidence,
            limit=limit,
        )

    def query_propositions(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        self._require_fit()
        return self._proposition_memory.query(
            query=query,
            user_id=user_id,
            limit=limit,
            min_confidence=min_confidence,
        )

    def suggest_for_user(
        self,
        user_id: str,
        top_k: int = 5,
        min_confidence: float = 0.2,
    ) -> list[dict]:
        self._require_fit()
        profile = self.get_user_profile(user_id)
        propositions = self.list_propositions(
            user_id=user_id,
            status="active",
            min_confidence=min_confidence,
            limit=200,
        )
        return self._suggestion_engine.generate(
            user_id=user_id,
            profile=profile,
            propositions=propositions,
            top_k=top_k,
        )

    def save(self, path: str | Path) -> None:
        self._require_fit()
        payload = {
            "n_clusters": self.n_clusters,
            "embedding_dim": self.embedding_dim,
            "random_state": self.random_state,
            "feature_artifacts": self._feature_artifacts,
            "fitted": self._fitted,
            "scaler": self._scaler,
            "pca": self._pca,
            "clusterer": self._clusterer,
            "anomaly": self._anomaly,
            "next_action_model": self._next_action_model,
            "next_action_fallback": self._next_action_fallback,
            "profiles": self._profiles,
            "events_seen": self._events_seen,
            "proposition_memory": self._proposition_memory,
            "suggestion_engine": self._suggestion_engine,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "GeneralUserModel":
        payload = joblib.load(path)
        model = cls(
            n_clusters=payload["n_clusters"],
            embedding_dim=payload["embedding_dim"],
            random_state=payload["random_state"],
        )
        model._feature_artifacts = payload["feature_artifacts"]
        model._fitted = payload["fitted"]
        model._scaler = payload["scaler"]
        model._pca = payload["pca"]
        model._clusterer = payload["clusterer"]
        model._anomaly = payload["anomaly"]
        model._next_action_model = payload.get("next_action_model")
        model._next_action_fallback = payload.get("next_action_fallback")
        model._profiles = payload["profiles"]
        model._events_seen = payload["events_seen"]
        model._proposition_memory = payload.get("proposition_memory", PropositionMemory())
        model._suggestion_engine = payload.get("suggestion_engine", SuggestionEngine())
        return model
