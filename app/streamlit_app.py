from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from general_user_model_experiment.dataio import validate_event_frame
from general_user_model_experiment.evaluation import evaluate_next_action
from general_user_model_experiment.model import GeneralUserModel
from general_user_model_experiment.simulation import generate_synthetic_events


st.set_page_config(page_title="General User Model Experiment", layout="wide")


@st.cache_data(show_spinner=False)
def _generate_synthetic(n_users: int, events_per_user: int, seed: int) -> pd.DataFrame:
    events = generate_synthetic_events(
        n_users=n_users,
        events_per_user=events_per_user,
        random_state=seed,
    )
    return events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]


def _train_model(events: pd.DataFrame, clusters: int) -> tuple[GeneralUserModel, dict]:
    model = GeneralUserModel(n_clusters=clusters)
    model.fit(events)
    summary = model.summary().__dict__
    return model, summary


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    decoded = uploaded.getvalue().decode("utf-8")
    frame = pd.read_csv(io.StringIO(decoded))
    return validate_event_frame(frame)


if "gum_model" not in st.session_state:
    st.session_state.gum_model = None
if "events" not in st.session_state:
    st.session_state.events = None
if "summary" not in st.session_state:
    st.session_state.summary = None

st.title("General User Model Experiment")
st.caption(
    "Open-source GUM-inspired model: observations → propositions → retrieval/revision + behavior embeddings + proactive suggestions."
)

with st.sidebar:
    st.header("Training")
    source = st.radio("Data source", ["Synthetic demo", "Upload CSV"], index=0)
    clusters = st.slider("Clusters", min_value=2, max_value=10, value=4)

    if source == "Synthetic demo":
        n_users = st.slider("Synthetic users", min_value=2, max_value=80, value=12)
        events_per_user = st.slider("Events per user", min_value=20, max_value=400, value=120)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)

        if st.button("Train on synthetic data", type="primary"):
            with st.spinner("Generating events + training model..."):
                events = _generate_synthetic(n_users=n_users, events_per_user=events_per_user, seed=int(seed))
                model, summary = _train_model(events, clusters)
                st.session_state.gum_model = model
                st.session_state.events = events
                st.session_state.summary = summary
            st.success("Model trained on synthetic data.")

        if st.button("Export current events CSV"):
            events = st.session_state.events
            if events is None:
                st.warning("Train or upload data first.")
            else:
                csv = events.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download events.csv",
                    data=csv,
                    file_name="events.csv",
                    mime="text/csv",
                )

    else:
        uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"])
        if st.button("Train on uploaded CSV", type="primary"):
            if uploaded is None:
                st.warning("Upload a CSV first.")
            else:
                with st.spinner("Validating CSV + training model..."):
                    events = _read_uploaded_csv(uploaded)
                    model, summary = _train_model(events, clusters)
                    st.session_state.gum_model = model
                    st.session_state.events = events
                    st.session_state.summary = summary
                st.success("Model trained on uploaded CSV.")

model: GeneralUserModel | None = st.session_state.gum_model
events: pd.DataFrame | None = st.session_state.events
summary: dict | None = st.session_state.summary

if model is None or events is None or summary is None:
    st.info("Train on synthetic data or upload a CSV to start exploring the model.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Users", summary["users"])
col2.metric("Events", summary["events"])
col3.metric("Clusters", summary["clusters"])
col4.metric("Active propositions", summary["proposition_count"])

profiles = model.get_user_profiles().sort_values("user_id").reset_index(drop=True)
users = profiles["user_id"].tolist()
selected_user = st.selectbox("Select user", users, index=0)

tab_overview, tab_user, tab_memory, tab_suggestions = st.tabs(
    ["Overview", "User Explorer", "Proposition Memory", "Suggestions"]
)

with tab_overview:
    st.subheader("Cluster map")
    embed_cols = [c for c in profiles.columns if c.startswith("embedding_")]
    if len(embed_cols) >= 2:
        fig = px.scatter(
            profiles,
            x=embed_cols[0],
            y=embed_cols[1],
            color=profiles["cluster"].astype(str),
            hover_data=["user_id", "dominant_app", "dominant_action", "anomaly_score"],
            title="User embeddings by cluster",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough embedding dimensions for a scatter plot.")

    st.subheader("Next-action validation")
    try:
        metrics = evaluate_next_action(events)
        st.json(metrics)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Evaluation unavailable for this dataset: {exc}")

with tab_user:
    st.subheader(f"Profile: {selected_user}")
    profile = model.get_user_profile(selected_user)
    left, right = st.columns([1, 1])

    with left:
        st.json(profile)

        sim = model.similar_users(selected_user, top_k=5)
        st.write("Similar users")
        st.dataframe(pd.DataFrame(sim), use_container_width=True, hide_index=True)

    with right:
        st.write("Next action probe")
        app_choices = sorted(events["app"].astype(str).unique().tolist())
        action_choices = sorted(events["action"].astype(str).unique().tolist())

        probe_app = st.selectbox("App", app_choices, index=0, key="probe_app")
        probe_action = st.selectbox("Action", action_choices, index=0, key="probe_action")
        probe_hour = st.slider("Hour", min_value=0, max_value=23, value=14, key="probe_hour")
        probe_duration = st.number_input(
            "Duration (sec)", min_value=0.0, max_value=3600.0, value=45.0, step=5.0, key="probe_duration"
        )

        pred = model.predict_next_action(
            app=probe_app,
            action=probe_action,
            hour_of_day=probe_hour,
            duration_sec=float(probe_duration),
        )
        st.success(f"Predicted next action: **{pred['predicted_action']}** (confidence {pred['confidence']:.2f})")

with tab_memory:
    st.subheader("Proposition memory")
    query = st.text_input("Search propositions", value="")
    min_conf = st.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    limit = st.slider("Results", min_value=5, max_value=50, value=15)

    if query.strip():
        propositions = model.query_propositions(
            query=query,
            user_id=selected_user,
            limit=limit,
            min_confidence=min_conf,
        )
    else:
        propositions = model.list_propositions(
            user_id=selected_user,
            min_confidence=min_conf,
            limit=limit,
        )

    if not propositions:
        st.info("No propositions for this filter.")
    else:
        frame = pd.DataFrame(propositions)
        preferred_cols = [
            "statement",
            "category",
            "confidence",
            "relevance_score",
            "status",
            "version",
            "last_updated",
            "key",
        ]
        show_cols = [c for c in preferred_cols if c in frame.columns]
        st.dataframe(frame[show_cols], use_container_width=True, hide_index=True)

with tab_suggestions:
    st.subheader("Proactive suggestions (GUMBO-style scoring)")
    suggestions = model.suggest_for_user(selected_user, top_k=6)

    for s in suggestions:
        with st.container(border=True):
            st.markdown(f"### {s['title']}")
            st.write(s["rationale"])
            a, b, c, d = st.columns(4)
            a.metric("Priority", f"{s['priority_score']:.2f}")
            b.metric("Benefit", f"{s['expected_benefit']:.2f}")
            c.metric("Interrupt cost", f"{s['interruption_cost']:.2f}")
            d.metric("Confidence", f"{s['confidence']:.2f}")
            st.caption(f"Type: {s['suggestion_type']} | Urgency: {s['urgency']:.2f}")
