"""
app.py — Streamlit Web Interface for House Price Prediction
=============================================================

PURPOSE:
    Provides a browser-based UI where users can:
    1. "Predict" page  — Fill in property details and get a price prediction
       from the deployed MLflow model server.
    2. "Project overview" page — See the pipeline flow, dataset preview, and
       quickstart instructions.

HOW TO RUN:
    streamlit run app.py      (or: make app)

REQUIREMENTS:
    The MLflow prediction server must be running at http://127.0.0.1:8000/invocations
    (or set the PREDICTION_URL environment variable). Start it with: python run_deployment.py
"""

import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


PREDICTION_URL_DEFAULT = os.environ.get(
    "PREDICTION_URL", "http://127.0.0.1:8000/invocations"
)

# Feature schema derived from the pipeline and predictor step
FEATURES: List[Tuple[str, str, float]] = [
    ("Order", "int", 1),
    ("PID", "int", 5286),
    ("MS SubClass", "int", 20),
    ("Lot Frontage", "float", 80.0),
    ("Lot Area", "int", 9600),
    ("Overall Qual", "int", 5),
    ("Overall Cond", "int", 7),
    ("Year Built", "int", 1961),
    ("Year Remod/Add", "int", 1961),
    ("Mas Vnr Area", "float", 0.0),
    ("BsmtFin SF 1", "float", 700.0),
    ("BsmtFin SF 2", "float", 0.0),
    ("Bsmt Unf SF", "float", 150.0),
    ("Total Bsmt SF", "float", 850.0),
    ("1st Flr SF", "int", 856),
    ("2nd Flr SF", "int", 854),
    ("Low Qual Fin SF", "int", 0),
    ("Gr Liv Area", "float", 1710.0),
    ("Bsmt Full Bath", "int", 1),
    ("Bsmt Half Bath", "int", 0),
    ("Full Bath", "int", 1),
    ("Half Bath", "int", 0),
    ("Bedroom AbvGr", "int", 3),
    ("Kitchen AbvGr", "int", 1),
    ("TotRms AbvGrd", "int", 7),
    ("Fireplaces", "int", 2),
    ("Garage Yr Blt", "int", 1961),
    ("Garage Cars", "int", 2),
    ("Garage Area", "float", 500.0),
    ("Wood Deck SF", "float", 210.0),
    ("Open Porch SF", "float", 0.0),
    ("Enclosed Porch", "float", 0.0),
    ("3Ssn Porch", "float", 0.0),
    ("Screen Porch", "float", 0.0),
    ("Pool Area", "float", 0.0),
    ("Misc Val", "float", 0.0),
    ("Mo Sold", "int", 5),
    ("Yr Sold", "int", 2010),
]


def build_input_form() -> Dict[str, float]:
    values: Dict[str, float] = {}

    st.subheader("Property details")
    st.caption("Fill in the property attributes used by the model.")

    sections = [
        (
            "Lot + Quality",
            [
                "Order",
                "PID",
                "MS SubClass",
                "Lot Frontage",
                "Lot Area",
                "Overall Qual",
                "Overall Cond",
            ],
        ),
        (
            "Year + Remodeling",
            ["Year Built", "Year Remod/Add"],
        ),
        (
            "Basement",
            [
                "Mas Vnr Area",
                "BsmtFin SF 1",
                "BsmtFin SF 2",
                "Bsmt Unf SF",
                "Total Bsmt SF",
                "Bsmt Full Bath",
                "Bsmt Half Bath",
            ],
        ),
        (
            "Living area",
            [
                "1st Flr SF",
                "2nd Flr SF",
                "Low Qual Fin SF",
                "Gr Liv Area",
                "Full Bath",
                "Half Bath",
                "Bedroom AbvGr",
                "Kitchen AbvGr",
                "TotRms AbvGrd",
                "Fireplaces",
            ],
        ),
        (
            "Garage",
            ["Garage Yr Blt", "Garage Cars", "Garage Area"],
        ),
        (
            "Porch + Outdoor",
            [
                "Wood Deck SF",
                "Open Porch SF",
                "Enclosed Porch",
                "3Ssn Porch",
                "Screen Porch",
                "Pool Area",
                "Misc Val",
            ],
        ),
        (
            "Sale timing",
            ["Mo Sold", "Yr Sold"],
        ),
    ]

    feature_map = {name: (kind, default) for name, kind, default in FEATURES}

    for section_title, fields in sections:
        with st.expander(section_title, expanded=True):
            for field in fields:
                kind, default = feature_map[field]
                if kind == "int":
                    values[field] = int(
                        st.number_input(
                            field,
                            value=int(default),
                            step=1,
                        )
                    )
                else:
                    values[field] = float(
                        st.number_input(
                            field,
                            value=float(default),
                            step=1.0,
                        )
                    )

    return values


def call_prediction_service(url: str, record: Dict[str, float]) -> Dict[str, object]:
    payload = {"dataframe_records": [record]}
    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def render_prediction():
    st.title("House Price Prediction")
    st.write(
        "Enter property details and get a price prediction using the deployed model service."
    )

    # Check if we have a local model serialized
    local_model_path = "model.joblib"
    has_local_model = os.path.exists(local_model_path)

    with st.sidebar:
        st.header("Prediction settings")
        if has_local_model:
            st.info("💡 Standalone mode: Running model in-memory from model.joblib.")
        else:
            prediction_url = st.text_input("Prediction URL", PREDICTION_URL_DEFAULT)
            st.caption("Default matches the MLflow model server from the deployment pipeline.")

    with st.form("prediction_form"):
        record = build_input_form()
        submitted = st.form_submit_button("Predict price")

    if not submitted:
        return

    st.subheader("Prediction result")
    
    if has_local_model:
        try:
            # 1. Load standalone model
            pipeline = joblib.load(local_model_path)
            
            # 2. Build DataFrame in expected format/order
            df = pd.DataFrame([record])
            expected_columns = [f[0] for f in FEATURES]
            df = df[expected_columns]
            
            # 3. Log transform Gr Liv Area to match training
            if "Gr Liv Area" in df.columns:
                df["Gr Liv Area"] = np.log1p(df["Gr Liv Area"])
            
            # 4. Predict and invert log transform
            pred_log = pipeline.predict(df)[0]
            pred_price = np.expm1(pred_log)
            
            st.success("Prediction completed (using standalone model).")
            st.metric("Predicted SalePrice", f"${pred_price:,.2f}")
            with st.expander("Show Technical Details"):
                st.write({
                    "model_source": "Local model.joblib",
                    "log1p_predicted_val": float(pred_log),
                    "exponentiated_val": float(pred_price)
                })
        except Exception as exc:
            st.error("Failed to run prediction using local model.")
            st.code(str(exc))
    else:
        # Fallback: Call prediction service via REST API
        api_record = record.copy()
        if "Gr Liv Area" in api_record:
            api_record["Gr Liv Area"] = np.log1p(api_record["Gr Liv Area"])
            
        try:
            response_json = call_prediction_service(prediction_url, api_record)
        except requests.RequestException as exc:
            st.error("Prediction service is not reachable.")
            st.info(
                "Start the deployment pipeline first, then retry. "
                "Run `python run_deployment.py` to deploy the model."
            )
            st.code(str(exc))
            return

        prediction = None
        if isinstance(response_json, dict):
            prediction = response_json.get("predictions")
        if prediction is None:
            prediction = response_json

        st.success("Prediction completed (using MLflow server).")
        st.write("Raw response:")
        st.json(response_json)

        if isinstance(prediction, list) and prediction:
            pred_val = prediction[0]
            # If the response is in log-scale (typical < 20 for housing prices), convert back
            if pred_val < 20:
                pred_price = np.expm1(pred_val)
            else:
                pred_price = pred_val
            st.metric("Predicted SalePrice", f"${pred_price:,.2f}")


def render_project_overview():
    st.title("Project overview")
    st.write(
        "This application wraps the end-to-end ZenML + MLflow pipeline for Ames Housing "
        "price prediction."
    )

    st.subheader("Pipeline flow")
    st.markdown(
        "- Data ingestion: ZIP ingestion reads the AmesHousing CSV.\n"
        "- Missing values: configurable strategy (mean by default).\n"
        "- Feature engineering: log transform on `Gr Liv Area` + `SalePrice`.\n"
        "- Outlier detection: Z-score filtering on numeric columns.\n"
        "- Train/test split: simple 80/20 split.\n"
        "- Model: scikit-learn pipeline with preprocessing + Linear Regression.\n"
        "- Evaluation: MSE + R2.\n"
        "- Deployment: MLflow model server via ZenML deployment pipeline."
    )

    data_path = os.path.join("extracted_data", "AmesHousing.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.subheader("Dataset preview")
        st.dataframe(df.head(10))
        st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    else:
        st.info("Dataset preview unavailable. The AmesHousing.csv file was not found.")

    st.subheader("How to run")
    st.markdown(
        "1. Train + deploy the model: `python run_deployment.py`\n"
        "2. Start the MLflow prediction server (if not already running).\n"
        "3. Launch the app: `streamlit run app.py`"
    )


def main():
    st.set_page_config(page_title="House Price Prediction", layout="wide")

    page = st.sidebar.radio("Navigate", ["Predict", "Project overview"])

    if page == "Predict":
        render_prediction()
    else:
        render_project_overview()


if __name__ == "__main__":
    main()
