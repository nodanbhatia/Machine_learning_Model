import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Streamlit configuration
# -----------------------------
st.set_page_config(
    page_title="Car Selling Price Predictor",
    page_icon="🚗",
    layout="wide",
)

TARGET = "Selling_Price_Lakh"
DEFAULT_CSV = "car_price_prediction_10000.csv"

REQUIRED_COLUMNS = [
    "Car_Name", "Year", "Present_Price_Lakh", "Kms_Driven",
    "Fuel_Type", "Seller_Type", "Transmission", "Owner",
    "Engine_CC", "Mileage_Kmpl", "City", "Color", TARGET
]


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def read_uploaded_file(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def read_default_file():
    if not os.path.exists(DEFAULT_CSV):
        return None
    return pd.read_csv(DEFAULT_CSV)


def validate_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, "Missing columns: " + ", ".join(missing)
    return True, ""


def prepare_data(df):
    df = df.copy().drop_duplicates()

    if "Car_ID" in df.columns:
        df = df.drop(columns="Car_ID")

    numeric_cols = [
        "Year", "Present_Price_Lakh", "Kms_Driven", "Owner",
        "Engine_CC", "Mileage_Kmpl", TARGET
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Car_Age" not in df.columns:
        df["Car_Age"] = datetime.now().year - df["Year"]
    else:
        df["Car_Age"] = pd.to_numeric(df["Car_Age"], errors="coerce")

    df.loc[df["Car_Age"] < 0, "Car_Age"] = np.nan
    return df.dropna().reset_index(drop=True)


# -----------------------------
# Model training
# -----------------------------
def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
    ])

    return preprocessor, numeric, categorical


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor, numeric, categorical = make_preprocessor(X)

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "Random Forest Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2 Score": r2_score(y_test, pred)
        })
        predictions[name] = pred

    results_df = pd.DataFrame(results).sort_values(
        "R2 Score", ascending=False
    ).reset_index(drop=True)

    best_name = results_df.loc[0, "Model"]

    return (
        models,
        models[best_name],
        best_name,
        results_df,
        predictions,
        X_test,
        y_test,
        numeric,
        categorical
    )


def get_feature_importance(model):
    rf = model.named_steps["model"]

    if not hasattr(rf, "feature_importances_"):
        return pd.DataFrame()

    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        return (
            pd.DataFrame({
                "Feature": names,
                "Importance": rf.feature_importances_
            })
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Header
# -----------------------------
st.title("🚗 Car Selling Price Predictor")
st.caption("Machine-learning application for used-car price prediction")


# -----------------------------
# Sidebar / dataset
# -----------------------------
st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload a CSV",
    type="csv",
    help="Upload a CSV containing the required car-price columns."
)

if uploaded:
    try:
        raw_df = read_uploaded_file(uploaded.getvalue())
        source = uploaded.name
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()
else:
    raw_df = read_default_file()
    source = DEFAULT_CSV

if raw_df is None:
    st.error(
        f"'{DEFAULT_CSV}' was not found. Put it in the same folder as app.py "
        "or upload a CSV from the sidebar."
    )
    st.stop()

valid, message = validate_data(raw_df)

if not valid:
    st.error(message)
    st.stop()

df = prepare_data(raw_df)

if df.empty:
    st.error("No usable rows remain after cleaning.")
    st.stop()


# -----------------------------
# Train
# -----------------------------
with st.spinner("Training models..."):
    (
        models,
        best_model,
        best_name,
        results_df,
        predictions,
        X_test,
        y_test,
        numeric_columns,
        categorical_columns
    ) = train_models(df)

st.sidebar.success("Model trained")
st.sidebar.write(f"**Dataset:** {source}")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Selected model:** {best_name}")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict Price",
    "Dashboard",
    "Model Performance",
    "Dataset"
])


# -----------------------------
# Prediction
# -----------------------------
with tab1:
    st.subheader("Enter Car Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        car_name = st.selectbox(
            "Car Name",
            sorted(df["Car_Name"].astype(str).unique())
        )

        year_min = int(df["Year"].min())
        year_max = int(max(df["Year"].max(), datetime.now().year))

        year = st.number_input(
            "Manufacturing Year",
            min_value=year_min,
            max_value=year_max,
            value=min(2020, year_max),
            step=1
        )

        present_price = st.number_input(
            "Present Price (₹ Lakh)",
            min_value=0.01,
            max_value=float(max(100, df["Present_Price_Lakh"].max() * 1.5)),
            value=float(df["Present_Price_Lakh"].median()),
            step=0.10
        )

        kms = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=int(max(500000, df["Kms_Driven"].max() * 1.5)),
            value=int(df["Kms_Driven"].median()),
            step=1000
        )

    with c2:
        fuel = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].astype(str).unique()))
        seller = st.selectbox("Seller Type", sorted(df["Seller_Type"].astype(str).unique()))
        transmission = st.selectbox(
            "Transmission", sorted(df["Transmission"].astype(str).unique())
        )
        city = st.selectbox("City", sorted(df["City"].astype(str).unique()))
        color = st.selectbox("Color", sorted(df["Color"].astype(str).unique()))

    with c3:
        owner = st.number_input(
            "Previous Owners",
            min_value=int(max(0, df["Owner"].min())),
            max_value=int(max(5, df["Owner"].max())),
            value=int(round(df["Owner"].median())),
            step=1
        )

        engine = st.number_input(
            "Engine CC",
            min_value=1,
            max_value=int(max(6000, df["Engine_CC"].max() * 1.5)),
            value=int(round(df["Engine_CC"].median())),
            step=50
        )

        mileage = st.number_input(
            "Mileage (KMPL)",
            min_value=0.1,
            max_value=float(max(100, df["Mileage_Kmpl"].max() * 1.5)),
            value=float(df["Mileage_Kmpl"].median()),
            step=0.1
        )

        age = datetime.now().year - int(year)
        st.metric("Calculated Car Age", f"{age} years")

    if st.button("Predict Selling Price", type="primary", use_container_width=True):
        new_car = pd.DataFrame({
            "Car_Name": [car_name],
            "Year": [year],
            "Present_Price_Lakh": [present_price],
            "Kms_Driven": [kms],
            "Fuel_Type": [fuel],
            "Seller_Type": [seller],
            "Transmission": [transmission],
            "Owner": [owner],
            "Engine_CC": [engine],
            "Mileage_Kmpl": [mileage],
            "City": [city],
            "Color": [color],
            "Car_Age": [age]
        })

        try:
            price = max(0.0, float(best_model.predict(new_car)[0]))

            st.success(f"Estimated Selling Price: ₹{price:.2f} Lakh")
            a, b, c = st.columns(3)

            a.metric("Estimated Price", f"₹{price:.2f} Lakh")
            b.metric("Present Price", f"₹{present_price:.2f} Lakh")
            c.metric("Difference", f"₹{price - present_price:.2f} Lakh")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------
# Dashboard
# -----------------------------
with tab2:
    st.subheader("Dataset Dashboard")

    a, b, c, d = st.columns(4)
    a.metric("Total Cars", f"{len(df):,}")
    b.metric("Average Price", f"₹{df[TARGET].mean():.2f} Lakh")
    c.metric("Minimum Price", f"₹{df[TARGET].min():.2f} Lakh")
    d.metric("Maximum Price", f"₹{df[TARGET].max():.2f} Lakh")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        ax.hist(df[TARGET], bins=30, edgecolor="black")
        ax.set_title("Selling Price Distribution")
        ax.set_xlabel("Selling Price (₹ Lakh)")
        ax.set_ylabel("Number of Cars")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots()
        ax.scatter(df["Present_Price_Lakh"], df[TARGET], alpha=0.45)
        ax.set_title("Present Price vs Selling Price")
        ax.set_xlabel("Present Price (₹ Lakh)")
        ax.set_ylabel("Selling Price (₹ Lakh)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Fuel Type Summary")
    st.dataframe(
        df.groupby("Fuel_Type")[TARGET]
        .agg(["count", "mean", "min", "max"])
        .round(2),
        use_container_width=True
    )

    st.subheader("City-wise Average Selling Price")
    st.bar_chart(
        df.groupby("City")[TARGET]
        .mean()
        .sort_values(ascending=False)
    )


# -----------------------------
# Model performance
# -----------------------------
with tab3:
    st.subheader("Model Comparison")

    st.dataframe(
        results_df.round(4),
        use_container_width=True
    )

    st.info(
        "Lower MAE/RMSE means smaller prediction errors. "
        "R² closer to 1 means more variation is explained by the model."
    )

    st.success(f"Selected model: {best_name}")

    actual_pred = pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": np.asarray(predictions[best_name])
    })

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual_pred["Actual"], actual_pred["Predicted"], alpha=0.45)

    low = min(actual_pred.min())
    high = max(actual_pred.max())
    ax.plot([low, high], [low, high], linestyle="--")

    ax.set_xlabel("Actual Selling Price (₹ Lakh)")
    ax.set_ylabel("Predicted Selling Price (₹ Lakh)")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
    plt.close(fig)

    importance = get_feature_importance(models["Random Forest Regression"])

    if not importance.empty:
        st.subheader("Random Forest Feature Importance")
        st.dataframe(importance.head(15), use_container_width=True)


# -----------------------------
# Dataset
# -----------------------------
with tab4:
    st.subheader("Dataset Preview")
    st.write(f"Rows after cleaning: **{len(df):,}**")
    st.dataframe(df.head(100), use_container_width=True)

    a, b = st.columns(2)

    with a:
        st.write("### Numerical Columns")
        st.write(numeric_columns)

    with b:
        st.write("### Categorical Columns")
        st.write(categorical_columns)

    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)
