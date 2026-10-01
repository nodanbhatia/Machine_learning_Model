
import io
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Car Selling Price Predictor",
    page_icon="https://img.icons8.com/?size=100&id=Y0Gmn6ZRBPfz&format=png&color=000000",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET = "Selling_Price_Lakh"
ID_COLUMN = "Car_ID"

DEFAULT_CSV = "car_price_prediction_10000.csv"

REQUIRED_COLUMNS = [
    "Car_Name",
    "Year",
    "Present_Price_Lakh",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
    "Engine_CC",
    "Mileage_Kmpl",
    "City",
    "Color",
    TARGET,
]

OPTIONAL_COLUMNS = ["Car_Age"]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .price-box {
        padding: 22px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1f77b4, #4aa3df);
        color: white;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .price-value {
        font-size: 38px;
        font-weight: 800;
    }

    .small-note {
        color: #666;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def load_default_csv():
    if not os.path.exists(DEFAULT_CSV):
        return None

    try:
        return pd.read_csv(DEFAULT_CSV)
    except Exception as exc:
        st.error(f"Could not read {DEFAULT_CSV}: {exc}")
        return None


# ============================================================
# DATA VALIDATION + FEATURE ENGINEERING
# ============================================================

def validate_dataset(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        return False, (
            "The uploaded CSV is missing these required columns: "
            + ", ".join(missing)
        )

    return True, "Dataset structure is valid."


def prepare_dataset(df):
    df = df.copy()

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove ID because it is only an identifier.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Make sure target is numeric.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Convert numeric columns where applicable.
    numeric_candidates = [
        "Year",
        "Present_Price_Lakh",
        "Kms_Driven",
        "Owner",
        "Engine_CC",
        "Mileage_Kmpl",
        TARGET,
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use Car_Age if supplied; otherwise calculate it.
    current_year = datetime.now().year

    if "Car_Age" not in df.columns:
        df["Car_Age"] = current_year - df["Year"]
    else:
        df["Car_Age"] = pd.to_numeric(df["Car_Age"], errors="coerce")

    # Remove impossible ages.
    df.loc[df["Car_Age"] < 0, "Car_Age"] = np.nan

    # Drop rows with missing values after conversion.
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# MODEL BUILDING
# ============================================================

def create_preprocessor(X):
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, numerical_columns, categorical_columns


def build_models(X):
    preprocessor, numerical_columns, categorical_columns = create_preprocessor(X)

    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    max_features=1.0,
                ),
            ),
        ]
    )

    return (
        {
            "Linear Regression": linear_model,
            "Random Forest Regression": rf_model,
        },
        numerical_columns,
        categorical_columns,
    )


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models, numerical_columns, categorical_columns = build_models(X)

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2,
            }
        )

        predictions[name] = pred

    results_df = pd.DataFrame(results).sort_values(
        by="R2 Score",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]
    best_model = models[best_model_name]

    return (
        models,
        best_model,
        best_model_name,
        results_df,
        predictions,
        X_test,
        y_test,
        numerical_columns,
        categorical_columns,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model):
    if "model" not in model.named_steps:
        return pd.DataFrame()

    fitted_model = model.named_steps["model"]

    if not hasattr(fitted_model, "feature_importances_"):
        return pd.DataFrame()

    preprocessor = model.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = fitted_model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        )

        return importance_df.sort_values(
            by="Importance",
            ascending=False,
        ).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-title">Car Selling Price Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "End-to-end machine learning application for used-car price prediction"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR - DATASET
# ============================================================

st.sidebar.header(" Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your car-price CSV",
    type=["csv"],
    help=(
        "Upload car_price_prediction_10000.csv or another CSV "
        "with the required columns."
    ),
)

if uploaded_file is not None:
    try:
        raw_df = load_csv_from_bytes(uploaded_file.getvalue())
        dataset_source = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    raw_df = load_default_csv()
    dataset_source = DEFAULT_CSV


if raw_df is None:
    st.error(
        "Dataset not found. Put "
        f"'{DEFAULT_CSV}' in the same folder as this Python file "
        "or upload a CSV from the sidebar."
    )

    st.info(
        "Required target column: Selling_Price_Lakh"
    )

    st.stop()


valid, validation_message = validate_dataset(raw_df)

if not valid:
    st.error(validation_message)
    st.stop()


df = prepare_dataset(raw_df)

if df.empty:
    st.error("No usable rows remain after data cleaning.")
    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner("Training machine-learning models..."):
    (
        models,
        best_model,
        best_model_name,
        results_df,
        predictions,
        X_test,
        y_test,
        numerical_columns,
        categorical_columns,
    ) = train_models(df)


# ============================================================
# SIDEBAR INFO
# ============================================================

st.sidebar.success("Model trained successfully")

st.sidebar.write(
    f"**Dataset:** {dataset_source}"
)

st.sidebar.write(
    f"**Rows:** {len(df):,}"
)

st.sidebar.write(
    f"**Features:** {len(df.columns) - 1}"
)

st.sidebar.write(
    f"**Selected model:** {best_model_name}"
)


# ============================================================
# TABS
# ============================================================

tab_predict, tab_dashboard, tab_models, tab_data = st.tabs(
    [
        " Predict Price",
        " Dashboard",
        " Model Performance",
        " Dataset",
    ]
)


# ============================================================
# TAB 1 - PREDICTION
# ============================================================

with tab_predict:

    st.subheader("Enter New Car Details")

    col1, col2, col3 = st.columns(3)

    # ---------- CAR NAME ----------
    with col1:
        car_names = sorted(
            df["Car_Name"].astype(str).unique().tolist()
        )

        car_name = st.selectbox(
            "Car Name",
            car_names,
        )

        year_min = int(df["Year"].min())
        year_max = int(max(df["Year"].max(), datetime.now().year))

        year = st.number_input(
            "Manufacturing Year",
            min_value=year_min,
            max_value=year_max,
            value=min(2020, year_max),
            step=1,
        )

        present_price = st.number_input(
            "Present Price (₹ Lakh)",
            min_value=0.01,
            max_value=float(max(100, df["Present_Price_Lakh"].max() * 1.5)),
            value=float(df["Present_Price_Lakh"].median()),
            step=0.10,
        )

        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=int(max(500000, df["Kms_Driven"].max() * 1.5)),
            value=int(df["Kms_Driven"].median()),
            step=1000,
        )

    # ---------- CATEGORIES ----------
    with col2:
        fuel_options = sorted(
            df["Fuel_Type"].astype(str).unique().tolist()
        )

        seller_options = sorted(
            df["Seller_Type"].astype(str).unique().tolist()
        )

        transmission_options = sorted(
            df["Transmission"].astype(str).unique().tolist()
        )

        city_options = sorted(
            df["City"].astype(str).unique().tolist()
        )

        color_options = sorted(
            df["Color"].astype(str).unique().tolist()
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_options,
        )

        seller_type = st.selectbox(
            "Seller Type",
            seller_options,
        )

        transmission = st.selectbox(
            "Transmission",
            transmission_options,
        )

        city = st.selectbox(
            "City",
            city_options,
        )

        color = st.selectbox(
            "Color",
            color_options,
        )

    # ---------- ENGINE ----------
    with col3:

        owner_min = int(max(0, df["Owner"].min()))
        owner_max = int(max(5, df["Owner"].max()))

        owner = st.number_input(
            "Previous Owners",
            min_value=owner_min,
            max_value=owner_max,
            value=int(round(df["Owner"].median())),
            step=1,
        )

        engine_cc = st.number_input(
            "Engine CC",
            min_value=1,
            max_value=int(max(6000, df["Engine_CC"].max() * 1.5)),
            value=int(round(df["Engine_CC"].median())),
            step=50,
        )

        mileage = st.number_input(
            "Mileage (KMPL)",
            min_value=0.1,
            max_value=float(max(100, df["Mileage_Kmpl"].max() * 1.5)),
            value=float(df["Mileage_Kmpl"].median()),
            step=0.1,
        )

        car_age = datetime.now().year - int(year)

        st.metric(
            "Calculated Car Age",
            f"{car_age} years",
        )

    st.divider()

    predict_button = st.button(
        " Predict Selling Price",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        new_car = pd.DataFrame(
            {
                "Car_Name": [car_name],
                "Year": [year],
                "Present_Price_Lakh": [present_price],
                "Kms_Driven": [kms_driven],
                "Fuel_Type": [fuel_type],
                "Seller_Type": [seller_type],
                "Transmission": [transmission],
                "Owner": [owner],
                "Engine_CC": [engine_cc],
                "Mileage_Kmpl": [mileage],
                "City": [city],
                "Color": [color],
                "Car_Age": [car_age],
            }
        )

        try:
            predicted_price = float(
                best_model.predict(new_car)[0]
            )

            predicted_price = max(0.0, predicted_price)

            st.markdown(
                f"""
                <div class="price-box">
                    <div>Estimated Selling Price</div>
                    <div class="price-value">
                        ₹ {predicted_price:.2f} Lakh
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success(
                f"Prediction generated using {best_model_name}."
            )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Estimated Price",
                    f"₹{predicted_price:.2f} Lakh",
                )

            with result_col2:
                st.metric(
                    "Present Price",
                    f"₹{present_price:.2f} Lakh",
                )

            with result_col3:
                difference = predicted_price - present_price

                st.metric(
                    "Predicted vs Present",
                    f"₹{difference:.2f} Lakh",
                )

            with st.expander("View input sent to model"):
                st.dataframe(
                    new_car,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )


# ============================================================
# TAB 2 - DASHBOARD
# ============================================================

with tab_dashboard:

    st.subheader("Dataset & Price Dashboard")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Total Cars",
            f"{len(df):,}",
        )

    with metric2:
        st.metric(
            "Average Selling Price",
            f"₹{df[TARGET].mean():.2f} Lakh",
        )

    with metric3:
        st.metric(
            "Minimum Selling Price",
            f"₹{df[TARGET].min():.2f} Lakh",
        )

    with metric4:
        st.metric(
            "Maximum Selling Price",
            f"₹{df[TARGET].max():.2f} Lakh",
        )

    st.divider()

    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            df[TARGET],
            bins=30,
            edgecolor="black",
        )

        ax.set_title("Selling Price Distribution")
        ax.set_xlabel("Selling Price (₹ Lakh)")
        ax.set_ylabel("Number of Cars")

        st.pyplot(fig)
        plt.close(fig)

    with chart2:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            df["Present_Price_Lakh"],
            df[TARGET],
            alpha=0.45,
        )

        ax.set_title(
            "Present Price vs Selling Price"
        )

        ax.set_xlabel(
            "Present Price (₹ Lakh)"
        )

        ax.set_ylabel(
            "Selling Price (₹ Lakh)"
        )

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Fuel Type vs Selling Price")

    fuel_summary = (
        df.groupby("Fuel_Type")[TARGET]
        .agg(["count", "mean", "min", "max"])
        .round(2)
    )

    st.dataframe(
        fuel_summary,
        use_container_width=True,
    )

    st.subheader("City-wise Average Selling Price")

    city_summary = (
        df.groupby("City")[TARGET]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )

    st.bar_chart(city_summary)


# ============================================================
# TAB 3 - MODEL PERFORMANCE
# ============================================================

with tab_models:

    st.subheader("Model Comparison")

    display_results = results_df.copy()

    for column in ["MAE", "MSE", "RMSE", "R2 Score"]:
        display_results[column] = display_results[column].round(4)

    st.dataframe(
        display_results,
        use_container_width=True,
    )

    st.info(
        "For MAE, MSE and RMSE, lower values indicate smaller prediction "
        "errors. For R², a value closer to 1 indicates that the model "
        "explains more of the variation in the target."
    )

    st.success(
        f"Selected model based on test-set R²: {best_model_name}"
    )

    best_pred = predictions[best_model_name]

    actual_vs_pred = pd.DataFrame(
        {
            "Actual": np.asarray(y_test),
            "Predicted": np.asarray(best_pred),
        }
    )

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        actual_vs_pred["Actual"],
        actual_vs_pred["Predicted"],
        alpha=0.45,
    )

    min_value = min(
        actual_vs_pred["Actual"].min(),
        actual_vs_pred["Predicted"].min(),
    )

    max_value = max(
        actual_vs_pred["Actual"].max(),
        actual_vs_pred["Predicted"].max(),
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    ax.set_xlabel("Actual Selling Price (₹ Lakh)")
    ax.set_ylabel("Predicted Selling Price (₹ Lakh)")
    ax.set_title("Actual vs Predicted Car Prices")

    st.pyplot(fig)
    plt.close(fig)

    # Feature importance is available for Random Forest.
    rf_importance = get_feature_importance(
        models["Random Forest Regression"]
    )

    if not rf_importance.empty:
        st.subheader(
            "Random Forest Feature Importance"
        )

        top_features = rf_importance.head(15)

        st.dataframe(
            top_features,
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1],
        )

        ax.set_title(
            "Top 15 Random Forest Features"
        )

        ax.set_xlabel("Importance")

        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# TAB 4 - DATASET
# ============================================================

with tab_data:

    st.subheader("Dataset Preview")

    st.write(
        f"Rows after cleaning: **{len(df):,}**"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("### Numerical Columns")
        st.write(numerical_columns)

    with info_col2:
        st.write("### Categorical Columns")
        st.write(categorical_columns)

    st.subheader("Missing Values")

    missing_df = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values",
    ]

    st.dataframe(
        missing_df,
        use_container_width=True,
    )


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

st.sidebar.divider()

if st.sidebar.button(
    " Save Trained Model"
):
    model_path = "car_price_prediction_model.pkl"

    joblib.dump(
        best_model,
        model_path,
    )

    st.sidebar.success(
        f"Saved as {model_path}"
    )


st.sidebar.divider()

st.sidebar.caption(
    "Car Selling Price Prediction | "
    "Python + Scikit-learn + Streamlit"
)
