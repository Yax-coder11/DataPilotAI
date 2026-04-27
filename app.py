import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AutoML Trainer", layout="wide")
st.title("🚀 AutoML Interactive Trainer")

# ---------------- SESSION STATE ----------------
if "model" not in st.session_state:
    st.session_state.model = None

if "trained" not in st.session_state:
    st.session_state.trained = False

if "columns" not in st.session_state:
    st.session_state.columns = None

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        if df.empty:
            st.error("Dataset is empty!")
            st.stop()

        # Missing values
        if df.isnull().sum().sum() > 0:
            st.warning("Missing values detected! Filling automatically...")
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)

        columns = df.columns.tolist()

        target = st.selectbox("Select Target Variable (Y)", columns)
        features = st.multiselect("Select Features (X)", columns)

        if target and features:

            if target in features:
                st.error("Target column cannot be in features!")
                st.stop()

            X = df[features]
            y = df[target]

            # Encoding
            X = pd.get_dummies(X)
            st.session_state.columns = X.columns

            # Split
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", 0, 100, 42)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # ---------------- TASK ----------------
            task = st.selectbox("Select Task", ["Classification", "Regression"])

            model = None

            # ================= REGRESSION =================
            if task == "Regression":

                reg_type = st.selectbox("Select Regression Type",
                                        ["Simple Linear", "Multiple Linear", "Polynomial"])

                # Simple Linear
                if reg_type == "Simple Linear":
                    if len(features) != 1:
                        st.warning("Simple Linear requires exactly ONE feature.")
                    model = LinearRegression()

                # Multiple Linear
                elif reg_type == "Multiple Linear":
                    model = LinearRegression()

                # Polynomial
                elif reg_type == "Polynomial":
                    degree = st.slider("Polynomial Degree", 2, 5, 2)

                    model = Pipeline([
                        ("poly", PolynomialFeatures(degree=degree)),
                        ("linreg", LinearRegression())
                    ])

            # ================= CLASSIFICATION =================
            else:
                model_name = st.selectbox("Select Model",
                                         ["KNN", "SVM", "Decision Tree", "Random Forest"])

                if model_name == "KNN":
                    k = st.slider("K Value", 1, 15, 5)
                    model = KNeighborsClassifier(n_neighbors=k)

                elif model_name == "SVM":
                    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                    model = SVC(kernel=kernel)

                elif model_name == "Decision Tree":
                    depth = st.slider("Max Depth", 1, 10, 3)
                    model = DecisionTreeClassifier(max_depth=depth)

                elif model_name == "Random Forest":
                    n = st.slider("Trees", 10, 200, 100)
                    model = RandomForestClassifier(n_estimators=n)

            # ---------------- TRAIN ----------------
            if st.button("Train Model"):
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.session_state.model = model
                    st.session_state.trained = True

                    st.success("Model Trained Successfully!")

                    # -------- REGRESSION --------
                    if task == "Regression":
                        st.subheader("📈 Regression Metrics")
                        st.write("R2 Score:", r2_score(y_test, y_pred))
                        st.write("MSE:", mean_squared_error(y_test, y_pred))

                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred)
                        ax.set_title("Actual vs Predicted")
                        ax.set_xlabel("Actual")
                        ax.set_ylabel("Predicted")
                        st.pyplot(fig)

                    # -------- CLASSIFICATION --------
                    else:
                        st.subheader("📊 Classification Metrics")

                        st.write("Accuracy:", accuracy_score(y_test, y_pred))
                        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
                        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
                        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

                        cm = confusion_matrix(y_test, y_pred)

                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                        ax.set_title("Confusion Matrix")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)

                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))

                        if model_name == "Decision Tree":
                            fig, ax = plt.subplots(figsize=(10, 5))
                            plot_tree(model, filled=True, feature_names=X.columns)
                            ax.set_title("Decision Tree Visualization")
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Training Error: {e}")

            # ---------------- PREDICTION ----------------
            if st.session_state.trained:

                st.subheader("🔮 Make Prediction")

                with st.form("prediction_form"):
                    input_data = []

                    for col in st.session_state.columns:
                        val = st.text_input(f"Enter {col}")
                        input_data.append(val)

                    submit = st.form_submit_button("Predict")

                if submit:
                    try:
                        input_array = np.array(input_data).reshape(1, -1).astype(float)
                        prediction = st.session_state.model.predict(input_array)

                        st.success(f"Prediction: {prediction[0]}")

                    except:
                        st.error("Invalid input! Enter numeric values properly.")

    except Exception as e:
        st.error(f"File Error: {e}")

else:
    st.info("Upload a CSV file to start.")
