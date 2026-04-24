import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    r2_score, mean_squared_error,
    classification_report,
    precision_score, recall_score, f1_score
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🚀 AutoML Interactive Trainer")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write(df.head())

    cols = df.columns.tolist()

    X_cols = st.multiselect("Select X", cols)
    y_col = st.selectbox("Select Y", cols)

    if X_cols and y_col:

        X = df[X_cols].copy()
        y = df[y_col].copy()

        # Encoding
        X = pd.get_dummies(X, drop_first=True)

        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state["le"] = le

        # Sidebar settings
        st.sidebar.header("Settings")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.sidebar.number_input("Random State", value=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        task = st.radio("Task", ["Regression", "Classification"])

        # ================= REGRESSION =================
        if task == "Regression":
            model_type = st.selectbox("Model", ["Linear", "Multiple", "Polynomial"])

            if model_type == "Polynomial":
                degree = st.slider("Degree", 2, 5, 2)

            if st.button("Train Model"):

                if model_type in ["Linear", "Multiple"]:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                else:
                    poly = PolynomialFeatures(degree=degree)
                    X_train_p = poly.fit_transform(X_train)
                    X_test_p = poly.transform(X_test)

                    model = LinearRegression()
                    model.fit(X_train_p, y_train)
                    y_pred = model.predict(X_test_p)

                    st.session_state["poly"] = poly

                # Save
                st.session_state["model"] = model
                st.session_state["X_cols"] = X.columns
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred
                st.session_state["trained"] = True

                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                st.metric("R²", f"{r2:.4f}")
                st.metric("MSE", f"{mse:.4f}")

        # ================= CLASSIFICATION =================
        else:
            model_type = st.selectbox(
                "Model",
                ["KNN", "SVM", "Decision Tree", "Random Forest"]
            )

            if model_type == "KNN":
                k = st.slider("K", 1, 15, 5)

            elif model_type == "SVM":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])

            elif model_type == "Decision Tree":
                depth = st.slider("Max Depth", 1, 20, 5)

            elif model_type == "Random Forest":
                trees = st.slider("Trees", 10, 200, 100)

            if st.button("Train Model"):

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                if model_type == "KNN":
                    model = KNeighborsClassifier(n_neighbors=k)
                    model.fit(X_train_s, y_train)

                elif model_type == "SVM":
                    model = SVC(kernel=kernel)
                    model.fit(X_train_s, y_train)

                elif model_type == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=depth)
                    model.fit(X_train, y_train)

                else:
                    model = RandomForestClassifier(n_estimators=trees)
                    model.fit(X_train, y_train)

                # Predict
                if model_type in ["KNN", "SVM"]:
                    y_pred = model.predict(X_test_s)
                else:
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                # Save
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["X_cols"] = X.columns
                st.session_state["model_type"] = model_type
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred
                st.session_state["trained"] = True

                st.metric("Accuracy", f"{acc:.4f}")

                if acc == 1:
                    st.warning("⚠️ Overfitting")

                # Extra metrics
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                cm = confusion_matrix(y_test, y_pred)

                tn = cm[0][0]
                fp = cm[0][1]
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

                error = 1 - acc

                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall: {recall:.4f}")
                st.write(f"F1 Score: {f1:.4f}")
                st.write(f"Specificity: {specificity:.4f}")
                st.write(f"Error: {error:.4f}")

                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(4,3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # Decision Tree Plot
                if model_type == "Decision Tree":
                    fig, ax = plt.subplots(figsize=(6,4))
                    plot_tree(model, filled=True, feature_names=X.columns)
                    st.pyplot(fig)

        # ================= SHOW RESULTS =================
        if st.session_state.get("trained"):

            st.subheader("📊 Scatter Plot (Actual vs Predicted)")

            fig, ax = plt.subplots(figsize=(4,3))
            ax.scatter(st.session_state["y_test"], st.session_state["y_pred"])
            st.pyplot(fig)

        # ================= PREDICTION =================
        if "model" in st.session_state:

            st.subheader("🔮 Predict")

            with st.form("form"):
                data = {}
                for col in X_cols:
                    data[col] = st.text_input(col)

                submit = st.form_submit_button("Predict")

                if submit:
                    inp = pd.DataFrame([data])
                    inp = pd.get_dummies(inp)
                    inp = inp.reindex(columns=st.session_state["X_cols"], fill_value=0)

                    if "poly" in st.session_state:
                        inp = st.session_state["poly"].transform(inp)

                    if st.session_state.get("model_type") in ["KNN", "SVM"]:
                        inp = st.session_state["scaler"].transform(inp)

                    pred = st.session_state["model"].predict(inp)

                    if "le" in st.session_state:
                        pred = st.session_state["le"].inverse_transform(pred)

                    st.success(f"Prediction: {pred[0]}")