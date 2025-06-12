# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
import optuna

st.set_page_config(layout="wide")
st.title("🔍 Credit Card Fraud Detection with Model Comparison")

# 🔍 Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\User\Downloads\Datasets\creditcard.csv")
    return df

df = load_data()
st.write("### Dataset Preview", df.head())

# 🧹 Data preprocessing
X = df.drop(columns=["Class"])
y = df["Class"]
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))
X = X.drop(columns=["Time"])

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ⚙️ Define evaluation function
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Probabilities": y_prob,
        "Predictions": y_pred,
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# 🤖 Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
}

# 📊 Evaluate models before SMOTE
results_before = [evaluate_model(name, model, X_train, y_train, X_test, y_test)
                  for name, model in models.items()]

# 🧬 Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 📊 Evaluate models after SMOTE
results_after = [evaluate_model(name, model, X_train_sm, y_train_sm, X_test, y_test)
                 for name, model in models.items()]

# 📋 Display metrics table
def build_metrics_table(results_before, results_after):
    df_before = pd.DataFrame([{k: v for k, v in d.items() if k not in ["Probabilities", "Predictions", "Confusion Matrix"]} for d in results_before]).set_index("Model")
    df_after = pd.DataFrame([{k: v for k, v in d.items() if k not in ["Probabilities", "Predictions", "Confusion Matrix"]} for d in results_after]).set_index("Model")
    df_before.columns = [col + " (Before SMOTE)" for col in df_before.columns]
    df_after.columns = [col + " (After SMOTE)" for col in df_after.columns]
    return pd.concat([df_before, df_after], axis=1)

comparison_df = build_metrics_table(results_before, results_after)
st.write("### 📊 Model Performance Comparison", comparison_df)

# 📈 Plot ROC and Precision-Recall curves
def plot_roc_pr_curves(results, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result["Probabilities"])
        axes[0].plot(fpr, tpr, label=result["Model"])
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title(f"ROC Curve {title_suffix}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, result["Probabilities"])
        axes[1].plot(recall, precision, label=result["Model"])
    axes[1].set_title(f"Precision-Recall Curve {title_suffix}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    st.pyplot(fig)

st.subheader("📈 ROC and Precision-Recall Curves")
plot_roc_pr_curves(results_before, title_suffix="(Before SMOTE)")
plot_roc_pr_curves(results_after, title_suffix="(After SMOTE)")

# 🔍 Plot confusion matrices
def plot_confusion_matrices(results, title_suffix=""):
    for result in results:
        cm = result["Confusion Matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix: {result['Model']} {title_suffix}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

st.subheader("🧮 Confusion Matrices Before SMOTE")
plot_confusion_matrices(results_before, title_suffix="(Before SMOTE)")
st.subheader("🧮 Confusion Matrices After SMOTE")
plot_confusion_matrices(results_after, title_suffix="(After SMOTE)")

# 🔧 Optuna Hyperparameter Tuning
@st.cache_data(show_spinner=False)
def run_optuna():
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
        rf = RandomForestClassifier(random_state=42, **params)
        score = cross_val_score(rf, X_train_sm, y_train_sm, cv=2, scoring='f1', n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study

if st.button("Run Optuna Tuning"):
    with st.spinner("Running Optuna Hyperparameter Tuning..."):
        study = run_optuna()
        st.write("### Best Hyperparameters from Optuna")
        st.json(study.best_params)

        best_rf = RandomForestClassifier(random_state=42, **study.best_params)
        optimized_rf_result = evaluate_model("Optimized Random Forest", best_rf, X_train_sm, y_train_sm, X_test, y_test)

        st.write("### Optimized Random Forest Performance")
        for k, v in optimized_rf_result.items():
            if k not in ["Probabilities", "Predictions", "Confusion Matrix"]:
                st.write(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        st.write("### Confusion Matrix of Optimized Model")
        cm = optimized_rf_result["Confusion Matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix: Optimized Random Forest")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
