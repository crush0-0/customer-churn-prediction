"""
Customer Churn Prediction Pipeline
A production-ready, self-contained machine learning pipeline for customer
churn prediction using the Telco Customer Churn dataset.

Pipeline Stages:
1. Data Ingestion & Cleaning
2. Quick Preprocessing & Feature Engineering
3. Baseline Model Training (Logistic Regression, Random Forest, XGBoost)
4. Evaluation & Metrics Comparison
5. Visualization Generation (ROC Curves, Confusion Matrices, Feature Importance)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Set non-interactive backend for headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import XGBoost with fallback to GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False


def load_and_clean_data(file_path: str | Path) -> pd.DataFrame:
    """Load dataset, clean columns, and handle missing values."""
    print("=" * 70)
    print("STEP 1: DATA INGESTION & CLEANING")
    print("=" * 70)

    df = pd.read_csv(file_path)
    print(f"[+] Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Drop customerID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("[+] Dropped 'customerID' identifier column.")

    # 2. Convert TotalCharges to numeric and fill NaNs with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"[+] Converted 'TotalCharges' to numeric (filled {nan_count} blank/NaN values with 0).")

    # 3. Map target column 'Churn': 'Yes' -> 1, 'No' -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print("[+] Mapped target 'Churn': 'Yes' -> 1, 'No' -> 0")

    return df


def preprocess_data(df: pd.DataFrame):
    """Inspect dataset, encode categorical features, split train/test, and scale continuous columns."""
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    print(f"[+] Dataset shape after cleaning: {df.shape[0]} rows x {df.shape[1]} columns")

    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    print("[+] Churn Class Distribution:")
    print(f"    - Retained (0): {churn_counts[0]:>5} ({churn_pct[0]:.2f}%)")
    print(f"    - Churned  (1): {churn_counts[1]:>5} ({churn_pct[1]:.2f}%)")

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # One-hot encode categorical features (drop_first=True to avoid collinearity)
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    print(f"[+] One-hot encoding applied: {X.shape[1]} feature columns generated.")

    # 80/20 train-test split stratified on target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] 80/20 Stratified train-test split:")
    print(f"    - Train set: {X_train.shape[0]} samples")
    print(f"    - Test set : {X_test.shape[0]} samples")

    # Scale continuous features
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    print(f"[+] Standardized continuous features {continuous_cols} without leakage.")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train baseline Logistic Regression, Random Forest, and XGBoost models."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)

    models = {}

    # 1. Logistic Regression
    print("[*] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = {"model": lr, "is_tree": False}
    print("    -> Logistic Regression trained.")

    # 2. Random Forest Classifier
    print("[*] Training Random Forest Classifier (n_estimators=100, random_state=42)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = {"model": rf, "is_tree": True}
    print("    -> Random Forest trained.")

    # 3. XGBoost Classifier (or fallback)
    if XGB_AVAILABLE:
        print("[*] Training XGBoost Classifier (n_estimators=100, random_state=42)...")
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        models["XGBoost"] = {"model": xgb, "is_tree": True}
        print("    -> XGBoost trained.")
    else:
        print("[!] XGBoost not available; using GradientBoostingClassifier fallback...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        models["Gradient Boosting (Fallback)"] = {"model": gb, "is_tree": True}
        print("    -> Gradient Boosting trained.")

    return models


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate each model and compile metrics into a comparison table."""
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION & COMPARISON")
    print("=" * 70)

    metrics_list = []

    for name, item in models.items():
        model = item["model"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        item["predictions"] = y_pred
        item["probabilities"] = y_proba
        item["metrics"] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "ROC-AUC": auc,
        }

        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "ROC-AUC": auc,
        })

    results_df = pd.DataFrame(metrics_list)

    # Format table for clean terminal display
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "ROC-AUC"]:
        display_df[col] = display_df[col].apply(lambda v: f"{v * 100:.2f}%")

    print("\n" + "-" * 62)
    print(f"{'Model':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<7} | {'ROC-AUC':<8}")
    print("-" * 62)
    for _, row in display_df.iterrows():
        print(f"{row['Model']:<28} | {row['Accuracy']:<9} | {row['Precision']:<9} | {row['Recall']:<7} | {row['ROC-AUC']:<8}")
    print("-" * 62)

    return results_df


def generate_visualizations(models: dict, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str], output_dir: Path):
    """Generate and save publication-grade evaluation charts."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Overlayed ROC Curves
    roc_path = output_dir / "roc_curves.png"
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#2e7d32", "XGBoost": "#d84315", "Gradient Boosting (Fallback)": "#e65100"}

    for name, item in models.items():
        y_proba = item["probabilities"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = item["metrics"]["ROC-AUC"]
        color = colors.get(name, "#333333")
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, lw=2.2)

    plt.plot([0, 1], [0, 1], color="#757575", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC curves: {roc_path}")

    # 2. Side-by-Side Confusion Matrices
    cm_path = output_dir / "confusion_matrices.png"
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.8), dpi=300)
    if len(models) == 1:
        axes = [axes]

    for ax, (name, item) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, item["predictions"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
        )
        acc = item["metrics"]["Accuracy"] * 100
        rec = item["metrics"]["Recall"] * 100
        ax.set_title(f"{name}\nAcc: {acc:.1f}% | Recall: {rec:.1f}%", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrix Comparison Across Models (Test Set)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrices: {cm_path}")

    # 3. Horizontal Feature Importance Bar Chart (Top 10 from best tree-based model)
    tree_models = {k: v for k, v in models.items() if v.get("is_tree", False)}
    if tree_models:
        best_tree_name, best_tree_item = max(
            tree_models.items(), key=lambda kv: kv[1]["metrics"]["ROC-AUC"]
        )
        best_tree_model = best_tree_item["model"]
        importances = best_tree_model.feature_importances_

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        # Reverse order for horizontal plot so top rank appears at the top
        feat_df = feat_df.iloc[::-1]

        fi_path = output_dir / "feature_importance.png"
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#1976d2", edgecolor="#0d47a1", height=0.65)

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=9.5,
                color="#212121",
                fontweight="bold"
            )

        ax.set_xlim(0, max(feat_df["Importance"]) * 1.15)
        ax.set_xlabel("Relative Feature Importance (Normalized Gain / Weight)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=11, fontweight="bold")
        ax.set_title(f"Top 10 Most Influential Features ({best_tree_name})", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(fi_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved feature importance plot: {fi_path}")

        # Console print top 10
        print(f"\nTop 10 Most Important Features ({best_tree_name}):")
        print("-" * 55)
        print(f"{'Rank':<6} | {'Feature Name':<32} | {'Importance':<10}")
        print("-" * 55)
        for rank, (feat, score) in enumerate(feat_df.iloc[::-1].itertuples(index=False), start=1):
            print(f"#{rank:<5} | {feat:<32} | {score:>9.4f} ({score * 100:.1f}%)")
        print("-" * 55)


def main():
    script_dir = Path(__file__).resolve().parent
    csv_file = script_dir / "Telco-Customer-Churn.csv"
    if not csv_file.exists():
        csv_file = Path("Telco-Customer-Churn.csv")

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset 'Telco-Customer-Churn.csv' not found at {csv_file.resolve()}!"
        )

    output_images_dir = script_dir / "images"

    # Pipeline execution
    df = load_and_clean_data(csv_file)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
    generate_visualizations(models, X_test, y_test, feature_names, output_images_dir)

    print("\n" + "=" * 70)
    print("CUSTOMER CHURN PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Visualizations saved to: {output_images_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
