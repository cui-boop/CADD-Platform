import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def prepare_label(y):
    """
    将 Active / Inactive 转换为 1 / 0。
    """
    return y.map({"Active": 1, "Inactive": 0})


def train_random_forest_classifier(X, y, test_size=0.2, random_state=42, n_estimators=200):
    """
    训练随机森林 QSAR 分类模型。
    """
    y_encoded = prepare_label(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else None,
    }

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Inactive", "Active"],
        zero_division=0,
        output_dict=True
    )

    return model, metrics, cm, report, X_train, X_test, y_train, y_test, y_pred, y_prob


def save_model(model, feature_names, model_path="models/qsar_random_forest.pkl"):
    """
    保存模型和特征名。
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_package = {
        "model": model,
        "feature_names": feature_names
    }

    joblib.dump(model_package, model_path)


def load_model(model_path="models/qsar_random_forest.pkl"):
    """
    加载模型。
    """
    return joblib.load(model_path)


def predict_single_smiles(model, descriptor_df):
    """
    对单个分子的描述符进行活性预测。
    """
    pred = model.predict(descriptor_df)[0]
    prob = model.predict_proba(descriptor_df)[0][1]

    prediction = "Active" if pred == 1 else "Inactive"

    return prediction, prob


def get_feature_importance(model, feature_names):
    """
    输出随机森林特征重要性。
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_df