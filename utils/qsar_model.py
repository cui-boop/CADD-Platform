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
    将标签列转换为 1 / 0。
    支持以下格式：
    - Active / Inactive
    - active / inactive
    - ACTIVE / INACTIVE
    - 1 / 0
    """
    y = pd.Series(y).copy()

    # 先尝试按数字标签处理，例如 1 / 0
    numeric_y = pd.to_numeric(y, errors="coerce")

    if numeric_y.notna().all():
        numeric_y = numeric_y.astype(int)

        if set(numeric_y.unique()).issubset({0, 1}):
            return numeric_y

    # 再按字符串标签处理，例如 Active / Inactive
    y_str = y.astype(str).str.strip()

    label_map = {
        "Active": 1,
        "active": 1,
        "ACTIVE": 1,
        "1": 1,

        "Inactive": 0,
        "inactive": 0,
        "INACTIVE": 0,
        "0": 0,
    }

    y_encoded = y_str.map(label_map)

    if y_encoded.isna().any():
        bad_labels = sorted(y_str[y_encoded.isna()].unique())
        raise ValueError(
            f"label 列中存在无法识别的标签：{bad_labels}。"
            "请确保标签为 Active / Inactive 或 1 / 0。"
        )

    return y_encoded.astype(int)


def train_random_forest_classifier(
    X,
    y,
    test_size=0.2,
    random_state=42,
    n_estimators=200,
    max_depth=None,
    max_features=0.3,
    min_samples_split=2,
    min_samples_leaf=1
):
    """
    训练随机森林 QSAR 分类模型。

    参数说明：
    - test_size：测试集比例
    - n_estimators：随机森林中决策树数量
    - max_depth：单棵树最大深度，None 表示不限制
    - max_features：每次分裂时参与选择的最大特征比例
    - min_samples_split：节点继续分裂所需的最小样本数
    - min_samples_leaf：叶节点最小样本数
    """
    X = X.copy().fillna(0)
    y_encoded = prepare_label(y)

    if y_encoded.isna().any():
        raise ValueError("label 列中存在无法识别的标签，请确保标签为 Active / Inactive。")

    class_counts = y_encoded.value_counts()
    stratify_y = y_encoded if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
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

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Inactive", "Active"],
        zero_division=0,
        output_dict=True
    )

    return model, metrics, cm, report, X_train, X_test, y_train, y_test, y_pred, y_prob


def save_model(model, feature_names, model_path="models/qsar_random_forest.pkl"):
    """
    保存模型和训练时使用的特征名。
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
    descriptor_df = descriptor_df.copy().fillna(0)

    pred = model.predict(descriptor_df)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(descriptor_df)[0][1]
    else:
        prob = float(pred)

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
