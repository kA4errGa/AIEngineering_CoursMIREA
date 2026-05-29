import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def compute_metrics(y_true, y_pred, y_proba=None):
    methrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is None:
        methrics["roc_auc"] = None
    else:
        try:
            methrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            methrics["roc_auc"] = None
    return methrics

def evaluate_on_test(model, X_test, y_test, label):
    pred = model.predict(X_test)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    methrics = compute_metrics(y_test, pred, proba)
    methrics["model"] = label
    return methrics

def save_metrics_to_csv(metrics_list, filepath):
    # Создаём DataFrame из списка
    df_metrics = pd.DataFrame(metrics_list)
    
    # Сортируем по ROC-AUC (опционально)
    df_metrics = df_metrics.sort_values('f1', ascending=False)
        
    # Сохраняем в CSV
    df_metrics.to_csv(filepath, index=False)

def train_model(): 
    df = pd.read_csv("./data/PROCESSED/LoanDefaultPredictionDatasetPROC.csv")
    X = df.drop("Default", axis=1)
    y = df["Default"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = Pipeline([('classifier',  RandomForestClassifier(random_state=42,
    class_weight="balanced",
    bootstrap=False,
    criterion="gini",
    max_depth=30,
    max_features="sqrt",
    min_samples_leaf=10,
    min_samples_split=37,
    n_estimators=127))])

    model.fit(X_train, y_train)

    artifacts = {
        'model': model,
        'feature_columns': X_train.columns.tolist(),  # порядок признаков
    }

    runTest = []

    runTest.append(evaluate_on_test(model, X_test, y_test, "BestModel"))

    # Сохраняем в один файл
    joblib.dump(artifacts, './artifacts/BestModel/best_model.joblib')

    save_metrics_to_csv(runTest, "./artifacts/BestModel/TestEval.csv")


    print("Модель и артефакты сохранены!")



