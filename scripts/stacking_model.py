import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier

# 데이터 불러오기
train = pd.read_csv("../data/final_train.csv")
test = pd.read_csv("../data/final_test.csv")
true_test = pd.read_csv("../data/test.csv")  # 교수님이 주신 원본 test 파일 (id 추출용)

# id, y 제거
drop_cols_train = [col for col in ["id", "y"] if col in train.columns]
drop_cols_test = [col for col in ["id"] if col in test.columns]

X = train.drop(columns=drop_cols_train)
y = train["y"]
X_test = test.drop(columns=drop_cols_test)
test_id = true_test["id"]

# 검증용 데이터셋 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Base 모델 정의
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgb', LGBMClassifier(n_estimators=100, random_state=42))
]

# Stacking 모델 정의
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1
)

# 학습
stack_model.fit(X_train, y_train)

# 검증 성능 평가
val_pred = stack_model.predict(X_val)
val_prob = stack_model.predict_proba(X_val)[:, 1]

val_df = pd.DataFrame({
    "Model": ["Stacking"],
    "Accuracy": [accuracy_score(y_val, val_pred)],
    "F1": [f1_score(y_val, val_pred)],
    "AUC": [roc_auc_score(y_val, val_prob)]
})
val_df["Mean"] = val_df[["Accuracy", "F1", "AUC"]].mean(axis=1)

print("Validation Performance:")
print(val_df)

# 교차검증 성능 평가
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "roc_auc"]

cv_scores = cross_validate(stack_model, X, y, cv=cv, scoring=scoring)

cv_df = pd.DataFrame({
    "Model": ["Stacking"],
    "CV_Accuracy": [cv_scores["test_accuracy"].mean()],
    "CV_F1": [cv_scores["test_f1"].mean()],
    "CV_AUC": [cv_scores["test_roc_auc"].mean()]
})
cv_df["Mean"] = cv_df[["CV_Accuracy", "CV_F1", "CV_AUC"]].mean(axis=1)

print("\nCross Validation Performance (5-fold):")
print(cv_df)

# 전체 학습셋으로 재학습
stack_model.fit(X, y)

# 테스트셋 예측
y_prob = stack_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 결과 저장
submission = pd.DataFrame({
    "id": test_id,
    "y_prob": y_prob,
    "y_predict": y_pred
})

os.makedirs("../outputs", exist_ok=True)
submission.to_csv("../outputs/prediction_stack.csv", index=False)
joblib.dump(stack_model, "../outputs/stacking_model.pkl")

print("prediction_stack.csv 저장 완료")
print("stacking_model.pkl 저장 완료")

# --- 최종 요약 결과 출력 ---
print("\n\n최종 요약 결과:")
print("Validation Performance:")
print(val_df)

print("\nCross Validation Performance (5-fold):")
print(cv_df)