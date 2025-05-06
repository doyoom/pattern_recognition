import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib

# 학습/테스트 데이터 불러오기
train = pd.read_csv("../data/final_train.csv")
test = pd.read_csv("../data/final_test.csv")
true_test = pd.read_csv("../data/test.csv")

# 안전하게 id, y 컬럼 제거
X_train = train.drop(columns=[col for col in ["id", "y"] if col in train.columns])
y_train = train["y"]
X_test = test.drop(columns=[col for col in ["id"] if col in test.columns])

# 교수님 제공 test.csv 기준으로 id 사용
test_id = true_test["id"]

# LightGBM 모델 학습 및 예측
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 저장
submission = pd.DataFrame({
    "id": test_id,
    "y_prob": y_prob,
    "y_predict": y_pred
})
submission.to_csv("../prediction_lgb.csv", index=False)
joblib.dump(model, "../lightgbm_model.pkl")

print("prediction_lgb.csv 저장 완료")
print("lightgbm_model.pkl 저장 완료")