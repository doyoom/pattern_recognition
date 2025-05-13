import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib

def get_lightgbm_model_tuned(random_state: int = 42) -> LGBMClassifier:
    """튜닝된 LightGBM 모델 반환"""
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )

if __name__ == "__main__":
    train = pd.read_csv("../train_final_with_selection.csv")
    test  = pd.read_csv("../zerofilled_test_selected.csv")

    X_train = train.drop(columns=["id", "y"], errors="ignore")
    y_train = train["y"]
    X_test  = test.drop(columns=["id"], errors="ignore")
    test_id = test.get("id", pd.Series(np.arange(len(test)), name="id"))

    model = get_lightgbm_model_tuned()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pd.DataFrame({"id": test_id, "y_prob": y_prob, "y_predict": y_pred}) \
        .to_csv("../prediction_lgb_tuned.csv", index=False)
    joblib.dump(model, "../lightgbm_model_tuned.pkl")

    print("prediction_lgb_tuned.csv, lightgbm_model_tuned.pkl 저장 완료")