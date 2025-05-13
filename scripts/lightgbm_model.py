import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib

def get_lightgbm_model(random_state: int = 42) -> LGBMClassifier:
    """experiment 스크립트에서 호출할 LightGBM 모델"""
    return LGBMClassifier(random_state=random_state)

if __name__ == "__main__":
    train = pd.read_csv("../train_final_with_selection.csv")
    test  = pd.read_csv("../zerofilled_test_selected.csv")

    drop_cols_train = [c for c in ("id", "y") if c in train.columns]
    drop_cols_test  = [c for c in ("id",)      if c in test.columns]

    X_train = train.drop(columns=drop_cols_train)
    y_train = train["y"]
    X_test  = test.drop(columns=drop_cols_test)

    test_id = test["id"] if "id" in test.columns else pd.Series(
        np.arange(len(test)), name="id"
    )

    model = get_lightgbm_model()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pd.DataFrame({"id": test_id, "y_prob": y_prob, "y_predict": y_pred}) \
        .to_csv("../prediction_lgb.csv", index=False)
    joblib.dump(model, "../lightgbm_model.pkl")

    print("prediction_lgb.csv, lightgbm_model.pkl 저장 완료")