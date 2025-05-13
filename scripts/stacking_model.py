import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

def get_stacking_model() -> StackingClassifier:
    """experiment 스크립트에서 호출할 Stacking 모델
    (RF + DT → LR, 5‑fold)"""
    base = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
    )

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

    model = get_stacking_model()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pd.DataFrame({"id": test_id, "y_prob": y_prob, "y_predict": y_pred}) \
        .to_csv("../prediction_stack.csv", index=False)
    joblib.dump(model, "../stacking_model.pkl")

    print("prediction_stack.csv, stacking_model.pkl 저장 완료")