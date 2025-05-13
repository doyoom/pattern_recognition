import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
from lightgbm import LGBMClassifier

def get_stacking_model_tuned() -> StackingClassifier:
    """튜닝된 Stacking 모델 반환 (RF + DT + LGB → LR meta)"""
    base_models = [
        ("lr", LogisticRegression(C=0.5, max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ("lgb", LGBMClassifier(
            n_estimators=300, learning_rate=0.03,
            num_leaves=31, max_depth=7, random_state=42))
    ]
    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=0.5),
        cv=5,
        n_jobs=-1
    )

if __name__ == "__main__":
    train = pd.read_csv("../train_final_with_selection.csv")
    test  = pd.read_csv("../zerofilled_test_selected.csv")

    X_train = train.drop(columns=["id", "y"], errors="ignore")
    y_train = train["y"]
    X_test  = test.drop(columns=["id"], errors="ignore")
    test_id = test.get("id", pd.Series(np.arange(len(test)), name="id"))

    model = get_stacking_model_tuned()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pd.DataFrame({"id": test_id, "y_prob": y_prob, "y_predict": y_pred}) \
        .to_csv("../prediction_stack_tuned.csv", index=False)
    joblib.dump(model, "../stacking_model_tuned.pkl")

    print("prediction_stack_tuned.csv, stacking_model_tuned.pkl 저장 완료")