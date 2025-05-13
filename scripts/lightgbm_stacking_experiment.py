import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from lightgbm_model import get_lightgbm_model
from stacking_model import get_stacking_model

train = pd.read_csv("../train_final_with_selection.csv")
X = train.drop(columns=["y"])
y = train["y"]

models = {
    "LightGBM": get_lightgbm_model(random_state=42),
    "Stacking": get_stacking_model(),
}

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

val_results, cv_results = {}, {}
for name, model in models.items():
    model.fit(X_tr, y_tr)

    # ── validation
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    val_results[name] = [
        accuracy_score(y_val, y_pred),
        f1_score(y_val, y_pred),
        roc_auc_score(y_val, y_prob),
    ]

    # ── 5‑fold CV
    scores = cross_validate(
        model, X, y, cv=5,
        scoring=["accuracy", "f1", "roc_auc"]
    )
    cv_results[name] = [
        scores["test_accuracy"].mean(),
        scores["test_f1"].mean(),
        scores["test_roc_auc"].mean(),
    ]

print("Model Performance on Validation Set:")
print(f"{'Model':<10} {'Accuracy':>10} {'F1-score':>10} {'AUC':>10} {'MeanScore':>12}")
for n, (acc, f1, auc) in val_results.items():
    mean = (acc + f1 + auc) / 3
    print(f"{n:<10} {acc:10.6f} {f1:10.6f} {auc:10.6f} {mean:12.6f}")

print("\nCross-Validation Results (5-fold):")
print(f"{'Model':<10} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'MeanScore':>12}")
for n, (cv_acc, cv_f1, cv_auc) in cv_results.items():
    cv_mean = (cv_acc + cv_f1 + cv_auc) / 3
    print(f"{n:<10} {cv_acc:10.6f} {cv_f1:10.6f} {cv_auc:10.6f} {cv_mean:12.6f}")

best_model_name = max(cv_results.items(),
                      key=lambda x: sum(x[1]) / 3)[0]
print(f"\nBest model: {best_model_name}")

final_model = models[best_model_name].fit(X, y)
test = pd.read_csv("../zerofilled_test_selected.csv")
X_test = test[X.columns]

pred = final_model.predict(X_test)
prob = final_model.predict_proba(X_test)[:, 1]

pd.DataFrame({"y_predict": pred, "y_prob": prob}) \
    .to_csv("../prediction_ls.csv", index=False)
print("prediction_ls.csv 저장 완료")