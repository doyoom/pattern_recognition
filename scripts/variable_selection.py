import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def drop_outliers(df: pd.DataFrame, drop_path: str) -> pd.DataFrame:
    if os.path.exists(drop_path):
        rows_to_drop = pd.read_csv(drop_path, header=None).squeeze().tolist()
        df = df.drop(index=rows_to_drop)
        print(f"✅ Dropped {len(rows_to_drop)} rows from train")
    else:
        print("⚠️ rows_to_drop.csv not found.")
    return df

def select_features(df: pd.DataFrame) -> list:
    dummy_cols = [col for col in df.columns if col.startswith('data_channel_') or col.startswith('weekday_')]
    numeric_cols = [col for col in df.columns if col not in dummy_cols and col not in {'id', 'shares', 'y'}]

    df_numeric = df[numeric_cols]

    # 1. 분산 기준 완화
    vt = VarianceThreshold(threshold=0.0005)  # 기존 0.001에서 완화
    vt.fit(df_numeric)
    low_var = df_numeric.columns[~vt.get_support()]

    # 2. 상관계수 기준 완화
    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.98)]  # 기존 0.95에서 완화

    # 3. VIF 기준 완화
    vif_vals = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]
    vif_drop = df_numeric.columns[np.array(vif_vals) > 20]  # 기존 15에서 완화

    # 4. 사용자 정의 변수 보호
    protected_vars = {'kw_avg_min', 'global_sentiment_polarity'}
    drop_cols = set(low_var) | set(high_corr) | set(vif_drop) - protected_vars

    # 5. 최종 변수 리스트
    final_numeric = [c for c in numeric_cols if c not in drop_cols]
    final_cols = final_numeric + dummy_cols
    return final_cols


if __name__ == "__main__":
    train_raw = pd.read_csv("data/train_encoded.csv")
    test = pd.read_csv("data/test_encoded.csv")

    # 1. 이상치 제거
    train_cleaned = drop_outliers(train_raw, "data/rows_to_drop.csv")

    # 2. y 분리
    y = train_cleaned['y']
    train_features_only = train_cleaned.drop(columns=['y'])

    # 3. 변수 선택
    selected_cols = select_features(train_cleaned)

    # 4. train: 선택된 컬럼 + y 병합
    X_train_selected = train_features_only[selected_cols]
    train_selected = pd.concat([X_train_selected, y], axis=1)
    train_selected.to_csv("data/train_selected.csv", index=False)

    # 5. test: 선택된 컬럼만 사용
    test_selected = test[[col for col in selected_cols if col in test.columns]]
    test_selected.to_csv("data/test_selected.csv", index=False)

    # 6. 검증
    print("\n🧪 결과 검증")
    print(f"📊 train_selected: {train_selected.shape}")
    print(f"📊 test_selected: {test_selected.shape}")

    if 'y' in test_selected.columns:
        raise ValueError("❌ test_selected에 'y' 컬럼이 포함되어 있음! 제거 필요 ❗")

    mismatch = set(train_selected.columns) - {'y'} != set(test_selected.columns)
    if mismatch:
        print("❌ 컬럼 불일치! y 제외하고도 구조가 다름")
        print("⚠️ train-only 컬럼:", set(train_selected.columns) - set(test_selected.columns) - {'y'})
        print("⚠️ test-only 컬럼:", set(test_selected.columns) - (set(train_selected.columns) - {'y'}))
    else:
        print("✅ 컬럼 구조 일치 (y 제외)")

    # 7. 중요 변수 검증
    print("\n🔍 중요 변수 검증:")
    important_cols = ['kw_max_min', 'global_sentiment_polarity']
    missing_important = [col for col in important_cols if col not in train_selected.columns]
    if missing_important:
        print("❌ 중요 변수 누락됨:", missing_important)
        raise ValueError("⚠️ 중요 변수가 train에 없습니다 — 기준을 다시 조정하세요!")
    else:
        print("✅ 중요 변수 모두 포함됨")
