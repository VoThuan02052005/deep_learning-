import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_features(
    data,
    target_col="price",
    test_size=0.2,
    random_state=42
):

    # =========================
    # 1. Train / Test Split
    # =========================
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    # =========================
    # 2. Time Features
    # =========================
    TIME_COLS = ['posted_date', 'crawl_date']
    TIME_COLS = [c for c in TIME_COLS if c in train_df.columns]

    for col in TIME_COLS:
        train_df[col] = pd.to_datetime(train_df[col], errors='coerce')
        test_df[col]  = pd.to_datetime(test_df[col], errors='coerce')

    for df in [train_df, test_df]:
        df['posted_year'] = df['posted_date'].dt.year
        df['posted_month'] = df['posted_date'].dt.month
        df['posted_wday'] = df['posted_date'].dt.weekday
        df['days_on_market'] = (df['crawl_date'] - df['posted_date']).dt.days

        # Cyclic encoding (BEST PRACTICE)
        df['month_sin'] = np.sin(2 * np.pi * df['posted_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['posted_month'] / 12)
        df['wday_sin']  = np.sin(2 * np.pi * df['posted_wday'] / 7)
        df['wday_cos']  = np.cos(2 * np.pi * df['posted_wday'] / 7)

        # Log-transform skewed time feature
        df['days_on_market'] = np.log1p(df['days_on_market'])

    time_feats = [
        'posted_year', 'days_on_market',
        'month_sin', 'month_cos',
        'wday_sin', 'wday_cos'
    ]

    train_df[time_feats] = train_df[time_feats].fillna(-1)
    test_df[time_feats]  = test_df[time_feats].fillna(-1)

    train_df.drop(columns=TIME_COLS + ['posted_month', 'posted_wday'], inplace=True)
    test_df.drop(columns=TIME_COLS + ['posted_month', 'posted_wday'], inplace=True)

    # =========================
    # 3. Categorical Features
    # =========================
    CAT_COLS = [
        'city', 'district',
        'property_type',
        'transaction_type',
        'legal_status'
    ]
    CAT_COLS = [c for c in CAT_COLS if c in train_df.columns]

    train_df[CAT_COLS] = train_df[CAT_COLS].fillna("Unknown").astype(str)
    test_df[CAT_COLS]  = test_df[CAT_COLS].fillna("Unknown").astype(str)

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = ohe.fit_transform(train_df[CAT_COLS])
    X_test_cat  = ohe.transform(test_df[CAT_COLS])

    cat_names = ohe.get_feature_names_out(CAT_COLS)

    X_train_cat = pd.DataFrame(X_train_cat, columns=cat_names, index=train_df.index)
    X_test_cat  = pd.DataFrame(X_test_cat, columns=cat_names, index=test_df.index)

    # =========================
    # 4. Numerical Features
    # =========================
    NUM_COLS = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    NUM_COLS.remove(target_col)

    # Log-transform numerical skewed features (example)
    SKEWED_COLS = [c for c in NUM_COLS if c not in time_feats]

    for col in SKEWED_COLS:
        train_df[col] = np.log1p(train_df[col])
        test_df[col]  = np.log1p(test_df[col])

    X_train_num = train_df[NUM_COLS]
    X_test_num  = test_df[NUM_COLS]

    # =========================
    # 5. Scaling (ONLY numerical)
    # =========================
    scaler_X = StandardScaler()

    X_train_num_scaled = scaler_X.fit_transform(X_train_num)
    X_test_num_scaled  = scaler_X.transform(X_test_num)

    X_train_num_scaled = pd.DataFrame(
        X_train_num_scaled, columns=NUM_COLS, index=train_df.index
    )
    X_test_num_scaled = pd.DataFrame(
        X_test_num_scaled, columns=NUM_COLS, index=test_df.index
    )

    # =========================
    # 6. Final Feature Matrix
    # =========================
    X_train = pd.concat([X_train_num_scaled, X_train_cat], axis=1).astype(np.float64)
    X_test  = pd.concat([X_test_num_scaled, X_test_cat], axis=1).astype(np.float64)

    # =========================
    # 7. Target Normalization (BEST PRACTICE)
    # =========================
    y_scaler = StandardScaler()

    y_train_log = np.log1p(train_df[target_col].values).reshape(-1, 1)
    y_test_log  = np.log1p(test_df[target_col].values).reshape(-1, 1)

    y_train = y_scaler.fit_transform(y_train_log).ravel()
    y_test  = y_scaler.transform(y_test_log).ravel()

    # =========================
    # 8. Return
    # =========================
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "scaler_y": y_scaler,
        "ohe": ohe,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS
    }
