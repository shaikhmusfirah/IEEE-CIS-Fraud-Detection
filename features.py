import numpy as np
import pandas as pd

from src.config import X_TRAIN, X_TEST, Y_TRAIN


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce RAM usage.
    """
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple new features without combining train and test into one giant dataframe.
    """
    new_cols = {}

    # Time features
    if "TransactionDT" in df.columns:
        dt = df["TransactionDT"].astype("int64")
        new_cols["dt_day"] = (dt // (60 * 60 * 24)).astype("int32")
        new_cols["dt_week"] = (dt // (60 * 60 * 24 * 7)).astype("int32")
        new_cols["dt_hour"] = ((dt // (60 * 60)) % 24).astype("int8")

    # Amount features
    if "TransactionAmt" in df.columns:
        amt = df["TransactionAmt"].astype("float32")
        new_cols["amt_log1p"] = np.log1p(amt).astype("float32")
        new_cols["amt_round"] = np.round(amt, 0).astype("float32")
        new_cols["amt_cents"] = (amt - np.floor(amt)).astype("float32")

    # Email match
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        new_cols["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("int8")

    # Missingness
    na_count = df.isna().sum(axis=1).astype("int16")
    new_cols["na_count"] = na_count
    new_cols["na_frac"] = (na_count / df.shape[1]).astype("float32")

    # Address combo
    if "addr1" in df.columns and "addr2" in df.columns:
        new_cols["addr12"] = (
            df["addr1"].astype("string").fillna("NA")
            + "_"
            + df["addr2"].astype("string").fillna("NA")
        )

    # Card combo
    card_cols = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6"] if c in df.columns]
    if len(card_cols) >= 2:
        card_part = df[card_cols].astype("string").fillna("NA")
        new_cols["card_combo"] = card_part.agg("_".join, axis=1)

    new_features = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_features], axis=1)

    return df


def make_freq_map(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    """
    Build a frequency map from train + test for one column only.
    This is much lighter than concatenating the full dataframes.
    """
    combined = pd.concat([train_series, test_series], axis=0)
    return combined.value_counts(dropna=False)


def add_frequency_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add frequency/count features column by column.
    """
    freq_cols = [
        c for c in
        ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
        if c in train.columns and c in test.columns
    ]

    for col in freq_cols:
        freq_map = make_freq_map(train[col], test[col])
        train[f"{col}_freq"] = train[col].map(freq_map).astype("float32")
        test[f"{col}_freq"] = test[col].map(freq_map).astype("float32")

    count_cols = [c for c in ["card1", "card2", "addr1", "addr2"] if c in train.columns and c in test.columns]
    for col in count_cols:
        freq_map = make_freq_map(train[col], test[col])
        train[f"{col}_count"] = train[col].map(freq_map).astype("float32")
        test[f"{col}_count"] = test[col].map(freq_map).astype("float32")

    for col in ["addr12", "card_combo"]:
        if col in train.columns and col in test.columns:
            freq_map = make_freq_map(train[col], test[col])
            train[f"{col}_count"] = train[col].map(freq_map).astype("float32")
            test[f"{col}_count"] = test[col].map(freq_map).astype("float32")

    return train, test


def drop_very_sparse_columns(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.95):
    """
    Drop columns with too many missing values.
    This reduces memory and often improves model quality.
    """
    missing_frac = train.isna().mean()
    keep_cols = missing_frac[missing_frac < threshold].index.tolist()

    # ensure test only keeps cols that exist in both
    keep_cols = [c for c in keep_cols if c in test.columns]

    return train[keep_cols].copy(), test[keep_cols].copy()


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Main feature engineering pipeline.
    """
    print("Copying input data...")
    train = train.copy()
    test = test.copy()

    print("Separating target...")
    y = train["isFraud"].astype("int8")
    train = train.drop(columns=["isFraud"])

    print("Dropping very sparse columns...")
    train, test = drop_very_sparse_columns(train, test, threshold=0.95)

    print("Adding basic features...")
    train = add_basic_features(train)
    test = add_basic_features(test)

    print("Adding frequency/count features...")
    train, test = add_frequency_features(train, test)

    print("Reducing memory usage...")
    train = reduce_memory(train)
    test = reduce_memory(test)

    print("Selecting numeric columns only...")
    train_numeric = train.select_dtypes(include=[np.number])
    test_numeric = test.select_dtypes(include=[np.number])

    common_cols = [c for c in train_numeric.columns if c in test_numeric.columns]
    X_train = train_numeric[common_cols].copy()
    X_test = test_numeric[common_cols].copy()

    print("Final feature shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("Saving engineered datasets...")
    X_train.to_parquet(X_TRAIN, index=False)
    X_test.to_parquet(X_TEST, index=False)
    y.to_frame("isFraud").to_parquet(Y_TRAIN, index=False)

    return X_train, X_test, y
