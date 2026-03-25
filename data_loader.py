import pandas as pd

from src.config import (
    TRAIN_TRANSACTION,
    TRAIN_IDENTITY,
    TEST_TRANSACTION,
    TEST_IDENTITY,
    TRAIN_MERGED,
    TEST_MERGED,
)


def load_raw_data():
    """
    Load the original Kaggle CSV files
    """
    train_tr = pd.read_csv(TRAIN_TRANSACTION)
    train_id = pd.read_csv(TRAIN_IDENTITY)

    test_tr = pd.read_csv(TEST_TRANSACTION)
    test_id = pd.read_csv(TEST_IDENTITY)

    return train_tr, train_id, test_tr, test_id


def merge_data():
    """
    Merge transaction and identity data
    """
    train_tr, train_id, test_tr, test_id = load_raw_data()

    print("Merging training data...")
    train = train_tr.merge(train_id, on="TransactionID", how="left")

    print("Merging test data...")
    test = test_tr.merge(test_id, on="TransactionID", how="left")

    print("Saving merged datasets...")
    train.to_parquet(TRAIN_MERGED, index=False)
    test.to_parquet(TEST_MERGED, index=False)

    return train, test


def load_merged_data():
    """
    Load already merged datasets
    """
    train = pd.read_parquet(TRAIN_MERGED)
    test = pd.read_parquet(TEST_MERGED)

    return train, test