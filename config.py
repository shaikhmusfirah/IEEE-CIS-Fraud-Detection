from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
SUBMISSION_DIR = BASE_DIR / "outputs" / "submission"

TRAIN_TRANSACTION = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY = RAW_DATA_DIR / "test_identity.csv"
SAMPLE_SUBMISSION = RAW_DATA_DIR / "sample_submission.csv"

TRAIN_MERGED = PROCESSED_DATA_DIR / "train_merged.parquet"
TEST_MERGED = PROCESSED_DATA_DIR / "test_merged.parquet"
X_TRAIN = PROCESSED_DATA_DIR / "X_train.parquet"
X_TEST = PROCESSED_DATA_DIR / "X_test.parquet"
Y_TRAIN = PROCESSED_DATA_DIR / "y_train.parquet"
MODEL_FILE = PROCESSED_DATA_DIR / "fraud_model.joblib"