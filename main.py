from src.config import PROCESSED_DATA_DIR, PLOTS_DIR, SUBMISSION_DIR
from src.utils import ensure_directories
from src.data_loader import merge_data
from src.eda import run_eda
from src.features import build_features
from src.train import train_model
from src.predict import create_submission


def main():
    ensure_directories([PROCESSED_DATA_DIR, PLOTS_DIR, SUBMISSION_DIR])

    print("Step 1: Merging raw data...")
    train, test = merge_data()

    print("Step 2: Running EDA...")
    run_eda(train, test)

    print("Step 3: Building features...")
    X_train, X_test, y_train = build_features(train, test)

    print("Step 4: Training model...")
    train_model(X_train, y_train)

    print("Step 5: Creating submission...")
    create_submission(X_test)

    print("\nProject pipeline completed successfully.")


if __name__ == "__main__":
    main()