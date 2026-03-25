import joblib
import pandas as pd

from src.config import MODEL_FILE, SAMPLE_SUBMISSION, SUBMISSION_DIR


def create_submission(X_test: pd.DataFrame) -> None:
    model = joblib.load(MODEL_FILE)
    preds = model.predict_proba(X_test)[:, 1]

    submission = pd.read_csv(SAMPLE_SUBMISSION)
    submission["isFraud"] = preds

    output_file = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(output_file, index=False)

    print(f"Submission file saved to: {output_file}")