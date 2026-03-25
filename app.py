import os
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("IEEE-CIS Fraud Detection Dashboard")

st.markdown("""
## Project Introduction

This project uses the IEEE-CIS Kaggle fraud detection dataset to analyze transaction behavior
and build a machine learning system for identifying potentially fraudulent transactions.

The workflow includes:

- merging transaction and identity datasets
- exploring fraud patterns through visual analysis
- engineering predictive features
- training a fraud detection model
- identifying the variables that contribute most to fraud prediction

### Project Objectives
- Understand how fraudulent transactions differ from legitimate ones
- Identify high-risk patterns across amount, product, device, email, and time
- Build a machine learning model for fraud prediction
- Generate business insights for fraud monitoring and risk reduction
""")

st.header("Exploratory Data Analysis")

plot_dir = Path("outputs/plots")

plots = {
    "Fraud vs Legitimate Transactions":
        "This chart shows the class imbalance in the dataset. Fraud cases are much fewer than legitimate transactions, which makes fraud detection a highly imbalanced classification problem.",

    "Transaction Amount Distribution":
        "This plot shows the overall distribution of transaction amounts on a log scale. Most transactions are concentrated at lower amounts, while very large transactions are relatively rare.",

    "Fraud Rate by Product Category":
        "This chart compares fraud rates across product categories. It helps identify which transaction categories appear more vulnerable to fraudulent behavior.",

    "Fraud Rate by Device Type":
        "This visual shows whether fraud is more common on certain device types. It helps assess whether mobile or desktop activity carries different fraud risk.",

    "Top Email Domains Associated with Fraud":
        "This plot highlights the email domains with the highest observed fraud rates. It can reveal domains that appear disproportionately in suspicious transactions.",

    "Transaction Amount Distribution by Fraud Class":
        "This chart compares the amount distributions of legitimate and fraudulent transactions. It helps show whether fraud tends to happen in specific spending ranges.",

    "Fraud Amount Distribution":
        "This boxplot compares the spread of transaction amounts for fraud and non-fraud cases using a log-transformed amount scale. It helps show differences in spending behavior between the two groups.",

    "Fraud Rate by Amount Range":
        "This line chart shows how fraud rate changes across transaction amount quantiles. It helps identify which transaction ranges are relatively riskier.",

    "Fraud Activity Heatmap":
        "This heatmap shows how fraud rates vary by hour and day based on transaction time. It helps identify time-based fraud patterns and suspicious activity windows.",

    "Card Usage Frequency":
        "This chart shows the most frequently appearing card identifiers in the dataset. Repeated usage patterns may sometimes indicate concentration of activity or anomalous behavior.",

    "Model Feature Importance":
        "This chart shows the top features used by the trained model. It helps explain which variables contribute most strongly to fraud prediction."
}

files = {
    "Fraud vs Legitimate Transactions": "class_balance.png",
    "Transaction Amount Distribution": "transaction_amount_distribution.png",
    "Fraud Rate by Product Category": "fraud_by_product.png",
    "Fraud Rate by Device Type": "fraud_by_device.png",
    "Top Email Domains Associated with Fraud": "fraud_by_email.png",
    "Transaction Amount Distribution by Fraud Class": "amount_vs_fraud.png",
    "Fraud Amount Distribution": "fraud_amount_boxplot.png",
    "Fraud Rate by Amount Range": "fraud_rate_by_amount.png",
    "Fraud Activity Heatmap": "fraud_time_heatmap.png",
    "Card Usage Frequency": "card_usage_frequency.png",
    "Model Feature Importance": "model_feature_importance.png"
}

items = list(plots.items())

for i in range(0, len(items), 2):
    cols = st.columns(2)

    for col, (title, description) in zip(cols, items[i:i + 2]):
        path = plot_dir / files[title]

        with col:
            st.subheader(title)
            st.write(description)

            if path.exists():
                image = Image.open(path)
                st.image(image, use_container_width=True)
            else:
                st.warning(f"Graph not found: {files[title]}")

st.header("Project Summary")

st.markdown("""
- The dataset is highly imbalanced, with fraud making up only a small portion of all transactions.
- Transaction amount patterns, device usage, email domains, and time-based behavior provide useful fraud signals.
- Feature engineering improves the model by capturing transaction timing, missingness, and frequency-based patterns.
- The trained model helps estimate fraud probability and identify the factors most associated with suspicious transactions.
""")