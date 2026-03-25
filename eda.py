import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def run_eda(train, test):
    print("\n===== EDA START =====")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    fraud_rate = train["isFraud"].mean()
    print(f"Fraud rate: {fraud_rate:.4f}")

    plot_class_balance(train)
    plot_transaction_amount_distribution(train)
    plot_fraud_by_product(train)
    plot_fraud_by_device(train)
    plot_fraud_by_email(train)
    plot_amount_distribution_by_fraud(train)
    plot_fraud_amount_boxplot(train)
    plot_fraud_rate_by_amount(train)
    plot_time_heatmap(train)
    plot_card_usage(train)

    print("\nEDA plots saved to outputs/plots")
    print("===== EDA END =====\n")


# ---------------------------
# 1 CLASS BALANCE
# ---------------------------

def plot_class_balance(train):
    counts = train["isFraud"].value_counts().sort_index()

    labels = ["Legitimate", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    plt.figure()
    sns.barplot(
        x=labels,
        y=values,
        hue=labels,
        palette=["#2ecc71", "#e74c3c"],
        legend=False
    )

    plt.title("Fraud vs Legitimate Transactions")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_balance.png")
    plt.close()


# ---------------------------
# 2 TRANSACTION AMOUNT DISTRIBUTION
# ---------------------------

def plot_transaction_amount_distribution(train):
    data = train["TransactionAmt"].dropna()

    plt.figure(figsize=(10, 6))
    sns.histplot(np.log1p(data), bins=50, kde=True)

    plt.title("Transaction Amount Distribution (log scale)")
    plt.xlabel("log(1 + TransactionAmt)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "transaction_amount_distribution.png")
    plt.close()


# ---------------------------
# 3 FRAUD BY PRODUCT
# ---------------------------

def plot_fraud_by_product(train):
    if "ProductCD" not in train.columns:
        return

    fraud_product = train.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False)

    plt.figure()
    sns.barplot(
        x=fraud_product.index,
        y=fraud_product.values,
        hue=fraud_product.index,
        palette="viridis",
        legend=False
    )

    plt.title("Fraud Rate by Product Category")
    plt.xlabel("ProductCD")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_by_product.png")
    plt.close()


# ---------------------------
# 4 DEVICE FRAUD
# ---------------------------

def plot_fraud_by_device(train):
    if "DeviceType" not in train.columns:
        return

    data = train[["DeviceType", "isFraud"]].dropna()
    if data.empty:
        return

    fraud_device = data.groupby("DeviceType")["isFraud"].mean().sort_values(ascending=False)

    plt.figure()
    sns.barplot(
        x=fraud_device.index,
        y=fraud_device.values,
        hue=fraud_device.index,
        palette="magma",
        legend=False
    )

    plt.title("Fraud Rate by Device Type")
    plt.xlabel("Device Type")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_by_device.png")
    plt.close()


# ---------------------------
# 5 EMAIL DOMAIN RISK
# ---------------------------

def plot_fraud_by_email(train):
    if "P_emaildomain" not in train.columns:
        return

    data = train[["P_emaildomain", "isFraud"]].dropna()
    if data.empty:
        return

    fraud_email = (
        data.groupby("P_emaildomain")["isFraud"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=fraud_email.values,
        y=fraud_email.index,
        hue=fraud_email.index,
        palette="rocket",
        legend=False
    )

    plt.title("Top Email Domains Associated With Fraud")
    plt.xlabel("Fraud Rate")
    plt.ylabel("Email Domain")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_by_email.png")
    plt.close()


# ---------------------------
# 6 AMOUNT DISTRIBUTION BY FRAUD CLASS
# ---------------------------

def plot_amount_distribution_by_fraud(train):
    data = train[["TransactionAmt", "isFraud"]].dropna().copy()
    if data.empty:
        return

    data["FraudLabel"] = data["isFraud"].map({0: "Legitimate", 1: "Fraud"})
    data["log_amt"] = np.log1p(data["TransactionAmt"])

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=data,
        x="log_amt",
        hue="FraudLabel",
        bins=50,
        stat="density",
        common_norm=False,
        element="step"
    )

    plt.title("Transaction Amount Distribution by Fraud Class")
    plt.xlabel("log(1 + TransactionAmt)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "amount_vs_fraud.png")
    plt.close()


# ---------------------------
# 7 FRAUD AMOUNT BOXPLOT
# ---------------------------

def plot_fraud_amount_boxplot(train):
    data = train[["TransactionAmt", "isFraud"]].dropna().copy()
    if data.empty:
        return

    data["FraudLabel"] = data["isFraud"].map({0: "Legitimate", 1: "Fraud"})
    data["log_amt"] = np.log1p(data["TransactionAmt"])

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=data,
        x="FraudLabel",
        y="log_amt",
        hue="FraudLabel",
        palette="Set2",
        legend=False
    )

    plt.title("Transaction Amount Distribution by Fraud")
    plt.xlabel("")
    plt.ylabel("log(1 + TransactionAmt)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_amount_boxplot.png")
    plt.close()


# ---------------------------
# 8 FRAUD RATE BY AMOUNT
# ---------------------------

def plot_fraud_rate_by_amount(train):
    data = train[["TransactionAmt", "isFraud"]].dropna().copy()
    if data.empty:
        return

    data["amount_bin"] = pd.qcut(
        data["TransactionAmt"],
        20,
        duplicates="drop"
    )

    fraud_rate = data.groupby("amount_bin", observed=False)["isFraud"].mean()

    plt.figure(figsize=(12, 6))
    fraud_rate.plot(marker="o")

    plt.title("Fraud Rate by Transaction Amount Range")
    plt.xlabel("Transaction Amount Quantile")
    plt.ylabel("Fraud Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_rate_by_amount.png")
    plt.close()


# ---------------------------
# 9 TIME HEATMAP
# ---------------------------

def plot_time_heatmap(train):
    if "TransactionDT" not in train.columns:
        return

    temp = train[["TransactionDT", "isFraud"]].dropna().copy()
    if temp.empty:
        return

    dt = temp["TransactionDT"]
    temp["hour"] = ((dt // 3600) % 24).astype(int)
    temp["day"] = (dt // (3600 * 24)).astype(int)

    pivot = temp.pivot_table(
        values="isFraud",
        index="hour",
        columns="day",
        aggfunc="mean"
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="coolwarm")

    plt.title("Fraud Activity Heatmap (Hour vs Day)")
    plt.xlabel("Day")
    plt.ylabel("Hour")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_time_heatmap.png")
    plt.close()


# ---------------------------
# 10 CARD USAGE
# ---------------------------

def plot_card_usage(train):
    if "card1" not in train.columns:
        return

    counts = train["card1"].dropna().value_counts().head(20)
    if counts.empty:
        return

    plt.figure(figsize=(10, 7))
    sns.barplot(
        x=counts.values,
        y=counts.index.astype(str),
        hue=counts.index.astype(str),
        palette="plasma",
        legend=False
    )

    plt.title("Most Frequently Used Cards")
    plt.xlabel("Usage Count")
    plt.ylabel("card1")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "card_usage_frequency.png")
    plt.close()