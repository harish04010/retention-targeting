"""Loading and cleaning the IBM Telco customer churn data."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
SOURCE_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]
# SeniorCitizen is deliberately not a model input. It adds about 0.001 AUC,
# and the model decides who gets a discount, so age-based targeting isn't
# worth the risk. It's still used to check that the model treats seniors fairly.
BINARY = ["Partner", "Dependents", "PhoneService", "MultipleLines", "PaperlessBilling", *SERVICES]
NUMERIC = ["tenure", "MonthlyCharges", "services_count"]
FEATURES = CATEGORICAL + BINARY + NUMERIC
GROUP_COLUMNS = ["gender", "SeniorCitizen"]  # used to check fairness, gender is not a model input


def load_raw() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(SOURCE_URL).to_csv(DATA, index=False)
    return pd.read_csv(DATA)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # TotalCharges arrives as text. It's blank for exactly the customers with
    # tenure 0: they signed up and haven't been billed yet. Zero is the true
    # value, so they stay in instead of being dropped.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all(), "blank TotalCharges outside tenure 0"
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # "No internet service" and "No phone service" are just "No" for the
    # add-on itself. Whether they have internet at all lives in InternetService.
    for col in SERVICES + ["MultipleLines"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    for col in BINARY + ["SeniorCitizen"]:
        if df[col].dtype != "int64":
            df[col] = (df[col] == "Yes").astype(int)

    df["services_count"] = df[SERVICES].sum(axis=1) + df["PhoneService"] + (df["InternetService"] != "No").astype(int)
    df["churned"] = (df["Churn"] == "Yes").astype(int)
    df["gender"] = df["gender"].astype(str)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    # TotalCharges is close to tenure x MonthlyCharges, so the model uses
    # those two instead and stays easy to read.
    return df
