import json
from pathlib import Path

import numpy as np
import pytest

from src.business import Assumptions, expected_value, policies, realized_value
from src.data import FEATURES, clean, load_raw
from src.model import candidates, split

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def data():
    df = clean(load_raw())
    train, test = split(df)
    model = candidates()["logistic_regression"].fit(train[FEATURES], train["churned"])
    return df, train, test, model.predict_proba(test[FEATURES])[:, 1]


def test_blank_total_charges_are_new_customers_and_kept(data):
    df, *_ = data
    assert len(df) == 7043  # nothing dropped
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_split_is_stratified_and_disjoint(data):
    _, train, test, _ = data
    assert set(train["customerID"]).isdisjoint(test["customerID"])
    assert abs(train["churned"].mean() - test["churned"].mean()) < 0.005


def test_age_and_gender_are_not_model_inputs():
    assert "SeniorCitizen" not in FEATURES and "gender" not in FEATURES


def test_holdout_auc_is_in_the_expected_range(data):
    from sklearn.metrics import roc_auc_score
    *_, test, p = data
    assert 0.82 < roc_auc_score(test["churned"], p) < 0.87


def test_expected_value_matches_realized_value_on_average():
    """If churn really happens with probability p, the average realized value is the expected value."""
    rng = np.random.default_rng(0)
    a = Assumptions()
    p = rng.uniform(0, 1, 200_000)
    monthly = rng.uniform(20, 110, 200_000)
    churned = rng.uniform(0, 1, 200_000) < p
    assert abs(realized_value(churned, monthly, a).mean() - expected_value(p, monthly, a).mean()) < 0.1


def test_blanket_campaign_loses_money_and_targeting_does_not(data):
    *_, test, p = data
    results = {r["policy"]: r["profit"] for r in policies(p, test["churned"].values, test["MonthlyCharges"].values, Assumptions())}
    assert results["Contact everyone"] < 0
    assert results["Positive expected value"] > 0
    assert results["Positive expected value"] > results["Random 20%"]


def test_site_files_are_in_sync_with_the_build():
    model = json.loads((ROOT / "site" / "model.json").read_text())
    assert len(model["features"]) == len(model["coefficients"]) == len(model["baseline"])
    holdout = json.loads((ROOT / "site" / "holdout.json").read_text())
    assert len(holdout["p"]) == len(holdout["churned"]) == 1409
