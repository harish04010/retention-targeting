"""
Trains, evaluates, and exports everything: metrics, figures, and the model
file the browser simulator runs.

    python -m src.build
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from .business import Assumptions, expected_value, policies, profit_curve, realized_value, sensitivity
from .data import BINARY, CATEGORICAL, FEATURES, NUMERIC, SOURCE_URL, clean, load_raw
from .model import SEED, calibration_bins, candidates, compare, holdout_metrics, logistic, split

ROOT = Path(__file__).resolve().parent.parent
SITE, DOCS, RESULTS, TESTS = ROOT / "site", ROOT / "docs", ROOT / "results", ROOT / "tests"

# Plain-language names for the model's inputs, used in the simulator's explanations.
LABELS = {
    "Contract": "contract type", "InternetService": "internet type", "PaymentMethod": "payment method",
"Partner": "has a partner", "Dependents": "has dependents",
    "PhoneService": "phone service", "MultipleLines": "multiple lines", "PaperlessBilling": "paperless billing",
    "OnlineSecurity": "online security add-on", "OnlineBackup": "online backup add-on",
    "DeviceProtection": "device protection add-on", "TechSupport": "tech support add-on",
    "StreamingTV": "streaming TV", "StreamingMovies": "streaming movies",
    "tenure": "months as a customer", "MonthlyCharges": "monthly bill", "services_count": "number of services",
}


def bootstrap_difference(p, y, monthly, a, n=2000, seed=SEED):
    """How sure are we that one policy beats the other, given only 1,409 holdout customers?"""
    rng = np.random.default_rng(seed)
    realized = realized_value(y, monthly, a)
    ev_mask = expected_value(p, monthly, a) > 0
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(p), len(p))
        pp, rr = p[idx], realized[idx]
        top = np.zeros(len(idx), bool)
        top[np.argsort(-pp)[: int(0.2 * len(idx))]] = True
        diffs.append(rr[ev_mask[idx]].sum() - rr[top].sum())
    diffs = np.array(diffs)
    return {"mean": float(diffs.mean()), "low": float(np.percentile(diffs, 2.5)),
            "high": float(np.percentile(diffs, 97.5)), "share_ev_better": float((diffs > 0).mean())}


def fairness(test, p, a):
    ev_mask = expected_value(p, test["MonthlyCharges"].values, a) > 0
    rows = []
    for col, names in (("gender", None), ("SeniorCitizen", {0: "Under 65", 1: "Senior"})):
        for value, g in test.groupby(col):
            m = test.index.isin(g.index)
            rows.append({"group": col if col != "SeniorCitizen" else "age",
                         "value": names[value] if names else value, "customers": int(m.sum()),
                         "actual_churn": float(test.loc[m, "churned"].mean()),
                         "predicted_churn": float(p[m].mean()),
                         "roc_auc": float(roc_auc_score(test.loc[m, "churned"], p[m])),
                         "share_offered": float(ev_mask[m].mean())})
    return rows


def export_model(pipe, train) -> dict:
    """Everything the browser needs to reproduce predict_proba exactly."""
    prep, lr = pipe.named_steps["prep"], pipe.named_steps["model"]
    enc = prep.named_transformers_["cat"]
    scaler = prep.named_transformers_["num"]
    names = list(prep.get_feature_names_out())
    coef = lr.coef_[0]
    encoded_train = prep.transform(train[FEATURES])
    return {
        "categorical": {col: {"levels": [str(c) for c in cats], "dropped": str(cats[0])}
                        for col, cats in zip(CATEGORICAL, enc.categories_)},
        "binary": BINARY,
        "numeric": {col: {"mean": float(m), "scale": float(s)} for col, m, s in zip(NUMERIC, scaler.mean_, scaler.scale_)},
        "features": names,
        "coefficients": [float(c) for c in coef],
        "intercept": float(lr.intercept_[0]),
        "baseline": [float(v) for v in encoded_train.mean(axis=0)],  # the "average customer", for explanations
        "labels": LABELS,
    }


def figures(cal, curve, model_json, sens):
    DOCS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    pine, ink = "#2f5d50", "#1f262e"

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="#9aa5ae", lw=1, label="Perfect calibration")
    ax.plot([c["predicted"] for c in cal], [c["actual"] for c in cal], "o-", color=pine, lw=2, label="Model, by decile")
    ax.set(xlabel="Predicted churn probability", ylabel="Actual churn rate", xlim=(0, 1), ylim=(0, 1),
           title="Calibration on the holdout set")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(DOCS / "calibration.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="#9aa5ae", lw=1)
    ax.plot([c["share"] * 100 for c in curve], [c["profit"] for c in curve], color=pine, lw=2.2)
    best = max(curve, key=lambda c: c["profit"])
    ax.scatter([best["share"] * 100], [best["profit"]], color=ink, zorder=3)
    ax.annotate(f"Peak: contact {best['share']:.0%}, ${best['profit']:,.0f}", (best["share"] * 100, best["profit"]),
                xytext=(12, -4), textcoords="offset points")
    ax.set(xlabel="Share of customers contacted, best prospects first (%)", ylabel="Campaign profit ($)",
           title="Who's worth an offer? (holdout set, 1,409 customers)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout(); fig.savefig(DOCS / "profit_curve.png", dpi=160); plt.close(fig)

    # Coefficients, grouped and labeled for people.
    pairs = sorted(zip(model_json["features"], model_json["coefficients"]), key=lambda t: t[1])
    labels = [pretty(f) for f, _ in pairs]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.barh(labels, [c for _, c in pairs], color=[pine if c < 0 else "#a23b3b" for _, c in pairs])
    ax.axvline(0, color=ink, lw=0.8)
    ax.set(xlabel="Effect on churn log-odds (numeric inputs per standard deviation)",
           title="What moves churn risk (red raises it)")
    fig.tight_layout(); fig.savefig(DOCS / "drivers.png", dpi=160); plt.close(fig)


def pretty(feature: str) -> str:
    kind, name = feature.split("__", 1)
    if kind == "cat":
        col, level = name.split("_", 1)
        return f"{LABELS[col].capitalize()}: {level}"
    return LABELS[name].capitalize()


def main():
    df = clean(load_raw())
    train, test = split(df)
    comparison = compare(train)
    print(comparison.round(4).to_string(index=False))

    final = candidates()["logistic_regression"].fit(train[FEATURES], train["churned"])
    p = final.predict_proba(test[FEATURES])[:, 1]
    y, monthly = test["churned"].values, test["MonthlyCharges"].values
    a = Assumptions()

    # The shipped model leaves age out. Measure what including it would have bought.
    with_age_features = FEATURES + ["SeniorCitizen"]
    with_age = logistic(BINARY + ["SeniorCitizen"]).fit(train[with_age_features], train["churned"])
    auc_with_age = roc_auc_score(y, with_age.predict_proba(test[with_age_features])[:, 1])

    metrics = holdout_metrics(y, p)
    cal = calibration_bins(y, p)
    pol = policies(p, y, monthly, a)
    curve = profit_curve(p, y, monthly, a)
    sens = sensitivity(p, y, monthly, a)
    boot = bootstrap_difference(p, y, monthly, a)
    fair = fairness(test, p, a)
    model_json = export_model(final, train)

    results = {
        "data": {"source": SOURCE_URL, "license": "Apache-2.0 (IBM)", "rows": len(df),
                 "train": len(train), "holdout": len(test), "churn_rate": float(df["churned"].mean()),
                 "blank_total_charges_fixed": 11},
        "cv": comparison.to_dict(orient="records"),
        "holdout": metrics,
        "auc_with_age": auc_with_age,
        "calibration": cal,
        "assumptions": a.to_dict(),
        "policies": pol,
        "profit_curve": curve,
        "sensitivity": sens,
        "bootstrap_ev_minus_top20": boot,
        "fairness": fair,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=1))
    SITE.mkdir(exist_ok=True)
    (SITE / "model.json").write_text(json.dumps(model_json))
    (SITE / "results.json").write_text(json.dumps({k: v for k, v in results.items() if k != "sensitivity"} | {"sensitivity": sens}))
    (SITE / "holdout.json").write_text(json.dumps({"p": [round(float(v), 6) for v in p], "churned": [int(v) for v in y],
                                                   "monthly": [float(v) for v in monthly]}))
    # Cases for the JavaScript parity test: raw inputs and Python's probability.
    cases = test[FEATURES].head(40).copy()
    cases["expected"] = p[:40]
    (TESTS / "parity_cases.json").write_text(cases.to_json(orient="records"))
    figures(cal, curve, model_json, sens)

    print(f"\nHoldout: ROC AUC {metrics['roc_auc']:.3f}, PR AUC {metrics['pr_auc']:.3f}, Brier {metrics['brier']:.3f}")
    print(f"Adding SeniorCitizen would give ROC AUC {auc_with_age:.4f} ({auc_with_age - metrics['roc_auc']:+.4f})")
    for r in pol:
        print(f"  {r['policy']:<24} contacts {r['contacted']:>4}  profit ${r['profit']:>9,.0f}")
    print(f"EV policy minus riskiest-20%: mean ${boot['mean']:,.0f}, 95% CI ${boot['low']:,.0f} to ${boot['high']:,.0f}, "
          f"EV better in {boot['share_ev_better']:.0%} of resamples")
    for f in fair:
        print(f"  {f['group']:<7} {str(f['value']):<9} n={f['customers']:<4} actual {f['actual_churn']:.1%} "
              f"predicted {f['predicted_churn']:.1%} AUC {f['roc_auc']:.3f} offered {f['share_offered']:.0%}")


if __name__ == "__main__":
    main()
