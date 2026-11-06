"""
Turning churn scores into a decision: who gets a retention offer?

A score alone doesn't answer that. An offer costs money every time it's
accepted, including by customers who were never going to leave. The value of
calling someone depends on how likely they are to churn, how much they're
worth, and how often an offer actually changes their mind.

Every number below is an assumption, stated so it can be argued with. The
sensitivity analysis shows how far each one can move before the conclusion
changes.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Assumptions:
    margin: float = 0.40              # share of the monthly bill that's profit
    horizon_months: int = 12          # how long a saved customer is counted for
    save_rate: float = 0.30           # share of would-be churners an offer keeps
    offer_months_free: float = 1.0    # the offer: a month's bill credited
    stayer_accept_rate: float = 0.60  # share of customers who'd have stayed anyway that take the offer
    contact_cost: float = 5.0         # cost of the call or campaign touch, per customer

    def to_dict(self) -> dict:
        return asdict(self)


def expected_value(p_churn, monthly, a: Assumptions):
    """Expected profit of contacting each customer, before we know who churns."""
    value_saved = monthly * a.margin * a.horizon_months
    offer_cost = monthly * a.offer_months_free
    gain = p_churn * a.save_rate * value_saved
    cost = a.contact_cost + (p_churn * a.save_rate + (1 - p_churn) * a.stayer_accept_rate) * offer_cost
    return gain - cost


def realized_value(churned, monthly, a: Assumptions):
    """
    What contacting each customer is worth once we know who actually churned.
    Uses expected save and acceptance rates rather than coin flips, so the
    comparison between policies isn't noise.
    """
    churned = np.asarray(churned).astype(bool)
    value_saved = monthly * a.margin * a.horizon_months
    offer_cost = monthly * a.offer_months_free
    if_churner = a.save_rate * value_saved - a.contact_cost - a.save_rate * offer_cost
    if_stayer = -a.contact_cost - a.stayer_accept_rate * offer_cost
    return np.where(churned, if_churner, if_stayer)


def policies(p, churned, monthly, a: Assumptions, seed: int = 0) -> list[dict]:
    """Compare targeting rules on the holdout set."""
    n = len(p)
    realized = realized_value(churned, monthly, a)
    ev = expected_value(p, monthly, a)
    rng = np.random.default_rng(seed)

    def summarize(name, mask, rule):
        return {"policy": name, "rule": rule, "contacted": int(mask.sum()), "share": float(mask.mean()),
                "profit": float(realized[mask].sum()),
                "churners_reached": int(np.asarray(churned)[mask].sum())}

    top20 = np.zeros(n, bool)
    top20[np.argsort(-p)[: int(0.2 * n)]] = True
    rand20 = np.zeros(n, bool)
    rand20[rng.choice(n, int(0.2 * n), replace=False)] = True
    return [
        summarize("Contact nobody", np.zeros(n, bool), "Do nothing"),
        summarize("Contact everyone", np.ones(n, bool), "Blanket campaign"),
        summarize("Random 20%", rand20, "What a campaign looks like without a model"),
        summarize("Riskiest 20%", top20, "Highest churn probability first"),
        summarize("Positive expected value", ev > 0, "Anyone whose expected profit from a call is above zero"),
    ]


def profit_curve(p, churned, monthly, a: Assumptions, points: int = 50) -> list[dict]:
    """Cumulative profit as we contact customers in order of expected value."""
    order = np.argsort(-expected_value(p, monthly, a))
    cum = np.cumsum(realized_value(churned, monthly, a)[order])
    idx = np.unique(np.linspace(0, len(p) - 1, points).astype(int))
    return [{"share": float((i + 1) / len(p)), "profit": float(cum[i])} for i in idx]


def sensitivity(p, churned, monthly, base: Assumptions) -> list[dict]:
    """Vary one assumption at a time. Does the model-driven policy still beat the alternatives?"""
    grid = {
        "save_rate": [0.10, 0.20, 0.30, 0.40, 0.50],
        "margin": [0.25, 0.40, 0.55],
        "stayer_accept_rate": [0.30, 0.60, 0.90],
        "offer_months_free": [0.5, 1.0, 2.0],
    }
    rows = []
    for field, values in grid.items():
        for v in values:
            a = Assumptions(**{**base.to_dict(), field: v})
            results = {r["policy"]: r for r in policies(p, churned, monthly, a)}
            rows.append({"assumption": field, "value": v,
                         "ev_policy_profit": results["Positive expected value"]["profit"],
                         "ev_policy_share": results["Positive expected value"]["share"],
                         "top20_profit": results["Riskiest 20%"]["profit"],
                         "everyone_profit": results["Contact everyone"]["profit"]})
    return rows
