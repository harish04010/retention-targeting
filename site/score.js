// The model, reimplemented for the browser. Exported as an ES module so the
// page and the parity test (tests/parity.test.mjs) share one implementation.

export const SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];

export function servicesCount(c) {
  return SERVICES.reduce((n, s) => n + (+c[s] || 0), 0) + (+c.PhoneService || 0) + (c.InternetService !== "No" ? 1 : 0);
}

// Same order as scikit-learn's get_feature_names_out: one-hot categoricals
// (first level dropped), then binary flags, then standardized numbers.
export function encode(model, c) {
  const x = [];
  for (const [col, { levels }] of Object.entries(model.categorical)) {
    for (const level of levels.slice(1)) x.push(c[col] === level ? 1 : 0);
  }
  for (const b of model.binary) x.push(+c[b] || 0);
  for (const [col, { mean, scale }] of Object.entries(model.numeric)) x.push((Number(c[col]) - mean) / scale);
  return x;
}

export function predict(model, c) {
  const x = encode(model, c);
  const z = model.intercept + x.reduce((s, v, i) => s + v * model.coefficients[i], 0);
  return 1 / (1 + Math.exp(-z));
}

// Why is this customer riskier or safer than the average customer?
// Each input's push on the log-odds, compared with the training average,
// summed per original field so "contract type" is one reason, not three columns.
export function reasons(model, c) {
  const x = encode(model, c);
  const byField = new Map();
  model.features.forEach((name, i) => {
    const field = fieldOf(name);
    const push = model.coefficients[i] * (x[i] - model.baseline[i]);
    byField.set(field, (byField.get(field) ?? 0) + push);
  });
  return [...byField.entries()]
    .map(([field, push]) => ({ field, label: model.labels[field] ?? field, push }))
    .sort((a, b) => Math.abs(b.push) - Math.abs(a.push));
}

function fieldOf(name) {
  const [kind, rest] = name.split("__");
  if (kind === "cat") return rest.slice(0, rest.indexOf("_"));
  return rest;
}

// ---- the decision, mirroring src/business.py ----

export function expectedValue(p, monthly, a) {
  const saved = monthly * a.margin * a.horizon_months;
  const offer = monthly * a.offer_months_free;
  return p * a.save_rate * saved - a.contact_cost - (p * a.save_rate + (1 - p) * a.stayer_accept_rate) * offer;
}

export function realizedValue(churned, monthly, a) {
  const saved = monthly * a.margin * a.horizon_months;
  const offer = monthly * a.offer_months_free;
  return churned
    ? a.save_rate * saved - a.contact_cost - a.save_rate * offer
    : -a.contact_cost - a.stayer_accept_rate * offer;
}
