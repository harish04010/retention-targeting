import { expectedValue, predict, realizedValue, reasons, SERVICES, servicesCount } from "./score.js";

const [model, results, holdout] = await Promise.all(
  ["model.json", "results.json", "holdout.json"].map((f) => fetch(f).then((r) => r.json())),
);

const usd = (n) => (n < 0 ? "−$" : "$") + Math.abs(Math.round(n)).toLocaleString("en-US");
const pct = (n, d = 0) => `${(n * 100).toFixed(n > 0 && n < 0.01 && d === 0 ? 1 : d)}%`;
const el = (tag, attrs = {}, html = "") => Object.assign(document.createElement(tag), attrs, { innerHTML: html });

// ------------------------------------------------------------ simulator

const FIELDS = [
  { key: "Contract", label: "Contract", options: ["Month-to-month", "One year", "Two year"] },
  { key: "tenure", label: "Months as a customer", range: [0, 72, 1] },
  { key: "MonthlyCharges", label: "Monthly bill", range: [18, 120, 1], money: true },
  { key: "InternetService", label: "Internet", options: ["DSL", "Fiber optic", "No"] },
  { key: "PaymentMethod", label: "Pays by", options: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] },
  { key: "PaperlessBilling", label: "Paperless billing", toggle: true },
  { key: "TechSupport", label: "Tech support", toggle: true, internet: true },
  { key: "OnlineSecurity", label: "Online security", toggle: true, internet: true },
  { key: "OnlineBackup", label: "Online backup", toggle: true, internet: true },
  { key: "DeviceProtection", label: "Device protection", toggle: true, internet: true },
  { key: "StreamingTV", label: "Streaming TV", toggle: true, internet: true },
  { key: "StreamingMovies", label: "Streaming movies", toggle: true, internet: true },
  { key: "PhoneService", label: "Phone line", toggle: true },
  { key: "MultipleLines", label: "Multiple lines", toggle: true },
  { key: "Partner", label: "Has a partner", toggle: true },
  { key: "Dependents", label: "Has dependents", toggle: true },
];

// Starts as a typical at-risk customer: new, month-to-month, fiber, paying by e-check.
const customer = {
  Contract: "Month-to-month", tenure: 4, MonthlyCharges: 85, InternetService: "Fiber optic",
  PaymentMethod: "Electronic check", PaperlessBilling: 1, TechSupport: 0, OnlineSecurity: 0, OnlineBackup: 0,
  DeviceProtection: 0, StreamingTV: 1, StreamingMovies: 0, PhoneService: 1, MultipleLines: 0, Partner: 0, Dependents: 0,
};

const form = document.getElementById("customer");
for (const f of FIELDS) {
  const id = `f-${f.key}`;
  const wrap = el("div", { className: `field ${f.toggle ? "field--toggle" : ""}` });
  if (f.options) {
    wrap.append(el("label", { htmlFor: id }, f.label));
    const s = el("select", { id, name: f.key });
    f.options.forEach((o) => s.append(el("option", { value: o, selected: customer[f.key] === o }, o)));
    wrap.append(s);
  } else if (f.range) {
    const out = el("output", { id: `${id}-out` });
    wrap.append(el("label", { htmlFor: id }, `${f.label} `), out);
    const r = el("input", { id, name: f.key, type: "range", min: f.range[0], max: f.range[1], step: f.range[2], value: customer[f.key] });
    wrap.append(r);
  } else {
    const c = el("input", { id, name: f.key, type: "checkbox", checked: !!customer[f.key] });
    wrap.append(c, el("label", { htmlFor: id }, f.label));
  }
  if (f.internet) wrap.dataset.internet = "1";
  form.append(wrap);
}

form.addEventListener("input", (e) => {
  const t = e.target;
  customer[t.name] = t.type === "checkbox" ? (t.checked ? 1 : 0) : t.type === "range" ? Number(t.value) : t.value;
  renderCustomer();
});

const avgP = 1 / (1 + Math.exp(-(model.intercept + model.baseline.reduce((s, v, i) => s + v * model.coefficients[i], 0))));

function renderCustomer() {
  const noInternet = customer.InternetService === "No";
  form.querySelectorAll("[data-internet] input").forEach((i) => {
    i.disabled = noInternet;
    if (noInternet) { i.checked = false; customer[i.name] = 0; }
  });
  if (!customer.PhoneService) { customer.MultipleLines = 0; form.MultipleLines.checked = false; }
  form.MultipleLines.disabled = !customer.PhoneService;
  form.querySelector("#f-tenure-out").textContent = `${customer.tenure}`;
  form.querySelector("#f-MonthlyCharges-out").textContent = `$${customer.MonthlyCharges}`;

  const c = { ...customer, services_count: servicesCount(customer) };
  const p = predict(model, c);
  document.getElementById("risk").textContent = pct(p);
  document.getElementById("risk-marker").style.left = pct(p, 2);
  document.getElementById("avg-marker").style.left = pct(avgP, 2);
  document.getElementById("avg-label").style.left = pct(avgP, 2);
  document.querySelector(".sim__out").dataset.level = p > 0.5 ? "high" : p > 0.25 ? "mid" : "low";

  const top = reasons(model, c).filter((r) => Math.abs(r.push) > 0.05).slice(0, 5);
  document.getElementById("reasons").innerHTML = top.length
    ? top.map((r) => `<li class="${r.push > 0 ? "up" : "down"}"><span>${describe(r.field, c)}</span>
        <span class="push">${r.push > 0 ? "raises" : "lowers"} risk</span></li>`).join("")
    : "<li>Close to the average customer on everything.</li>";

  const a = currentAssumptions();
  const ev = expectedValue(p, customer.MonthlyCharges, a);
  document.getElementById("verdict").innerHTML = ev > 0
    ? `<strong>Worth an offer.</strong> Expected profit from calling: ${usd(ev)}, using the campaign assumptions below.`
    : `<strong>Not worth an offer.</strong> Expected value of calling: ${usd(ev)}. The discount would cost more than the churn it’s likely to prevent.`;
  document.getElementById("verdict").dataset.go = ev > 0 ? "yes" : "no";
}

// Reasons in words a retention agent would use.
function describe(field, c) {
  const yes = (k) => (c[k] ? "has" : "no");
  switch (field) {
    case "Contract": return c.Contract === "Month-to-month" ? "Month-to-month contract" : `${c.Contract} contract`;
    case "tenure": return `${c.tenure} month${c.tenure === 1 ? "" : "s"} as a customer`;
    case "MonthlyCharges": return `$${c.MonthlyCharges} monthly bill`;
    case "InternetService": return c.InternetService === "No" ? "No internet service" : `${c.InternetService} internet`;
    case "PaymentMethod": return `Pays by ${c.PaymentMethod.toLowerCase()}`;
    case "PaperlessBilling": return c.PaperlessBilling ? "Paperless billing" : "Paper bills";
    case "services_count": return `${c.services_count} services in total`;
    default: return `${model.labels[field][0].toUpperCase()}${model.labels[field].slice(1)}: ${yes(field)}`;
  }
}

// ------------------------------------------------------------ campaign planner

const CONTROLS = [
  { key: "save_rate", label: "Offers that change a would-be leaver’s mind", min: 0.05, max: 0.6, step: 0.05, fmt: (v) => pct(v) },
  { key: "stayer_accept_rate", label: "Customers who’d have stayed but take the offer anyway", min: 0, max: 1, step: 0.05, fmt: (v) => pct(v) },
  { key: "offer_months_free", label: "Offer: free months of service", min: 0.5, max: 3, step: 0.5, fmt: (v) => `${v}` },
  { key: "margin", label: "Profit margin on a monthly bill", min: 0.1, max: 0.7, step: 0.05, fmt: (v) => pct(v) },
  { key: "contact_cost", label: "Cost to contact one customer", min: 0, max: 30, step: 1, fmt: (v) => `$${v}` },
];
const assumptions = { ...results.assumptions };
const controls = document.getElementById("assumptions");
for (const c of CONTROLS) {
  const id = `a-${c.key}`;
  const wrap = el("div", { className: "field" });
  const out = el("output", { id: `${id}-out` }, c.fmt(assumptions[c.key]));
  wrap.append(el("label", { htmlFor: id }, `${c.label} `), out,
    el("input", { id, type: "range", min: c.min, max: c.max, step: c.step, value: assumptions[c.key] }));
  wrap.querySelector("input").addEventListener("input", (e) => {
    assumptions[c.key] = Number(e.target.value);
    out.textContent = c.fmt(assumptions[c.key]);
    renderPlan();
    renderCustomer();
  });
  controls.append(wrap);
}
controls.append(el("button", { type: "button", className: "btn", id: "reset" }, "Back to the baseline assumptions"));
document.getElementById("reset").addEventListener("click", () => {
  Object.assign(assumptions, results.assumptions);
  for (const c of CONTROLS) {
    document.getElementById(`a-${c.key}`).value = assumptions[c.key];
    document.getElementById(`a-${c.key}-out`).textContent = c.fmt(assumptions[c.key]);
  }
  renderPlan(); renderCustomer();
});
function currentAssumptions() { return { ...assumptions, horizon_months: results.assumptions.horizon_months }; }

function renderPlan() {
  const a = currentAssumptions();
  const n = holdout.p.length;
  const realized = holdout.p.map((_, i) => realizedValue(holdout.churned[i], holdout.monthly[i], a));
  const ev = holdout.p.map((p, i) => expectedValue(p, holdout.monthly[i], a));
  const order = [...ev.keys()].sort((i, j) => ev[j] - ev[i]);
  let run = 0;
  const curve = order.map((i, k) => ({ share: (k + 1) / n, profit: (run += realized[i]) }));

  // policies
  const byRisk = [...holdout.p.keys()].sort((i, j) => holdout.p[j] - holdout.p[i]);
  const top20 = new Set(byRisk.slice(0, Math.floor(0.2 * n)));
  const sum = (pred) => holdout.p.reduce((s, _, i) => (pred(i) ? s + realized[i] : s), 0);
  const count = (pred) => holdout.p.filter((_, i) => pred(i)).length;
  const rows = [
    ["Contact everyone", () => true],
    ["Riskiest 20%", (i) => top20.has(i)],
    ["Positive expected value", (i) => ev[i] > 0],
  ].map(([name, pred]) => ({ name, contacted: count(pred), profit: sum(pred) }));
  const best = rows.reduce((b, r) => (r.profit > b.profit ? r : b));
  document.getElementById("policies").innerHTML = `
    <thead><tr><th>Policy</th><th class="num">Customers contacted</th><th class="num">Campaign profit</th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="${r === best ? "is-best" : ""}"><th scope="row">${r.name}</th>
      <td class="num">${r.contacted.toLocaleString()}</td><td class="num ${r.profit < 0 ? "neg" : ""}">${usd(r.profit)}</td></tr>`).join("")}
    <tr><th scope="row">Contact nobody</th><td class="num">0</td><td class="num">$0</td></tr></tbody>`;

  drawCurve(curve);
  const peak = curve.reduce((b, c) => (c.profit > b.profit ? c : b), { share: 0, profit: 0 });
  const evRow = rows.find((r) => r.name === "Positive expected value");
  document.getElementById("curve-note").textContent = peak.profit > 0
    ? `In hindsight, the best cutoff on these customers was the top ${pct(peak.share)}, for ${usd(peak.profit)}. ` +
      `Without peeking at outcomes, the expected-value rule chose ${pct(evRow.contacted / n)} and made ${usd(evRow.profit)}. ` +
      `Past the peak, every extra call loses money.`
    : "Under these assumptions no campaign pays. The best move is not to run one.";
}

function drawCurve(curve) {
  const svg = document.getElementById("curve");
  const W = 640, H = 300, L = 64, B = 50, T = 14, R = 12;
  const lo = Math.min(0, ...curve.map((c) => c.profit)), hi = Math.max(0, ...curve.map((c) => c.profit));
  const span = hi - lo || 1;
  const x = (s) => L + (W - L - R) * s;
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / span);
  const step = niceStep(span / 4);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  const path = curve.filter((_, i) => i % 3 === 0 || i === curve.length - 1)
    .map((c, i) => `${i ? "L" : "M"}${x(c.share).toFixed(1)},${y(c.profit).toFixed(1)}`).join(" ");
  const peak = curve.reduce((b, c) => (c.profit > b.profit ? c : b), curve[0]);
  svg.innerHTML = `
    ${ticks.map((v) => `<line class="grid ${v === 0 ? "zero" : ""}" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>
      <text class="axis" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${usd(v)}</text>`).join("")}
    ${[0, 0.25, 0.5, 0.75, 1].map((s) => `<text class="axis" x="${x(s)}" y="${H - 30}" text-anchor="middle">${pct(s)}</text>`).join("")}
    <text class="axis" x="${(L + W) / 2}" y="${H - 6}" text-anchor="middle">customers contacted, best prospects first</text>
    <path class="curve" d="${path}"/>
    ${peak.profit > 0 ? `<circle class="peak" cx="${x(peak.share)}" cy="${y(peak.profit)}" r="5"/>` : ""}`;
}
function niceStep(raw) {
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
}

// ------------------------------------------------------------ model card

function renderCard() {
  const r = results;
  const cv = r.cv.map((m) => `<tr class="${m.model === "logistic_regression" ? "is-best" : ""}">
      <th scope="row">${m.model.replace("_", " ").replace(/^./, (c) => c.toUpperCase())}${m.model === "logistic_regression" ? " (shipped)" : ""}</th>
      <td class="num">${m.roc_auc.toFixed(3)} ± ${m.roc_auc_sd.toFixed(3)}</td><td class="num">${m.pr_auc.toFixed(3)}</td><td class="num">${m.brier.toFixed(3)}</td></tr>`).join("");
  const fair = r.fairness.map((f) => `<tr><th scope="row">${f.value}</th><td class="num">${f.customers}</td>
      <td class="num">${pct(f.actual_churn, 1)}</td><td class="num">${pct(f.predicted_churn, 1)}</td>
      <td class="num">${f.roc_auc.toFixed(3)}</td><td class="num">${pct(f.share_offered)}</td></tr>`).join("");
  const b = r.bootstrap_ev_minus_top20;
  const cal = r.calibration;
  const cx = (v) => 30 + v * 240, cy = (v) => 250 - v * 220;
  document.getElementById("card").innerHTML = `
    <div class="card__block">
      <h3>Choosing the model</h3>
      <p>Five-fold cross-validation on the training set. Random forest scored a hair higher, by less than the noise between folds. Logistic regression was shipped: it’s just as accurate, better calibrated, explainable line by line, and small enough to run on this page.</p>
      <div class="scroll"><table><thead><tr><th>Model</th><th class="num">ROC AUC</th><th class="num">PR AUC</th><th class="num">Brier (lower is better)</th></tr></thead><tbody>${cv}</tbody></table></div>
      <p class="small note">On the untouched holdout set: ROC AUC ${r.holdout.roc_auc.toFixed(3)}, PR AUC ${r.holdout.pr_auc.toFixed(3)}, against a ${pct(r.holdout.base_rate, 1)} churn rate.</p>
    </div>
    <div class="card__block">
      <h3>Do the probabilities mean what they say?</h3>
      <p>Customers grouped by predicted risk. When the model says 60%, about 60% leave.</p>
      <svg viewBox="0 0 290 280" class="cal" role="img" aria-label="Calibration chart">
        <line class="grid" x1="${cx(0)}" y1="${cy(0)}" x2="${cx(1)}" y2="${cy(1)}" stroke-dasharray="4 3"/>
        <line class="grid" x1="${cx(0)}" y1="${cy(0)}" x2="${cx(1)}" y2="${cy(0)}"/>
        <line class="grid" x1="${cx(0)}" y1="${cy(0)}" x2="${cx(0)}" y2="${cy(1)}"/>
        <polyline class="curve" points="${cal.map((c) => `${cx(c.predicted)},${cy(c.actual)}`).join(" ")}"/>
        ${cal.map((c) => `<circle class="dot" cx="${cx(c.predicted)}" cy="${cy(c.actual)}" r="4"/>`).join("")}
        <text class="axis" x="150" y="276" text-anchor="middle">predicted</text>
        <text class="axis" x="12" y="140" text-anchor="middle" transform="rotate(-90 12 140)">actual</text>
      </svg>
    </div>
    <div class="card__block card__wide">
      <h3>Is it fair across groups?</h3>
      <p>Gender isn’t a model input. Age was tested and left out: it added only ${(r.auc_with_age - r.holdout.roc_auc).toFixed(4)} AUC, not worth using age to decide who gets a discount. Seniors still get offers far more often, because they really do churn more and the model sees it through other signals. The cost is that it under-predicts their churn by about ${Math.round((r.fairness[3].actual_churn - r.fairness[3].predicted_churn) * 100)} points, and ranks them less sharply.</p>
      <div class="scroll"><table><thead><tr><th>Group</th><th class="num">Customers</th><th class="num">Actually churned</th><th class="num">Predicted</th><th class="num">ROC AUC</th><th class="num">Offered a deal</th></tr></thead><tbody>${fair}</tbody></table></div>
    </div>
    <div class="card__block card__wide">
      <h3>What this can’t tell you</h3>
      <ul class="limits">
        <li>The two smart policies are a statistical tie at the baseline assumptions. Across ${2000} bootstrap resamples, expected value minus riskiest-20% averaged ${usd(b.mean)}, with a 95% interval of ${usd(b.low)} to ${usd(b.high)}. The expected-value rule earns its keep when assumptions move: it shrinks or grows the campaign on its own.</li>
        <li>Save rate and acceptance rate are guesses. The only way to know them is a randomized test: offer half the at-risk customers a deal and compare. Until then, treat the profit figures as a range, not a forecast.</li>
        <li>The data is one snapshot from a sample IBM published, not a live customer base. Churn here means “left last month”; a real system would score customers on a schedule and watch for drift.</li>
        <li>Predicting who leaves isn’t the same as predicting who an offer would keep. Uplift modeling, trained on that randomized test, would target persuadable customers directly.</li>
      </ul>
    </div>`;
}

renderCustomer();
renderPlan();
renderCard();
