// The browser must give the same answer as scikit-learn, to the last decimal that matters.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { predict, reasons, servicesCount } from "../site/score.js";

const model = JSON.parse(readFileSync(new URL("../site/model.json", import.meta.url)));
const cases = JSON.parse(readFileSync(new URL("./parity_cases.json", import.meta.url)));

test("JavaScript predictions match scikit-learn", () => {
  for (const c of cases) {
    assert.ok(Math.abs(predict(model, c) - c.expected) < 1e-9, `${predict(model, c)} vs ${c.expected}`);
  }
});

test("services count is rebuilt the same way as in Python", () => {
  for (const c of cases) assert.equal(servicesCount(c), c.services_count);
});

test("reasons add up to the gap between this customer and the average", () => {
  const c = cases[0];
  const total = reasons(model, c).reduce((s, r) => s + r.push, 0);
  const x = model.baseline;
  const zAvg = model.intercept + x.reduce((s, v, i) => s + v * model.coefficients[i], 0);
  const zThis = Math.log(predict(model, c) / (1 - predict(model, c)));
  assert.ok(Math.abs(total - (zThis - zAvg)) < 1e-9);
});
