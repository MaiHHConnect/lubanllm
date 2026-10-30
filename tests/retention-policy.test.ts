import { expect, test } from "bun:test";
import { evaluateRetention, retentionAcceptsGrowth, retentionClass, softForgetBudget } from "../backend/retention-policy.ts";

const soft = (id: string) => ({ id, provenance: { generation_check: { type: "exact", expected: id } } });

test("bounded retention accepts one soft regression when five old items gain two", () => {
  const records = Array.from({ length: 7 }, (_, i) => soft(`item-${i}`));
  const before = new Set(records.slice(0, 5).map(row => row.id));
  const after = new Set(["item-0", "item-1", "item-2", "item-3", "item-5", "item-6"]);
  const value = evaluateRetention(records, before, after);
  expect(softForgetBudget(5)).toBe(1);
  expect(value.soft_forgotten_ids).toEqual(["item-4"]);
  expect(retentionAcceptsGrowth(value, 2)).toBe(true);
});

test("small soft cohorts keep a zero regression budget and hard identity stays absolute", () => {
  const softRecords = Array.from({ length: 4 }, (_, i) => soft(`small-${i}`));
  const softValue = evaluateRetention(softRecords, new Set(softRecords.map(row => row.id)), new Set(["small-0", "small-1", "small-2"]));
  expect(softValue.accepted).toBe(false);
  const hard = { id: "identity", provenance: { protocol_id: "explicit-model-identity-v1", teaching_fact: "model_name" } };
  expect(retentionClass(hard)).toBe("hard");
  expect(evaluateRetention([hard], new Set([hard.id]), new Set()).accepted).toBe(false);
});
