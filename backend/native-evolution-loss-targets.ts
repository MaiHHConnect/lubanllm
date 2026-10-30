/** Read-only CE targets; never add generated validation answers to TRAIN. */
import assert from "node:assert/strict";
import { materialProvenance } from "./native-evolution-material.ts";
import { evaluateGeneration, validateGenerationCheck } from "./generation-checks.ts";

export function retentionLossRecords(records: Record<string, any>[], baseline: Record<string, any>[]): readonly Readonly<Record<string, any>>[] {
  const observed = new Map(baseline.map(row => [row.id, row]));
  assert.equal(observed.size, baseline.length, "Duplicate retention baseline observations");
  return Object.freeze(records.map(row => {
    assert(typeof row.answer === "string", "Retention loss target must be a string");
    if (row.answer !== "") return Object.freeze({ ...row });
    const value = observed.get(row.id);
    assert(value?.passed === true && value.complete === true && value.branch_enabled === false
      && value.reference_used === false && typeof value.answer === "string" && value.answer.trim(),
    "Empty retention target requires a correct complete core-only baseline answer");
    const check = validateGenerationCheck(materialProvenance(row).generation_check, row.answer);
    assert(evaluateGeneration(value.answer, check, true).passed, "Retention baseline answer failed its original check");
    return Object.freeze({ ...row, answer: value.answer });
  }));
}
