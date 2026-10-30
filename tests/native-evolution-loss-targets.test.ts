import { expect, test } from "bun:test";
import { retentionLossRecords } from "../backend/native-evolution-loss-targets.ts";

const row = (answer = "") => ({ id: "ordinary", answer, prompt: "2+2?", training_allowed: false,
  provenance: { generation_check: { type: "semantic", validator: "integer", expected: 4 } } });
const observation = () => ({ id: "ordinary", answer: "4", passed: true, complete: true, branch_enabled: false, reference_used: false });

test("empty semantic gold becomes a frozen read-only verified answer, without changing original records", () => {
  const original = [row()], before = structuredClone(original), baseline = observation();
  const targets = retentionLossRecords(original, [baseline]);
  expect(targets[0]!.answer).toBe("4"); expect(targets[0]!.training_allowed).toBe(false);
  expect(Object.isFrozen(targets)).toBe(true); expect(Object.isFrozen(targets[0])).toBe(true);
  baseline.answer = "wrong"; expect(targets[0]!.answer).toBe("4"); expect(original).toEqual(before);
});

test("authored nonempty gold stays unchanged, including whitespace, without requiring a generated substitute", () => {
  expect(retentionLossRecords([row(" 4 ")], [])[0]!.answer).toBe(" 4 ");
  expect(retentionLossRecords([row("four")], [observation()])[0]!.answer).toBe("four");
});

test("missing, incorrect, empty, truncated, expert/reference or duplicate baseline cannot provide loss targets", () => {
  expect(() => retentionLossRecords([row()], [])).toThrow();
  for (const change of [{ answer: "5" }, { answer: "" }, { answer: "  " }, { passed: false },
    { complete: false }, { branch_enabled: true }, { reference_used: true }, { reference_used: undefined }]) {
    expect(() => retentionLossRecords([row()], [{ ...observation(), ...change }])).toThrow();
  }
  expect(() => retentionLossRecords([row()], [observation(), observation()])).toThrow();
});
