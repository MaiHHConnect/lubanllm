/** Product material policy, ported from retention.py, original_protection.py and automatic.py.
 * Pure collection: callers supply the effective core and its already-bound lifecycle inputs.
 * No model loading, tensor reads, snapshot creation, publication or per-sample disk audit.
 */
import { createHash } from "node:crypto";
import { ledgerParse, ledgerStringify, pythonStrip } from "./memory";
import { encoded } from "./canonical-json";
import { evaluateGeneration, pythonNFKC, validateGenerationCheck } from "./generation-checks";

export const EVOLUTION_GROUP_KEYS = ["new_training", "new_validation", "replay_training", "replay_validation"] as const;
export const DIRECTED_TEACHING = "explicit-model-identity-v1";
type Row = Record<string, any>;
export type EvolutionGroups = Record<typeof EVOLUTION_GROUP_KEYS[number], Row[]>;
const object = (v: any): v is Row => v !== null && typeof v === "object" && !Array.isArray(v) && !(v instanceof Number);
function need(condition: unknown, message: string): asserts condition { if (!condition) throw new Error(message); }
const exactKeys = (v: any, keys: string[]) => object(v) && Object.keys(v).sort().join("\0") === [...keys].sort().join("\0");
const truthy = (v: any): boolean => v instanceof Number ? !!Number(v) : Array.isArray(v) ? !!v.length : object(v) ? !!Object.keys(v).length : !!v;
const clone = <T>(v: T): T => ledgerParse(ledgerStringify(v));
export const materialContentSHA = (v: any, compact = false) => createHash("sha256").update(encoded(v, compact)).digest("hex");
const same = (a: any, b: any) => encoded(a) === encoded(b);
const order = (a: string, b: string) => { const x = Array.from(a, c => c.codePointAt(0)!), y = Array.from(b, c => c.codePointAt(0)!); for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i]! - y[i]!; return x.length - y.length; };
export function materialProvenance(row: Row): Row {
  const p = typeof row.provenance === "string" ? ledgerParse(row.provenance) : row.provenance === undefined ? {} : row.provenance;
  need(object(p), "材料来源记录必须为对象"); return p;
}
export function materialMessages(row: Row): Row[] {
  const p = materialProvenance(row), rows = truthy(p.messages) ? p.messages : [{ role: "user", content: row.prompt }];
  need(Array.isArray(rows) && rows.length && rows.every(r => object(r) && ["system", "user", "assistant"].includes(r.role) && typeof r.content === "string" && pythonStrip(r.content)), "保护材料需要完整的角色与非空上下文"); return rows;
}
export const materialContext = (row: Row) => encoded(materialMessages(row).map(r => ({ ...r, content: pythonNFKC(r.content).replace(/[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/gu, "") })), true);
function verified(row: Row) { const p = materialProvenance(row); return row.source === "human" || (p.program_verified === true || p.human_verified === true) && typeof p.verification_evidence === "string" && !!pythonStrip(p.verification_evidence); }
export function eligibleMaterial(row: Row) { const p = materialProvenance(row); return truthy(row.approved) && truthy(row.training_allowed) && !truthy(p.route_only) && !truthy(p.retention_generation_only) && verified(row); }
export function dailyLearningMaterial(row: Row) {
  const p = materialProvenance(row);
  if (!["human", "teacher"].includes(row.source) || ["dataset", "protocol_id", "benchmark_run", "route_only", "retention_generation_only"].some(k => truthy(p[k]))) return false;
  return p.verified_by === "local requester" || truthy(p.feeding_batch_id) && p.human_verified === true && truthy(p.verification_evidence) || row.source === "teacher" && truthy(p.request_id) && truthy(p.messages);
}
export function validateMaterialRecord(row: any, generationOnly = false): asserts row is Row {
  need(object(row) && typeof row.id === "string" && row.id && typeof row.answer === "string" && ["training", "validation"].includes(row.purpose), "核心保护材料缺少完整记录或有效拆分");
  materialContext(row); const p = materialProvenance(row);
  need(generationOnly ? row.purpose === "validation" && truthy(row.approved) && !truthy(row.training_allowed) && p.retention_generation_only === true && verified(row) : eligibleMaterial(row) && pythonStrip(row.answer), "核心保护材料未确认或许可无效");
  validateGenerationCheck(p.generation_check, row.answer);
}
export function validateMaterialGroups(groups: any): asserts groups is EvolutionGroups {
  need(exactKeys(groups, [...EVOLUTION_GROUP_KEYS]), "核心保护快照需要完整四组材料");
  const ids = new Set(), contexts = new Set(), families = new Map();
  for (const [key, rows] of Object.entries(groups)) {
    need(Array.isArray(rows), "核心保护材料组必须为列表"); const [role, purpose] = key.split("_");
    for (const row of rows) {
      validateMaterialRecord(row); const p = materialProvenance(row), c = materialContext(row);
      need(row.purpose === purpose && p.evolution_role === role, "核心保护材料角色或拆分不匹配");
      need(!ids.has(row.id) && !contexts.has(c), "核心保护快照存在重复ID或上下文"); ids.add(row.id); contexts.add(c);
      if (truthy(p.split_family)) { need(typeof p.split_family === "string" && (!families.has(p.split_family) || families.get(p.split_family) === purpose), "核心训练与独立检验出现相同模板家族，不能进化"); families.set(p.split_family, purpose); }
    }
  }
}
export function validateNativeCoreRetention(value: any, identity: string, version: string) {
  need(typeof version === "string" && /^core-[a-f0-9]{12}$/.test(version) && exactKeys(value, ["schema_version", "version", "identity", "groups", "regression_validation", "content_sha256"]) && value.schema_version === 1 && value.version === version && value.identity === identity, "核心保护快照格式、版本或基座身份不兼容");
  const { content_sha256, ...content } = value;
  need(materialContentSHA(content, true) === content_sha256, "核心保护快照内容校验失败");
  validateMaterialGroups(value.groups); need(Array.isArray(value.regression_validation), "核心生成保护材料必须为列表");
  const ids = new Set((Object.values(value.groups) as Row[][]).flat().map(r => r.id)), checks = new Map();
  for (const row of value.regression_validation) {
    validateMaterialRecord(row, true); need(!ids.has(row.id), "核心生成保护材料ID重复"); ids.add(row.id);
    const c = materialContext(row), check = validateGenerationCheck(materialProvenance(row).generation_check, row.answer);
    need(!checks.has(c) || same(checks.get(c), check), "核心生成保护规则对同一上下文有冲突"); checks.set(c, check);
  }
  return value;
}
export function originalCoreProtectionRows(payload: any, identity: string, originalTensorSHA256: string): Row[] {
  need(object(payload) && payload.schema_version === 1 && payload.protocol === "original-core-protection-v1" && payload.identity === identity && payload.passed === true && Array.isArray(payload.cases) && Array.isArray(payload.generation) && payload.cases.length === payload.generation.length && payload.cases.length === 8, "原始核心保护基线格式或基座身份无效");
  need(/^[a-f0-9]{64}$/.test(originalTensorSHA256) && payload.original_tensor_sha256 === originalTensorSHA256, "原始核心保护基线与当前原始张量不匹配");
  const ids = new Set();
  return payload.cases.map((sample: Row, i: number) => {
    need(object(sample) && (typeof sample.provenance === "string" || object(sample.provenance)) && typeof sample.id === "string" && /^product-ordinary-[0-7]$/.test(sample.id) && !ids.has(sample.id), "原始核心保护基线含无效或身份样本"); ids.add(sample.id);
    const p = materialProvenance(sample), result = payload.generation[i], check = validateGenerationCheck(p.generation_check, sample.answer);
    need(object(result) && result.id === sample.id && result.passed === true && result.complete === true && result.branch_enabled === false && evaluateGeneration(result.answer, check, true).passed, "原始核心保护基线含未通过的逐条生成证据");
    const row = { ...clone(sample), source: "human", purpose: "validation", approved: true, training_allowed: false, provenance: ledgerStringify({ ...clone(p), evolution_role: "replay", retention_generation_only: true, retention_scope: "original_core", base_identity: identity, original_core_baseline: true }) };
    validateMaterialRecord(row, true); return row;
  });
}
function retentionConflicts(snapshot: any, rows: Row[]) {
  if (!snapshot) return;
  const protectedRows = Object.values(snapshot.groups).flat() as Row[];
  for (const row of rows) {
    const c = materialContext(row), old = protectedRows.find(r => r.id === row.id) ?? protectedRows.find(r => materialContext(r) === c);
    need(!old || materialContext(old) === c && old.answer === row.answer && old.purpose === row.purpose, "当前材料与已采纳知识保护的答案或拆分冲突；纠错替代协议尚未实现");
    for (const r of snapshot.regression_validation) if (materialContext(r) === c) need(evaluateGeneration(row.answer, validateGenerationCheck(materialProvenance(r).generation_check, r.answer), true).passed, "当前材料与已采纳知识保护的生成保护规则冲突；纠错替代协议尚未实现");
  }
}
export function consolidateMaterialGroups(samples: Row[], snapshot: any | null, dailyOnly = false): EvolutionGroups {
  const usable = (r: Row) => { const p = materialProvenance(r); return eligibleMaterial(r) && ["new", "replay"].includes(p.evolution_role) && ["training", "validation"].includes(r.purpose) && !(p.evolution_role === "replay" && r.purpose === "validation" && p.baseline_preservation_eligible === false); };
  const current = samples.filter(r => usable(r) && (!dailyOnly || materialProvenance(r).evolution_role === "replay" || dailyLearningMaterial(r)));
  retentionConflicts(snapshot, current);
  const groups: EvolutionGroups = { new_training: [], new_validation: [], replay_training: [], replay_validation: [] };
  const retained: Row[] = [], protectedIDs = new Set(), protectedContexts = new Set();
  if (snapshot) for (const original of Object.values(snapshot.groups).flat() as Row[]) { const r = clone(original); r.provenance = ledgerStringify({ ...materialProvenance(r), evolution_role: "replay", retention_from_version: snapshot.version }); retained.push(r); protectedIDs.add(r.id); protectedContexts.add(materialContext(r)); }
  const contexts = new Map<string, Row>(), ids = new Map<string, string>();
  for (const [isRetained, rows] of [[true, retained], [false, current]] as const) for (const row of rows) {
    if (!usable(row)) continue; const c = materialContext(row), p = materialProvenance(row);
    if (!isRetained && (protectedIDs.has(row.id) || protectedContexts.has(c))) continue;
    need(!ids.has(row.id) || ids.get(row.id) === c, "核心材料同一ID对应不同上下文"); const prior = contexts.get(c);
    if (prior) { need(prior.purpose === row.purpose && prior.answer === row.answer && materialProvenance(prior).evolution_role === p.evolution_role, "核心训练与独立检验上下文、答案或角色冲突"); continue; }
    validateMaterialRecord(row); contexts.set(c, row); ids.set(row.id, c); groups[(p.evolution_role + "_" + row.purpose) as keyof EvolutionGroups].push(clone(row));
  }
  for (const rows of Object.values(groups)) rows.sort((a, b) => order(a.id, b.id)); validateMaterialGroups(groups); return groups;
}
function validateDirectedTeaching(teaching: any, groups: EvolutionGroups) {
  need(exactKeys(teaching, ["protocol", "authorization", "facts"]) && teaching.protocol === DIRECTED_TEACHING, "定向教学协议无效");
  const auth = teaching.authorization;
  need(exactKeys(auth, ["source", "explicit", "training_allowed", "reference"]) && auth.source === "human" && auth.explicit === true && auth.training_allowed === true && typeof auth.reference === "string" && pythonStrip(auth.reference), "定向教学缺少真实明确的人类训练授权引用");
  const facts = teaching.facts, expected: Row = { model_name: "luban0", model_fine_tuner: "麦好火" };
  if (object(facts) && Object.hasOwn(facts, "model_positioning")) expected.model_positioning = "无限进化";
  need(exactKeys(facts, Object.keys(expected)), "定向教学仅授权模型名称、微调方及可选的无限进化定位，不包含用户身份");
  const covered: Record<string, Set<string>> = { training: new Set(), validation: new Set() };
  for (const [key, value] of Object.entries(expected)) {
    const fact = facts[key]; need(exactKeys(fact, ["subject", "value", "training_ids", "validation_ids"]) && fact.subject === "model" && fact.value === value, "定向教学事实必须匹配授权的模型身份与定位");
    for (const purpose of ["training", "validation"] as const) {
      const inherited = groups[("replay_" + purpose) as keyof EvolutionGroups].filter(r => /^core-[a-f0-9]{12}$/.test(materialProvenance(r).retention_from_version ?? ""));
      const ids = fact[purpose + "_ids"], rows = new Map([...groups[("new_" + purpose) as keyof EvolutionGroups], ...inherited].map(r => [r.id, r]));
      need(Array.isArray(ids) && ids.length && new Set(ids).size === ids.length && ids.every(id => typeof id === "string" && !covered[purpose]!.has(id) && rows.has(id)), "每个教学事实需要独立完整且不重复的训练与验证ID");
      for (const id of ids) { covered[purpose]!.add(id); const row = rows.get(id)!, p = materialProvenance(row), check = validateGenerationCheck(p.generation_check, row.answer);
        need(row.source === "human" && p.protocol_id === DIRECTED_TEACHING && p.authorization_reference === auth.reference && p.teaching_fact === key && typeof p.split_family === "string" && pythonStrip(p.split_family) && row.answer === value && check.type === "exact" && check.expected === value, "教学材料必须绑定授权事实、独立问法家族与准确短答案"); }
    }
  }
  need(["training", "validation"].every(p => groups[("new_" + p) as keyof EvolutionGroups].every(r => covered[p]!.has(r.id))), "全部NEW材料必须属于本次明确授权的模型身份与定位");
}
export function uniqueMaterialRecords(groups: Partial<EvolutionGroups>, protection: Row[], regressions: Row[]) {
  const rows: Row[] = [], ids = new Map<string, Row>();
  for (const row of [...Object.values(groups).flat(), ...protection, ...regressions]) {
    const prior = ids.get(row.id);
    if (prior) { need(same(materialMessages(prior), materialMessages(row)) && prior.answer === row.answer && same(materialProvenance(prior).generation_check ?? null, materialProvenance(row).generation_check ?? null), "部分进化同一ID有冲突材料"); continue; }
    ids.set(row.id, row); rows.push(clone(row));
  }
  return rows;
}
export function validateNativeProductRecords(records: any) {
  const common = ["groups", "protection", "retention_regressions"];
  need(exactKeys(records, [...common, "experience"]) || exactKeys(records, [...common, "directed_teaching"]), "产品450轮需要互斥的冻结350经验或明确授权定向教学");
  validateMaterialGroups(records.groups); need(records.groups.new_training.length >= 1 && records.groups.new_training.length <= 57, "产品固定450轮只训练至多57条待学新内容");
  need(Array.isArray(records.protection) && Array.isArray(records.retention_regressions), "保护材料必须为列表");
  for (const row of records.protection) { validateMaterialRecord(row, materialProvenance(row).retention_generation_only === true); need(row.purpose === "validation", "保护材料必须保持检验拆分"); }
  for (const row of records.retention_regressions) validateMaterialRecord(row, true);
  uniqueMaterialRecords(records.groups, records.protection, records.retention_regressions);
  if (Object.hasOwn(records, "directed_teaching")) { validateDirectedTeaching(records.directed_teaching, records.groups); return; }
  const pool = records.experience; need(Array.isArray(pool) && pool.length === 350, "产品进化需要350条有效去重新内容");
  const ids = new Set(), contexts = new Set(), allowed = new Set();
  for (const row of pool) { validateMaterialRecord(row); const c = materialContext(row); need(!ids.has(row.id) && !contexts.has(c) && materialProvenance(row).evolution_role === "new", "经验材料重复或不是新知识"); ids.add(row.id); contexts.add(c); allowed.add(encoded(row)); }
  need([...records.groups.new_training, ...records.groups.new_validation].every(row => allowed.has(encoded(row))), "本轮新知识不属于已冻结经验材料");
}
const materialKey = (row: Row) => { const p = { ...materialProvenance(row) }; for (const k of ["evolution_role", "retention_from_version", "retention_scope"]) delete p[k]; return encoded({ ...Object.fromEntries(["id", "prompt", "answer", "source", "purpose"].map(k => [k, row[k] ?? null])), provenance: p }); };
const experienceKey = (row: Row) => encoded({ context: materialContext(row), answer: row.answer, check: validateGenerationCheck(materialProvenance(row).generation_check, row.answer) });
export type NativeMaterialInput = {
  identity: string; activeCore: string | null; originalTensorSHA256?: string; originalProtection?: any;
  retention?: any; samples: Row[]; directedTeaching?: any; retentionRegressions?: Row[]; runs?: Row[];
};
export function collectNativeEvolutionMaterial(input: NativeMaterialInput) {
  need(typeof input.identity === "string" && input.identity && Array.isArray(input.samples), "进化材料缺少基座身份或样本");
  let snapshot: any = null, protection: Row[];
  if (input.activeCore === null) protection = originalCoreProtectionRows(input.originalProtection, input.identity, input.originalTensorSHA256 ?? "");
  else { snapshot = validateNativeCoreRetention(input.retention, input.identity, input.activeCore); protection = clone([...snapshot.groups.new_validation, ...snapshot.groups.replay_validation, ...snapshot.regression_validation]); }
  const directed = input.directedTeaching !== undefined;
  let groups = consolidateMaterialGroups(input.samples, snapshot, !directed), experience: Row[] | null = null, unspentCount = 0;
  if (!directed) {
    const consumed = new Set<string>();
    for (const run of input.runs ?? []) {
      if (run.kind !== "evolution" || run.scope === "research" && run.product_accepted !== true || !object(run.records) || !Object.hasOwn(run.records, "experience")) continue;
      need(run.binding?.identity === input.identity && materialContentSHA(run.records) === run.binding?.records_sha256, "进化经验账本绑定不完整，等待恢复"); validateNativeProductRecords(run.records);
      const keys = new Set(run.records.experience.map(materialKey)); for (const row of run.records.experience) consumed.add(experienceKey(row));
      if (!experience && run.records.experience.every(dailyLearningMaterial) && groups.new_training.some(row => keys.has(materialKey(row)))) experience = clone(run.records.experience);
    }
    const unspent = [...groups.new_training, ...groups.new_validation].filter(row => !consumed.has(experienceKey(row))); unspentCount = unspent.length;
    if (!experience && unspent.length >= 350) experience = unspent.sort((a, b) => order(a.id, b.id)).slice(0, 350);
    need(experience, "等待350条有效去重新内容；已有经验不会因重试增加");
    const keys = new Set(experience.map(materialKey)); groups = { ...groups, new_training: groups.new_training.filter(r => keys.has(materialKey(r))).slice(0, 57), new_validation: groups.new_validation.filter(r => keys.has(materialKey(r))) };
  }
  // Snapshot regressions remain in protection; additional explicit boundary cases stay read-only.
  const records = { groups, protection, retention_regressions: clone(input.retentionRegressions ?? []), ...(directed ? { directed_teaching: clone(input.directedTeaching) } : { experience }) };
  validateNativeProductRecords(records);
  if (snapshot) {
    const replayIDs = new Set([...groups.replay_training, ...groups.replay_validation].map(row => row.id));
    need((Object.values(snapshot.groups).flat() as Row[]).every(row => replayIDs.has(row.id)), "冻结核心材料遗漏已采纳知识的重放保护");
    retentionConflicts(snapshot, Object.values(groups).flat());
  }
  need(groups.new_training.length && groups.replay_training.length && groups.replay_validation.length, "冻结核心材料需要完整的新知识与旧能力训练/检验");
  return { records, unspent_count: unspentCount, training_started: false as const, quality_admitted: false as const };
}

// Names shared with the native quality observer; keep one material contract.
export { materialProvenance as provenance, materialMessages as messages };
export function uniqueNativeEvolutionRecords(groups: Partial<EvolutionGroups>, protection: Row[] = [], regressions: Row[] = []) {
  return uniqueMaterialRecords(groups, protection, regressions);
}
