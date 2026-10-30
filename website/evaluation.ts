import { Database } from "bun:sqlite";
import { existsSync, readFileSync } from "node:fs";

export class EvaluationError extends Error {
  constructor(message: string, readonly status = 400) { super(message); }
}
type Question = { id: string; kind: "benchmark" | "generalization"; prompt: string; answer: unknown; scoring: "choice" | "exact_number" | "json" };
type Suite = { suite_id: string; policy: Record<string, unknown>; questions: Question[] };
type Answer = { id: string; text: string; finish_reason: string };
type Receipt = { receipt_id: string; suite_id: string; suite_sha256: string; policy_hash: string; model_version: string; benchmark: number; generalization: number; benchmark_delta: number | null; generalization_delta: number | null; completed_at: string };
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(",")}}`;
  return JSON.stringify(value);
}
const hash = (value: string) => new Bun.CryptoHasher("sha256").update(value).digest("hex");
const version = (value: unknown): value is string => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
const exactKeys = (value: unknown, keys: string[]): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));

export function createEvaluations(db: Database, suitePath?: string) {
  const path = suitePath || process.env.LEADERBOARD_SUITE || (existsSync(`${import.meta.dir}/ranking-v1.json`) ? `${import.meta.dir}/ranking-v1.json` : `${import.meta.dir}/../benchmarks/ranking-v1.json`);
  const raw = readFileSync(path, "utf8");
  const suite = JSON.parse(raw) as Suite;
  if (suite.suite_id !== "luban-ranking-v1" || !suite.policy || !Array.isArray(suite.questions) || suite.questions.length !== 48 || new Set(suite.questions.map(q => q.id)).size !== 48 || ["benchmark", "generalization"].some(kind => suite.questions.filter(q => q.kind === kind).length !== 24) || suite.questions.some(q => !q.id || !q.prompt || !["choice", "exact_number", "json"].includes(q.scoring) || q.answer === undefined)) throw new Error("排行评测题库无效");
  for (const [key,value] of Object.entries({temperature:0,seed:42,max_output_tokens:512,route:"local_core",teacher_assistance:false,thinking:"off",shots:0})) {
    if (suite.policy[key] !== value) throw new Error("排行评测固定条件无效");
  }
  if (typeof suite.policy.system_prompt !== "string" || !suite.policy.system_prompt) throw new Error("排行评测系统提示缺失");
  const policy_hash = hash(canonical(suite.policy)), suite_sha256 = hash(raw);
  db.exec(`CREATE TABLE IF NOT EXISTS evaluation_challenges (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, model_version TEXT NOT NULL,
    suite_id TEXT NOT NULL, suite_sha256 TEXT NOT NULL, policy_hash TEXT NOT NULL,
    expires_at INTEGER NOT NULL, created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS evaluation_challenge_owner ON evaluation_challenges(owner,model_version);
  CREATE TABLE IF NOT EXISTS evaluation_receipts (
    id TEXT PRIMARY KEY, challenge_id TEXT NOT NULL UNIQUE, owner TEXT NOT NULL, model_version TEXT NOT NULL,
    suite_id TEXT NOT NULL, suite_sha256 TEXT NOT NULL, policy_hash TEXT NOT NULL,
    benchmark REAL NOT NULL, generalization REAL NOT NULL, benchmark_delta REAL, generalization_delta REAL,
    answer_hash TEXT NOT NULL, completed_at TEXT NOT NULL,
    UNIQUE(owner,model_version,suite_id,suite_sha256,policy_hash)
  );`);
  const receiptSelect = "SELECT id AS receipt_id,suite_id,suite_sha256,policy_hash,model_version,benchmark,generalization,benchmark_delta,generalization_delta,completed_at FROM evaluation_receipts";
  function currentReceipt(owner: string, model: string) {
    return db.query<Receipt, string[]>(`${receiptSelect} WHERE owner=? AND model_version=? AND suite_id=? AND suite_sha256=? AND policy_hash=?`).get(owner, model, suite.suite_id, suite_sha256, policy_hash);
  }
  function challenge(owner: string, body: unknown) {
    if (!exactKeys(body, ["model_version"]) || !version(body.model_version)) throw new EvaluationError("评测需要有效的当前模型版本");
    const model = body.model_version;
    const existing = currentReceipt(owner, model);
    if (existing) return { receipt: existing, ...metadata };
    const now = Date.now();
    let row = db.query<{ id: string; expires_at: number }, string[]>("SELECT id,expires_at FROM evaluation_challenges WHERE owner=? AND model_version=? AND suite_id=? AND suite_sha256=? AND policy_hash=? AND expires_at>? ORDER BY created_at DESC LIMIT 1").get(owner, model, suite.suite_id, suite_sha256, policy_hash, String(now));
    if (!row) {
      row = { id: hash(`${crypto.randomUUID()}:${crypto.randomUUID()}`), expires_at: now + 2 * 60 * 60 * 1000 };
      db.query("INSERT INTO evaluation_challenges VALUES (?,?,?,?,?,?,?,?)").run(row.id, owner, model, suite.suite_id, suite_sha256, policy_hash, row.expires_at, new Date(now).toISOString());
    }
    return { challenge_id: row.id, ...metadata, policy: suite.policy, model_version: model, expires_at: new Date(row.expires_at).toISOString(), questions: suite.questions.map(({ id, kind, prompt }) => ({ id, kind, prompt })) };
  }
  function correct(question: Question, answer: Answer) {
    if (answer.finish_reason !== "stop") return false;
    const text = answer.text.normalize("NFKC").trim();
    if (question.scoring === "choice") return /^[A-J]$/i.test(text) && text.toUpperCase() === String(question.answer).toUpperCase();
    if (question.scoring === "exact_number") return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text) && Number.isFinite(Number(text)) && Number(text) === Number(question.answer);
    try { return canonical(JSON.parse(text)) === canonical(question.answer); } catch { return false; }
  }
  const submit = db.transaction((owner: string, body: unknown) => {
    if (!exactKeys(body, ["challenge_id", "model_version", "answers"]) || !version(body.challenge_id) || !version(body.model_version) || !Array.isArray(body.answers) || body.answers.length !== 48) throw new EvaluationError("评测答卷字段无效，需要完整48题");
    const row = db.query<{ owner: string; model_version: string; suite_id: string; suite_sha256: string; policy_hash: string; expires_at: number }, [string]>("SELECT * FROM evaluation_challenges WHERE id=?").get(body.challenge_id);
    if (!row || row.owner !== owner || row.model_version !== body.model_version) throw new EvaluationError("评测挑战与安装或模型版本不匹配", 409);
    if (row.suite_id !== suite.suite_id || row.suite_sha256 !== suite_sha256 || row.policy_hash !== policy_hash) throw new EvaluationError("评测题库已更新，请重新测试", 409);
    const questionIds = new Set(suite.questions.map(q => q.id));
    const answers: Answer[] = body.answers.map(input => {
      if (!exactKeys(input, ["id", "text", "finish_reason"]) || typeof input.id !== "string" || !questionIds.delete(input.id) || typeof input.text !== "string" || new TextEncoder().encode(input.text).length > 8192 || typeof input.finish_reason !== "string" || input.finish_reason.length > 32) throw new EvaluationError("评测答案包含重复题、未知题或无效内容");
      return input as Answer;
    });
    const answer_hash = hash(canonical(answers.toSorted((a,b) => a.id.localeCompare(b.id))));
    const previousReceipt = db.query<{ answer_hash: string }, [string]>("SELECT answer_hash FROM evaluation_receipts WHERE challenge_id=?").get(body.challenge_id);
    if (previousReceipt) {
      if (previousReceipt.answer_hash !== answer_hash) throw new EvaluationError("已完成评测不可修改或重复刷分", 409);
      return currentReceipt(owner, body.model_version)!;
    }
    if (row.expires_at <= Date.now()) throw new EvaluationError("评测挑战已过期，请重新测试", 409);
    if (currentReceipt(owner, body.model_version)) throw new EvaluationError("此模型版本已经完成评测，不能重复刷分", 409);
    const byId = new Map(answers.map(answer => [answer.id, answer]));
    const scores = ["benchmark", "generalization"].map(kind => Math.round(10000 * suite.questions.filter(q => q.kind === kind && correct(q, byId.get(q.id)!)).length / 24) / 100);
    const previous = db.query<Receipt, string[]>(`${receiptSelect} WHERE owner=? AND model_version<>? AND suite_id=? AND suite_sha256=? AND policy_hash=? ORDER BY completed_at DESC,id DESC LIMIT 1`).get(owner, body.model_version, suite.suite_id, suite_sha256, policy_hash);
    const receipt: Receipt = { receipt_id: hash(`${body.challenge_id}:${answer_hash}`), ...metadata, model_version: body.model_version, benchmark: scores[0], generalization: scores[1], benchmark_delta: previous ? Math.round((scores[0] - previous.benchmark) * 100) / 100 : null, generalization_delta: previous ? Math.round((scores[1] - previous.generalization) * 100) / 100 : null, completed_at: new Date().toISOString() };
    db.query("INSERT INTO evaluation_receipts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)").run(receipt.receipt_id, body.challenge_id, owner, receipt.model_version, receipt.suite_id, receipt.suite_sha256, receipt.policy_hash, receipt.benchmark, receipt.generalization, receipt.benchmark_delta, receipt.generalization_delta, answer_hash, receipt.completed_at);
    return receipt;
  });
  const metadata = { suite_id: suite.suite_id, suite_sha256, policy_hash };
  function validatePublication(owner: string, model: unknown, id: unknown) {
    if (!version(model) || !version(id)) throw new EvaluationError("发布评分需要当前模型版本与服务器评测凭据");
    const receipt = currentReceipt(owner, model);
    if (!receipt || receipt.receipt_id !== id) throw new EvaluationError("评测凭据不属于此安装或当前模型版本", 409);
    return receipt;
  }
  return { challenge, submit: (owner: string, body: unknown) => submit.immediate(owner, body), validatePublication, metadata };
}
