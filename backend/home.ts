import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { Memory, ledgerParse } from "./memory.ts";
import { privateJSON } from "./settings.ts";

export function modelName(value: unknown): string {
  if (typeof value !== "string" || /[\p{Cc}\p{Cf}\p{Cs}]/u.test(value)) throw new TypeError("模型名称需要不含控制字符的文本");
  const name = value.trim();
  if (!name || Array.from(name).length > 32) throw new TypeError("模型名称需要 1–32 个字符");
  return name;
}

function names(home: string): Record<string, string> {
  const path = join(home, "model-names.json");
  try { if (!lstatSync(path).isFile()) throw new Error("模型昵称需要普通私有文件"); }
  catch (error: any) { if (error.code === "ENOENT") return {}; throw error; }
  const data = JSON.parse(readFileSync(path, "utf8"));
  if (!data || typeof data !== "object" || Array.isArray(data)
      || Object.entries(data).some(([identity, name]) => !/^[a-f0-9]{64}$/.test(identity) || modelName(name) !== name)) throw new Error("模型昵称记录损坏，请检查私有设置目录");
  return data;
}

export function readModelName(home: string, identity: string | null): string {
  return identity ? names(home)[identity] ?? "luban0" : "luban0";
}

export function saveModelName(home: string, identity: string, value: unknown): string {
  if (!/^[a-f0-9]{64}$/.test(identity)) throw new Error("当前模型身份尚未就绪");
  const name = modelName(value);
  privateJSON(join(home, "model-names.json"), { ...names(home), [identity]: name });
  return name;
}

function boundaries(now: Date) {
  const today = new Date(now); today.setHours(0, 0, 0, 0);
  const week = new Date(today); week.setDate(week.getDate() - (week.getDay() + 6) % 7);
  return { today: today.getTime() / 1000, week: week.getTime() / 1000, end: now.getTime() / 1000 };
}

// Report only recorded model calls and actual teacher opinions. A teacher's
// "wrong" opinion does not prove correctness, repair, or new neural mastery.
export async function homeLedger(store: Memory, identity: string | null, now = new Date()) {
  const empty = { local_calls: 0, teacher_calls: 0, feedback_received: 0, teacher_flagged_gaps: 0, corrections: null, mastered: null };
  const result = { today: { ...empty }, week: { ...empty }, application_attribution: false,
    evolution_count: 0, recent: [] as Array<{ id: string; kind: string; created: number; promoted: boolean }>,
    last_evolution: null as null | { created: number; promoted: boolean; progress: any } };
  if (!identity) return result;
  const begin = await store.dispatch("db_begin", [], {}, identity);
  const transaction = begin.transaction;
  const rows = async (sql: string, parameters: any[] = []) => {
    const response = await store.dispatch("db_execute", [], { sql, parameters }, identity, transaction);
    return response.rows.map((row: any[]) => Object.fromEntries(response.columns.map((name: string, index: number) => [name, row[index]])));
  };
  try {
    const times = boundaries(now);
    for (const period of ["today", "week"] as const) {
      const calls = await rows(`SELECT json_extract(body,'$.decision.route') AS route,COUNT(*) AS n FROM events
        WHERE kind='decision' AND created>=? AND created<=? AND json_extract(body,'$.base_identity')=? GROUP BY route`, [times[period], times.end, identity]);
      for (const row of calls) if (row.route === "local" || row.route === "teacher") result[period][row.route === "local" ? "local_calls" : "teacher_calls"] = Number(row.n);
      const opinions = await rows(`SELECT COUNT(DISTINCT json_extract(r.body,'$.candidate_id')) AS n,
        COUNT(DISTINCT CASE WHEN json_extract(r.body,'$.status')='complete' AND json_extract(r.body,'$.result.correct')=0
          THEN json_extract(r.body,'$.candidate_id') END) AS gaps FROM events r
        WHERE r.kind='teacher_review' AND r.created>=? AND r.created<=? AND (
          EXISTS(SELECT 1 FROM events d WHERE d.kind='decision' AND 'route:'||d.id=json_extract(r.body,'$.candidate_id')
            AND json_extract(d.body,'$.base_identity')=?)
          OR EXISTS(SELECT 1 FROM samples s WHERE 'material:'||s.id=json_extract(r.body,'$.candidate_id')
            AND json_extract(s.provenance,'$.base_identity')=?))
        AND json_extract(r.body,'$.status') IN ('complete','pending') AND json_type(r.body,'$.result.correct') IN ('true','false')`,
      [times[period], times.end, identity, identity]);
      result[period].feedback_received = Number(opinions[0]?.n ?? 0);
      result[period].teacher_flagged_gaps = Number(opinions[0]?.gaps ?? 0);
    }
    const hasJournal = (await rows("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'")).length > 0;
    // A shipped persona core has a durable product journal too, but is never
    // a personal evolution. Do not inherit earlier bases or research candidates.
    if (hasJournal) {
      const runs = await rows(`SELECT COUNT(*) AS n FROM training_runs t WHERE identity=? AND kind='evolution' AND state='committed'
        AND (COALESCE(json_extract(body,'$.scope'),'product')<>'research' OR json_extract(body,'$.product_accepted')=1)
        AND json_extract(body,'$.commit.report.promoted')=1 AND json_extract(body,'$.commit.report.publication_state')='published'
        AND json_extract(body,'$.commit.report.identity')=?
        AND COALESCE(json_extract(body,'$.commit.report.foundation_install'),0)<>1
        AND json_type(body,'$.commit.report.native_identity_verified') IS NULL
        AND COALESCE(json_extract(body,'$.commit.report.training_updates_executed'),1)<>0
        AND EXISTS(SELECT 1 FROM events e WHERE e.id='training:'||t.id AND e.kind='evolution'
          AND json_extract(e.body,'$.identity')=? AND json_extract(e.body,'$.promoted')=1
          AND json_extract(e.body,'$.version')=json_extract(t.body,'$.commit.report.version'))`, [identity, identity, identity]);
      result.evolution_count = Number(runs[0]?.n ?? 0);
    }
    const events = await rows(`SELECT id,kind,created,body FROM events WHERE kind IN ('learning','evolution','evolution_failure')
      AND COALESCE(json_extract(body,'$.identity'),json_extract(body,'$.base_identity'))=? AND COALESCE(json_extract(body,'$.foundation_install'),0)<>1
      AND json_type(body,'$.native_identity_verified') IS NULL ORDER BY created DESC LIMIT 8`, [identity]);
    const last = (await rows(`SELECT id,created,body FROM events WHERE kind IN ('evolution','evolution_failure')
      AND json_extract(body,'$.identity')=? AND COALESCE(json_extract(body,'$.foundation_install'),0)<>1
      AND json_type(body,'$.native_identity_verified') IS NULL ORDER BY created DESC LIMIT 1`, [identity]))[0];
    const runIds = [...events, ...(last ? [last] : [])].filter((row: any) => row.id.startsWith("training:")).map((row: any) => row.id.slice(9));
    const committed = hasJournal && runIds.length ? await rows(`SELECT id FROM training_runs WHERE identity=? AND state='committed'
      AND (COALESCE(json_extract(body,'$.scope'),'product')<>'research' OR json_extract(body,'$.product_accepted')=1)
      AND id IN (${runIds.map(() => "?").join(",")})`, [identity, ...runIds]) : [];
    const adopted = new Set(committed.map((row: any) => "training:" + row.id));
    result.recent = events.map((row: any) => ({ id: row.id, kind: row.kind, created: Number(row.created), promoted: adopted.has(row.id) && ledgerParse(row.body).promoted === true }));
    if (last) result.last_evolution = { created: Number(last.created), promoted: adopted.has(last.id) && ledgerParse(last.body).promoted === true, progress: null };
    return result;
  } finally { await store.dispatch("db_close", [], {}, identity, transaction); }
}
