/** Bun owns the existing SQLite ledger; model code talks to these methods over private RPC. */
import { Database, type SQLQueryBindings } from "bun:sqlite";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, lstatSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { requireStorage } from "./settings";
import { floatSource, floatValue, parseJSON } from "./wire";
import {initializeMemoryDatabase} from './memory-database';

type RecordValue = Record<string, any>;
type Scalar = string | number | bigint | null | Uint8Array;
type Message = { role: string; content: string };
const SCHEMA = `
CREATE TABLE IF NOT EXISTS samples (
 id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL,
 source TEXT NOT NULL, purpose TEXT NOT NULL, created REAL NOT NULL,
 approved INTEGER NOT NULL, training_allowed INTEGER NOT NULL, provenance TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, messages TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage (
 route TEXT PRIMARY KEY, calls INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL,
 completion_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL, estimated_calls INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS teacher_materials (
 id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL, messages TEXT NOT NULL,
 created REAL NOT NULL, provenance TEXT NOT NULL, status TEXT NOT NULL,
 reason TEXT NOT NULL, purpose TEXT NOT NULL, sample_id TEXT);
`;
// Python str.isspace includes ASCII separators U+001C–001F and excludes JS's BOM.
const PY_WHITESPACE = "\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const STRIP = new RegExp(`^[${PY_WHITESPACE}]+|[${PY_WHITESPACE}]+$`, "g");
const SPACE = new RegExp(`[${PY_WHITESPACE}]+`, "g");
export const pythonStrip = (value: string) => value.replace(STRIP, "");
const unicode = (value: string): string => {
    if (/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(value)) throw new TypeError("文本包含无法编码的 Unicode 代理字符");
    return value;
};
const length = (value: string) => Array.from(value).length;
let lastTimestamp = 0;
const now = () => lastTimestamp = Math.max((performance.timeOrigin + performance.now()) / 1000, lastTimestamp + 0.000001);
const id = () => randomUUID().replaceAll("-", "");
const integer = (value: bigint | number) => typeof value === "bigint" && value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value;
const normalizeRows = (rows: RecordValue[]) => rows.map(row => Object.fromEntries(Object.entries(row).map(([key, value]) => [key, typeof value === "bigint" ? integer(value) : value])));
const truthy = (value: any): boolean => value instanceof Number ? !!value.valueOf() : Array.isArray(value) ? value.length > 0 : value && typeof value === "object" ? Object.keys(value).length > 0 : !!value;

/** JSON values retain Python integer precision in persisted provenance and event bodies. */
export function ledgerStringify(value: any): string {
    if (typeof value === "bigint") return value.toString();
    if (value instanceof Number) {
        const number = value.valueOf();
        if (!Number.isFinite(number)) throw new TypeError("账本数值需要有限数值");
        return floatSource(value);
    }
    if (value === null || typeof value !== "object") {
        if (typeof value === "string") unicode(value);
        const encoded = JSON.stringify(value);
        if (encoded === undefined) throw new TypeError("账本内容不能包含未定义值");
        return encoded;
    }
    if (Array.isArray(value)) return "[" + value.map(ledgerStringify).join(", ") + "]";
    return "{" + Object.entries(value).map(([key, item]) => JSON.stringify(unicode(key)) + ": " + ledgerStringify(item)).join(", ") + "}";
}
export const ledgerParse = parseJSON;

function rejectLink(path: string, message = "存储目录和数据库不能是符号链接") {
    try { if (lstatSync(path).isSymbolicLink()) throw new Error(message); }
    catch (error: any) { if (error.code !== "ENOENT") throw error; }
}
function prepareStorage(root: string, readonly: boolean) {
    if (root.startsWith("/Volumes/")) {
        const mount = "/Volumes/" + root.split("/")[2];
        if (!existsSync(mount) || statSync(mount).dev === statSync(dirname(mount)).dev) throw new Error("存储盘未连接，请连接后重试");
    }
    // macOS /tmp and /var are system links; validate the configured root and its writable children.
    rejectLink(root);
    if (!readonly) requireStorage(root);
    rejectLink(join(root, "app"));
    if (!readonly) mkdirSync(join(root, "app"), { recursive: true });
    for (const suffix of ["", "-journal", "-wal", "-shm"]) rejectLink(join(root, "app/memory.sqlite3") + suffix);
}

/** Bun binds integral JS Numbers as INTEGER; Python float parameters must remain REAL. */
function floatParameters(sql: string, parameters: Scalar[] | Record<string, Scalar>): string {
    let index = 0;
    const named = new Map<string, number>();
    return sql.replace(/'(?:''|[^'])*'|"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]*\]|--[^\n]*(?:\n|$)|\/\*[\s\S]*?\*\/|\?(?:\d+)?|[:@$][\p{L}\p{N}_]+/gu, token => {
        if (!/^[?:@$]/.test(token)) return token;
        let position: number;
        if (token === "?") position = ++index;
        else if (/^\?\d+$/.test(token)) { position = Number(token.slice(1)); index = Math.max(index, position); }
        else { position = named.get(token) ?? ++index; named.set(token, position); }
        const value = Array.isArray(parameters) ? parameters[position - 1] : parameters[token.slice(1)];
        return value instanceof Number ? `CAST(${token} AS REAL)` : token;
    });
}

export class Memory {
    readonly root: string;
    readonly path: string;
    model_identity: string | null = null;
    private lease?: { token: string; db: Database; identity: string | null; readonly: boolean; timer: ReturnType<typeof setTimeout> };
    private waiters = new Set<() => void>();
    private closed = false;
    constructor(storage_root: string, private readonly leaseTimeoutMs = 60_000) {
        if (typeof storage_root !== "string" || !isAbsolute(storage_root)) throw new Error("存储目录必须是绝对路径");
        this.root = resolve(storage_root);
        this.path = join(this.root, "app/memory.sqlite3");
    }
    private open(readonly = false) {
        if (this.closed) throw new Error("数据服务已关闭");
        prepareStorage(this.root, readonly);
        const db = new Database(this.path, { readonly, create: !readonly, safeIntegers: true, strict: true });
        try {
            initializeMemoryDatabase(db,readonly,SCHEMA);
            return db;
        } catch (error) { db.close(); throw error; }
    }
    /** Synchronous operations run to completion in Bun's event loop; RPC waits for any SQL lease. */
    connect<T>(operation: (db: Database) => T): T {
        if (this.lease) throw new Error("数据库事务正在使用中");
        const db = this.open();
        try {
            db.exec("BEGIN IMMEDIATE");
            const result = operation(db);
            if (result && typeof (result as any).then === "function") throw new Error("数据库同步事务不能返回 Promise");
            db.exec("COMMIT");
            return result;
        } catch (error) { if (db.inTransaction) db.exec("ROLLBACK"); throw error; }
        finally { db.close(); }
    }
    growth_reserved(messages: Message[]): boolean {
        const path = join(this.root, "app/growth/rounds.json");
        rejectLink(path, "个人成长检验资料不能是符号链接");
        rejectLink(dirname(path), "个人成长检验资料不能是符号链接");
        if (!existsSync(path)) return false;
        const reserved = ledgerParse(readFileSync(path, "utf8")).reserved_prompts;
        if (!Array.isArray(reserved) || reserved.some(prompt => typeof prompt !== "string")) throw new Error("成长题隔离记录损坏，暂停资料入库");
        const normalize = (value: string) => value.normalize("NFKC").replace(SPACE, "");
        const prompts = new Set(reserved.map(normalize));
        return messages.some(message => message.role === "user" && prompts.has(normalize(message.content ?? "")));
    }
    add_sample(prompt: string, answer: string, source = "human", purpose = "training", approved = false, training_allowed = false, provenance: RecordValue | null = null): string {
        if (!["training", "validation"].includes(purpose) || !["human", "teacher", "synthetic"].includes(source)) throw new Error("学习材料类型无效");
        if (typeof prompt !== "string" || typeof answer !== "string" || !pythonStrip(prompt) || !pythonStrip(answer) || length(prompt) + length(answer) > 12000) throw new Error("请填写问题和正确答案，总长度不超过 12000 字符");
        unicode(prompt); unicode(answer);
        if (!truthy(approved) || !truthy(training_allowed)) throw new Error("学习材料需要确认正确且允许训练");
        const sample_id = id();
        provenance = { ...(provenance || {}) };
        if (this.growth_reserved(truthy(provenance.messages) ? provenance.messages : [{ role: "user", content: prompt }])) throw new Error("成长评测题只能检验，不能作为学习材料");
        const identity = this.model_identity;
        if (identity) provenance.base_identity = identity;
        const scope = identity ? " AND json_extract(provenance, '$.base_identity')=?" : "";
        const params = identity ? [identity] : [];
        this.connect(db => {
            if (db.query("SELECT id FROM samples WHERE prompt=? AND purpose<>?" + scope).get(pythonStrip(prompt), purpose, ...params)) throw new Error("同一问题不能同时用于训练和检验，请使用独立的新问题");
            db.query("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)").run(sample_id, pythonStrip(prompt), pythonStrip(answer), source, purpose, now(), 1, 1, ledgerStringify(provenance));
        });
        return sample_id;
    }
    samples(purpose: string | null = null): RecordValue[] {
        const records = this.connect(db => normalizeRows(db.query("SELECT * FROM samples" + (purpose ? " WHERE purpose=?" : "") + " ORDER BY created DESC").all(...(purpose ? [purpose] : [])) as RecordValue[]));
        return records.filter(row => (!this.model_identity || ledgerParse(row.provenance).base_identity === this.model_identity) && !this.growth_reserved(truthy(ledgerParse(row.provenance).messages) ? ledgerParse(row.provenance).messages : [{ role: "user", content: row.prompt }]));
    }
    delete_sample(sample_id: string): void {
        this.connect(db => {
            db.query("DELETE FROM samples WHERE id=?").run(sample_id);
            db.query("UPDATE teacher_materials SET status='pending',reason='学习材料已移除',sample_id=NULL WHERE sample_id=?").run(sample_id);
        });
    }
    record_usage(route: string, usage: RecordValue): void {
        if (!["local", "teacher"].includes(route)) throw new Error("统计来源无效");
        const keys = ["prompt_tokens", "completion_tokens", "total_tokens"];
        const unknown = usage.unknown === true;
        if (unknown && !keys.every(key => key in usage && usage[key] === null)) throw new Error("未知词元统计需要显式空值");
        const counts = keys.map(key => unknown ? 0 : key in usage ? usage[key] : 0);
        if (counts.some(n => typeof n !== "bigint" && (typeof n !== "number" || !Number.isSafeInteger(n)) || n < 0)) throw new Error("词元统计需要非负整数");
        if (counts.some(n => BigInt(n) > 9223372036854775807n)) throw new RangeError("Python int too large to convert to SQLite INTEGER");
        this.connect(db => {
            db.query("INSERT INTO usage VALUES (?,?,?,?,?,?) ON CONFLICT(route) DO UPDATE SET calls=calls+1,prompt_tokens=prompt_tokens+excluded.prompt_tokens,completion_tokens=completion_tokens+excluded.completion_tokens,total_tokens=total_tokens+excluded.total_tokens,estimated_calls=estimated_calls+excluded.estimated_calls").run(route, 1, ...counts, Number(truthy(usage.estimated)));
            if (unknown) db.query("INSERT INTO events VALUES (?,?,?,?)").run(id(), "usage_unknown", ledgerStringify({ route }), now());
        });
    }
    usage(): RecordValue {
        const result: RecordValue = Object.fromEntries(["local", "teacher"].map(route => [route, { calls: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, estimated_calls: 0 }]));
        this.connect(db => {
            for (const { route, ...record } of normalizeRows(db.query("SELECT * FROM usage").all() as RecordValue[])) result[route] = record;
            for (const row of normalizeRows(db.query("SELECT json_extract(body,'$.route') AS route,COUNT(*) AS count FROM events WHERE kind='usage_unknown' GROUP BY route").all() as RecordValue[])) {
                if (row.route in result) Object.assign(result[row.route], { prompt_tokens: null, completion_tokens: null, total_tokens: null, unknown_calls: row.count });
            }
        });
        return result;
    }
    capture_teacher(messages: Message[], answer: string, provenance: RecordValue, eligible: boolean, reason = ""): string {
        const prompt = messages.findLast(message => message.role === "user")?.content ?? "";
        unicode(prompt); unicode(answer); unicode(reason);
        eligible = !!(truthy(eligible) && pythonStrip(prompt) && pythonStrip(answer) && length(prompt) + length(answer) <= 12000);
        if (this.growth_reserved(messages)) { eligible = false; reason = "成长评测题只用于检验，不进入学习"; }
        let purpose = BigInt("0x" + createHash("sha256").update(pythonStrip(prompt)).digest("hex")) % 5n === 0n ? "validation" : "training";
        const material_id = id();
        provenance = { ...provenance, messages, verification_scope: "structural checks only; no factual truth guarantee" };
        const identity = this.model_identity;
        if (identity) provenance.base_identity = identity;
        const scope = identity ? " AND json_extract(provenance, '$.base_identity')=?" : "";
        const params = identity ? [identity] : [];
        const encoded = ledgerStringify(provenance);
        this.connect(db => {
            let sample_id: string | null = null;
            if (eligible) {
                const previous = db.query("SELECT id,answer,purpose FROM samples WHERE prompt=?" + scope + " ORDER BY created LIMIT 1").get(pythonStrip(prompt), ...params) as RecordValue | null;
                if (previous) purpose = previous.purpose;
                const duplicate = db.query("SELECT id FROM samples WHERE prompt=? AND answer=?" + scope + (identity ? " AND approved=1 AND training_allowed=1" : "")).get(pythonStrip(prompt), pythonStrip(answer), ...params) as RecordValue | null;
                if (duplicate) sample_id = duplicate.id;
                else {
                    sample_id = id();
                    db.query("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)").run(sample_id, pythonStrip(prompt), pythonStrip(answer), "teacher", purpose, now(), 1, 1, encoded);
                }
            }
            db.query("INSERT INTO teacher_materials VALUES (?,?,?,?,?,?,?,?,?,?)").run(material_id, prompt, answer, ledgerStringify(messages), now(), encoded, eligible ? "eligible" : "pending", reason, purpose, sample_id);
        });
        return material_id;
    }
    teacher_materials(): RecordValue[] { return this.connect(db => normalizeRows(db.query("SELECT * FROM teacher_materials ORDER BY created DESC LIMIT 100").all() as RecordValue[])); }
    material_counts(): RecordValue {
        return this.connect(db => {
            const counts = Object.fromEntries(normalizeRows(db.query("SELECT status,COUNT(*) AS n FROM teacher_materials GROUP BY status").all() as RecordValue[]).map(row => [row.status, row.n])) as Record<string, number>;
            return { total: Object.values(counts).reduce((a, b) => a + b, 0), pending: counts.pending || 0, eligible: counts.eligible || 0 };
        });
    }
    save_conversation(conversation_id: string | null, messages: Message[]): string {
        conversation_id ||= id();
        const first = messages.find(message => message.role === "user");
        const title = first ? Array.from(first.content).slice(0, 40).join("") : "新对话";
        this.connect(db => db.query("INSERT OR REPLACE INTO conversations VALUES (?,?,?,?)").run(conversation_id, title, ledgerStringify(messages), now()));
        return conversation_id;
    }
    conversations(): RecordValue[] { return this.connect(db => normalizeRows(db.query("SELECT id,title,updated FROM conversations ORDER BY updated DESC LIMIT 50").all() as RecordValue[])); }
    conversation(conversation_id: string): RecordValue | null {
        const row = this.connect(db => db.query("SELECT * FROM conversations WHERE id=?").get(conversation_id) as RecordValue | null);
        return row ? { ...row, messages: ledgerParse(row.messages) } : null;
    }
    delete_conversation(conversation_id: string): void {
        this.connect(db => db.query("DELETE FROM conversations WHERE id=?").run(conversation_id));
    }
    event(kind: string, body: any): string {
        const event_id = id();
        this.connect(db => db.query("INSERT INTO events VALUES (?,?,?,?)").run(event_id, kind, ledgerStringify(body), now()));
        return event_id;
    }
    events(): RecordValue[] { return this.connect(db => normalizeRows(db.query("SELECT * FROM events ORDER BY created DESC LIMIT 30").all() as RecordValue[]).map(row => ({ ...row, body: ledgerParse(row.body) }))); }

    private refreshLease() {
        if (!this.lease) return;
        clearTimeout(this.lease.timer);
        this.lease.timer = setTimeout(() => this.releaseLease(false), this.leaseTimeoutMs);
        this.lease.timer.unref?.();
    }
    private releaseLease(commit: boolean) {
        const lease = this.lease;
        if (!lease) return;
        try { if (lease.db.inTransaction) lease.db.exec(commit ? "COMMIT" : "ROLLBACK"); }
        finally {
            try { lease.db.close(); }
            finally { clearTimeout(lease.timer); this.lease = undefined; for (const waiter of [...this.waiters]) waiter(); }
        }
    }
    private sqlExecute(sql: string, parameters: Scalar[] | Record<string, Scalar> = []) {
        const db = this.lease!.db;
        const leading = sql.replace(/^(?:\s+|--[^\n]*(?:\n|$)|\/\*[\s\S]*?\*\/)+/, "");
        const dml = /^(?:INSERT|UPDATE|DELETE|REPLACE)\b/i.test(leading);
        if (dml && !db.inTransaction && !this.lease!.readonly) db.exec("BEGIN");
        const bind = (value: any): any => value instanceof Number ? value.valueOf() : typeof value === "string" ? unicode(value) : value;
        const bound = Array.isArray(parameters) ? parameters.map(bind) : Object.fromEntries(Object.entries(parameters).map(([key, value]) => [key, bind(value)]));
        const query = db.prepare(floatParameters(sql, parameters));
        try {
            const rows = ((Array.isArray(bound) ? query.values(...bound as SQLQueryBindings[]) : query.values(bound as any)) ?? []).map(row => row.map(value => typeof value === "bigint" ? integer(value) : typeof value === "number" ? floatValue(value, Object.is(value, -0) ? "-0.0" : String(value)) : value));
            const columns = query.columnNames ?? [];
            // Bun 1.3.5 drops duplicate names even though values() keeps every position.
            // Refuse ambiguous cursors instead of silently pairing the wrong key/value.
            if (query.native.columnsCount !== columns.length) throw new Error("SQLite查询列名称重复，请显式使用独立别名");
            const info = db.query("SELECT changes() AS n, last_insert_rowid() AS id").get() as RecordValue;
            return { columns, rows, rowcount: dml ? integer(info.n) : -1, lastrowid: integer(info.id), in_transaction: db.inTransaction };
        } finally { query.finalize(); }
    }
    /** Short local ledger work shares the RPC lease queue; never hold it across model awaits. */
    async runWhenIdle<T>(identity: string | null, operation: () => T, signal?: AbortSignal): Promise<T> {
        while (this.lease) await new Promise<void>((resolve, reject) => {
            const cleanup = () => { this.waiters.delete(ready); signal?.removeEventListener("abort", abort); };
            const ready = () => { cleanup(); resolve(); };
            const abort = () => { cleanup(); reject(signal!.reason); };
            this.waiters.add(ready);
            signal?.addEventListener("abort", abort, { once: true });
            if (signal?.aborted) abort();
        });
        signal?.throwIfAborted();
        if (this.closed) throw new Error("数据服务已关闭");
        const previous = this.model_identity;
        this.model_identity = identity;
        try {
            const result = operation();
            if (result && typeof (result as any).then === "function") throw new Error("数据库同步操作不能返回 Promise");
            return result;
        } finally { this.model_identity = previous; }
    }
    async dispatch(method: string, args: any[] = [], kwargs: RecordValue = {}, identity: string | null = null, transaction?: string, signal?: AbortSignal): Promise<any> {
        signal?.throwIfAborted();
        if (this.closed) throw new Error("数据服务已关闭");
        if (!Array.isArray(args) || !kwargs || typeof kwargs !== "object" || Array.isArray(kwargs)) throw new Error("RPC参数无效");
        if (method.startsWith("db_") && method !== "db_begin") {
            if (!this.lease || transaction !== this.lease.token || identity !== this.lease.identity) throw new Error("数据库事务已结束或身份不匹配");
            this.refreshLease();
            if (method === "db_execute") return this.sqlExecute(kwargs.sql ?? args[0], kwargs.parameters ?? args[1] ?? []);
            if (method === "db_script") {
                if (this.lease.db.inTransaction) this.lease.db.exec("COMMIT");
                this.lease.db.exec(kwargs.sql ?? args[0]);
                return { in_transaction: this.lease.db.inTransaction };
            }
            if (method === "db_commit" || method === "db_rollback") {
                if (this.lease.db.inTransaction) this.lease.db.exec(method === "db_commit" ? "COMMIT" : "ROLLBACK");
                return { in_transaction: false };
            }
            if (method === "db_close") { this.releaseLease(false); return null; }
            throw new Error("未知数据库方法");
        }
        return this.runWhenIdle(identity, () => {
        if (method === "db_begin") {
            const token = id();
            this.lease = { token, db: this.open(kwargs.readonly === true), identity, readonly: kwargs.readonly === true, timer: setTimeout(() => {}, 0) };
            this.refreshLease();
            return { transaction: token, in_transaction: false };
        }
        const parameters: Record<string, string[]> = {
            growth_reserved: ["messages"], add_sample: ["prompt", "answer", "source", "purpose", "approved", "training_allowed", "provenance"],
            samples: ["purpose"], delete_sample: ["sample_id"], record_usage: ["route", "usage"], usage: [],
            capture_teacher: ["messages", "answer", "provenance", "eligible", "reason"], teacher_materials: [], material_counts: [],
            save_conversation: ["conversation_id", "messages"], conversations: [], conversation: ["conversation_id"], delete_conversation: ["conversation_id"], event: ["kind", "body"], events: [],
        };
        const names = parameters[method];
        if (!names || args.length > names.length || Object.keys(kwargs).some(name => !names.includes(name))) throw new Error("未知账本方法或参数");
        const values = names.map((name, index) => {
            if (index < args.length && name in kwargs) throw new Error("重复账本参数");
            return index < args.length ? args[index] : kwargs[name];
        });
        return (this as any)[method](...values) ?? null;
        }, signal);
    }
    close(): void { this.closed = true; this.releaseLease(false); }
}
