import { afterEach, describe, expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync, readFileSync, chmodSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { Memory, ledgerStringify, ledgerParse, pythonStrip } from "../backend/memory";
import { decodeWire, encodeWire, parseJSON, floatValue } from "../backend/wire";

const roots: string[] = [], memories: Memory[] = [];
function fixture(timeout?: number) {
    const root = mkdtempSync(join(tmpdir(), "luban-bun-memory-")); roots.push(root);
    const memory = new Memory(root, timeout); memories.push(memory); return memory;
}
afterEach(() => { for (const memory of memories.splice(0)) memory.close(); for (const root of roots.splice(0)) rmSync(root, { force: true, recursive: true }); });
const add = (memory: Memory, prompt = "同一个问题", answer = "同一个答案", purpose = "training", provenance = {}) => memory.add_sample(prompt, answer, "human", purpose, true, true, provenance);
const capture = (memory: Memory, prompt = "同一个问题", answer = "同一个答案", eligible = true) => {
    const identifier = memory.capture_teacher([{ role: "user", content: prompt }], answer, {}, eligible);
    return memory.teacher_materials().find(row => row.id === identifier)!;
};
async function begin(memory: Memory, identity: string | null = null, readonly = false) { return (await memory.dispatch("db_begin", [], { readonly }, identity)).transaction as string; }
async function sql(memory: Memory, transaction: string, query: string, parameters: any[] = [], identity: string | null = null) { return memory.dispatch("db_execute", [], { sql: query, parameters }, identity, transaction); }
async function close(memory: Memory, transaction: string, identity: string | null = null) { await memory.dispatch("db_commit", [], {}, identity, transaction); await memory.dispatch("db_close", [], {}, identity, transaction); }
function python(script: string, args: string[] = []) {
    const result = Bun.spawnSync([resolve(".venv/bin/python"), "-c", script, ...args], { env: { ...process.env, PYTHONPATH: resolve("src") } });
    if (result.exitCode !== 0) throw new Error(result.stderr.toString());
    return JSON.parse(result.stdout.toString(), (_key, value, context) => {
        if (typeof value === "number" && context?.source && /^-?\d+$/.test(context.source)) {
            const exact = BigInt(context.source);
            return exact > BigInt(Number.MAX_SAFE_INTEGER) || exact < BigInt(Number.MIN_SAFE_INTEGER) ? exact : value;
        }
        return value;
    });
}

describe("Bun ledger uses the existing Python SQLite contract", () => {
    test("conversation deletion via RPC is idempotent and preserves learning records", async () => {
        const memory = fixture(); const sample = add(memory);
        const material = capture(memory, "teacher question", "teacher answer");
        const conversation = memory.save_conversation(null, [{ role: "user", content: "history" }]);
        await memory.dispatch("delete_conversation", [conversation]);
        expect(memory.conversation(conversation)).toBeNull(); expect(memory.conversations()).toEqual([]);
        await memory.dispatch("delete_conversation", [conversation]);
        expect(memory.samples().some(row => row.id === sample)).toBe(true);
        expect(memory.teacher_materials().some(row => row.id === material.id)).toBe(true);
    });
    test("active identity scopes duplicates and overwrites forged provenance", () => {
        const memory = fixture(); memory.model_identity = "old-base"; const old = add(memory, undefined, undefined, "validation");
        memory.model_identity = "new-base"; const current = add(memory, undefined, undefined, "training", { base_identity: "old-base" });
        expect(memory.samples().map(row => row.id)).toEqual([current]);
        expect(ledgerParse(memory.samples()[0]!.provenance).base_identity).toBe("new-base");
        memory.model_identity = "old-base"; expect(memory.samples().map(row => row.id)).toEqual([old]);
        memory.model_identity = null; expect(() => add(memory)).toThrow("训练和检验");
    });
    test("same-base cross-split guard works in both directions and ignores legacy unscoped rows", () => {
        const memory = fixture(); add(memory, "legacy", "answer", "validation");
        memory.model_identity = "base"; add(memory, "legacy");
        for (const [original, other] of [["training", "validation"], ["validation", "training"]]) {
            add(memory, original, "answer", original);
            expect(() => add(memory, " " + original + " ", "answer", other)).toThrow("训练和检验");
        }
    });
    test("teacher keeps manual split and idempotently reuses same-base eligible sample", () => {
        const memory = fixture(); memory.model_identity = "base";
        const existing = add(memory, undefined, undefined, "validation");
        expect(capture(memory).sample_id).toBe(existing);
        const different = capture(memory, undefined, "不同答案");
        expect(different.sample_id).not.toBe(existing); expect(different.purpose).toBe("validation");
        expect(capture(memory, undefined, "不同答案").sample_id).toBe(different.sample_id);
        expect(memory.samples()).toHaveLength(2); expect(memory.material_counts()).toEqual({ total: 3, eligible: 3, pending: 0 });
    });
    test("retirement preserves split history without absorbing fresh teacher material", () => {
        const memory = fixture(); memory.model_identity = "base";
        const existing = add(memory, undefined, undefined, "validation");
        memory.connect(db => db.query("UPDATE samples SET approved=0 WHERE id=?").run(existing));
        const fresh = capture(memory); expect(fresh.sample_id).not.toBe(existing); expect(fresh.purpose).toBe("validation");
        expect(memory.samples().find(row => row.id === existing)!.approved).toBe(0);
        expect(() => add(memory)).toThrow("训练和检验");
        memory.delete_sample(fresh.sample_id); const detached = memory.teacher_materials()[0]!;
        expect(detached.sample_id).toBeNull(); expect(detached.status).toBe("pending"); expect(detached.reason).toBe("学习材料已移除");
        expect(capture(memory).sample_id).not.toBe(existing);
    });
    test("teacher split hash and cross-base provenance agree with Python", () => {
        const memory = fixture();
        for (const prompt of ["教师去重测试", "😀", "\u0085 hi \u001f", "\ufeff", "é", "你好"] ) {
            const expected = python("import hashlib,json,sys;print(json.dumps('validation' if int(hashlib.sha256(sys.argv[1].strip().encode()).hexdigest(),16)%5==0 else 'training'))", [prompt]);
            memory.model_identity = "old"; const previous = add(memory, prompt, "answer", expected === "training" ? "validation" : "training");
            memory.model_identity = "new"; const material = capture(memory, prompt, "answer");
            expect(material.sample_id).not.toBe(previous); expect(material.purpose).toBe(expected);
        }
    });
    test("Python Unicode whitespace, codepoint limits, and 40-codepoint titles", () => {
        const memory = fixture();
        expect(pythonStrip("\u001c\u0085 hello\u001f")).toBe("hello"); expect(pythonStrip("\ufeffhello\ufeff")).toBe("\ufeffhello\ufeff");
        add(memory, "😀".repeat(6000), "🧠".repeat(6000)); expect(memory.samples()[0]!.prompt).toBe("😀".repeat(6000));
        expect(() => add(memory, "😀".repeat(6001), "🧠".repeat(6000))).toThrow("12000");
        expect(() => add(memory, "\u001c\u0085", "answer")).toThrow("正确答案");
        expect(() => add(memory, "\ud800", "answer")).toThrow("Unicode");
        expect(() => memory.event("bad", { answer: "\udfff" })).toThrow("Unicode");
        const cid = memory.save_conversation(null, [{ role: "user", content: "😀".repeat(45) }]);
        expect(memory.conversation(cid)!.title).toBe("😀".repeat(40));
    });
    test("growth NFKC/whitespace protects any user turn, filters historical samples, and never auto-learns teacher", () => {
        const memory = fixture(); add(memory, "Ａ\u0085B", "answer");
        mkdirSync(join(memory.root, "app/growth")); writeFileSync(join(memory.root, "app/growth/rounds.json"), JSON.stringify({ reserved_prompts: ["A B"] }));
        expect(memory.samples()).toEqual([]);
        expect(() => add(memory, "other", "answer", "training", { messages: [{ role: "user", content: "Ａ\u001cＢ" }, { role: "user", content: "other" }] })).toThrow("成长评测");
        const material = memory.capture_teacher([{ role: "user", content: "ＡＢ" }, { role: "user", content: "other" }], "answer", {}, true);
        expect(memory.teacher_materials().find(row => row.id === material)!.status).toBe("pending");
        writeFileSync(join(memory.root, "app/growth/rounds.json"), JSON.stringify({ reserved_prompts: [42] }));
        expect(() => memory.samples()).toThrow("损坏");
    });
    test("growth symlinks and database/directory/sidecar symlinks fail before writing targets", () => {
        const memory = fixture(), target = fixture();
        mkdirSync(join(memory.root, "app/growth"), { recursive: true });
        writeFileSync(join(target.root, "rounds.json"), JSON.stringify({ reserved_prompts: [] }));
        symlinkSync(join(target.root, "rounds.json"), join(memory.root, "app/growth/rounds.json"));
        expect(() => memory.growth_reserved([])).toThrow("符号链接");
        const another = fixture(); symlinkSync(target.root, join(another.root, "app"));
        expect(() => another.usage()).toThrow("符号链接");
        for (const suffix of ["", "-journal", "-wal", "-shm"]) {
            const linked = fixture(); mkdirSync(join(linked.root, "app"));
            writeFileSync(join(target.root, "victim"), "preserve"); symlinkSync(join(target.root, "victim"), linked.path + suffix);
            expect(() => linked.usage()).toThrow("符号链接"); expect(readFileSync(join(target.root, "victim"), "utf8")).toBe("preserve");
        }
    });
    test("usage validates booleans/fractions/negatives and preserves SQLite signed64 integers", () => {
        const memory = fixture();
        for (const n of [true, -1, 1.2, "1", null, 9007199254740992]) expect(() => memory.record_usage("local", { prompt_tokens: n })).toThrow("非负整数");
        expect(() => memory.record_usage("unknown", {})).toThrow("来源");
        expect(() => memory.record_usage("local", { prompt_tokens: 9223372036854775808n })).toThrow("SQLite INTEGER");
        memory.record_usage("local", { prompt_tokens: 9007199254740993n, completion_tokens: 4, total_tokens: 9007199254740997n, estimated: [] });
        memory.record_usage("local", { prompt_tokens: 1, estimated: true });
        expect(memory.usage().local).toEqual({ calls: 2, prompt_tokens: 9007199254740994n, completion_tokens: 4, total_tokens: 9007199254740997n, estimated_calls: 1 });
        expect(memory.usage().teacher.calls).toBe(0);
    });
    test("BigInt provenance/event JSON is exact in SQLite and Python reads it losslessly", () => {
        const memory = fixture(); memory.event("numbers", { n: 9223372036854775807n, unicode: "中文😀" });
        expect(memory.events()[0]!.body.n).toBe(9223372036854775807n);
        const value = python("import sqlite3,json,sys;db=sqlite3.connect(sys.argv[1]);print(db.execute('SELECT body FROM events').fetchone()[0])", [memory.path]);
        expect(value).toEqual({ n: 9223372036854775807n, unicode: "中文😀" });
        expect(ledgerParse(ledgerStringify({ int: -9223372036854775808n }))).toEqual({ int: -9223372036854775808n });
    });
    test("Python floats retain float type through JSON persistence, SQL binding and wire", async () => {
        const memory = fixture();
        const body = parseJSON('{"integer_float":16.0,"exp":1e-9,"zero":-0.0}');
        memory.event("floats", body);
        expect(ledgerStringify(memory.events()[0]!.body)).toBe('{"integer_float": 16.0, "exp": 1e-9, "zero": -0.0}');
        expect(encodeWire(memory.events()[0]!.body).integer_float).toEqual({ $luban_float: "16.0" });
        expect(() => memory.record_usage("local", { prompt_tokens: body.integer_float })).toThrow("非负整数");
        memory.record_usage("local", { estimated: floatValue(1) }); expect(memory.usage().local.estimated_calls).toBe(1);
        const transaction = await begin(memory);
        const result = await sql(memory, transaction, "SELECT ? AS f,? AS n,? AS data", [body.integer_float, 9223372036854775807n, new Uint8Array([0, 255])]);
        const wire = encodeWire(result.rows[0]); expect(wire[0]).toEqual({ $luban_float: "16" }); expect(wire[1]).toEqual({ $luban_int: "9223372036854775807" });
        expect(decodeWire(wire)[2]).toEqual(new Uint8Array([0, 255]));
        const named = await memory.dispatch("db_execute", [], { sql: "SELECT ':n' AS literal,typeof(:n) AS kind,:n AS value /* :n */", parameters: { n: floatValue(1) } }, null, transaction);
        expect(named.rows[0].slice(0, 2)).toEqual([":n", "real"]);
        const zero = await sql(memory, transaction, "SELECT ? AS zero", [body.zero]);
        expect(encodeWire(zero.rows[0][0])).toEqual({ $luban_float: "-0.0" });
        await close(memory, transaction);
    });
    test("high-frequency samples preserve the earliest source order without millisecond ties", () => {
        const memory = fixture(), identifiers: string[] = [];
        for (let index = 0; index < 64; index++) identifiers.push(add(memory, "same", "gold"));
        const rows = memory.samples().reverse(); expect(rows.map(row => row.id)).toEqual(identifiers);
        for (let index = 1; index < rows.length; index++) expect(rows[index]!.created).toBeGreaterThan(rows[index - 1]!.created);
    });
    test("SQL transaction is lazy, rollback is real, same identity required and duplicate names fail safely", async () => {
        const memory = fixture(), transaction = await begin(memory, "base");
        await expect(sql(memory, transaction, "SELECT 1 AS n,2 AS n", [], "base")).rejects.toThrow("列名称重复");
        const select = await sql(memory, transaction, "SELECT 1 AS n,2 AS p", [], "base");
        expect(select).toMatchObject({ columns: ["n", "p"], rows: [[1, 2]], rowcount: -1, in_transaction: false });
        await expect(sql(memory, transaction, "SELECT 1", [], "wrong")).rejects.toThrow("身份");
        await sql(memory, transaction, "CREATE TABLE training_runs (id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)", [], "base");
        const insert = await sql(memory, transaction, "INSERT INTO training_runs VALUES (?,?,?,?,?,?)", ["run", "base", "evolution", "committing", "{}", 1], "base");
        expect(insert).toMatchObject({ rowcount: 1, in_transaction: true });
        await expect(sql(memory, transaction, "INSERT INTO training_runs VALUES (?,?,?,?,?,?)", ["run", "base", "evolution", "committing", "{}", 1], "base")).rejects.toThrow();
        await memory.dispatch("db_rollback", [], {}, "base", transaction); await memory.dispatch("db_close", [], {}, "base", transaction);
        const later = await begin(memory); expect((await sql(memory, later, "SELECT * FROM training_runs")).rows).toEqual([]); await close(memory, later);
    });
    test("journal/material/event writes share one transaction and trigger failures rollback all", async () => {
        const memory = fixture(), transaction = await begin(memory);
        await sql(memory, transaction, "CREATE TABLE training_runs (id TEXT PRIMARY KEY,state TEXT)");
        await sql(memory, transaction, "CREATE TRIGGER fail_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'fixture'); END");
        await sql(memory, transaction, "BEGIN IMMEDIATE");
        await sql(memory, transaction, "INSERT INTO training_runs VALUES ('run','committing')");
        await sql(memory, transaction, "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", ["material", "question", "answer", "teacher", "training", 0, 1, 1, "{}"]);
        await expect(sql(memory, transaction, "INSERT INTO events VALUES ('training:run','evolution','{}',0)")).rejects.toThrow("fixture");
        await memory.dispatch("db_rollback", [], {}, null, transaction); await memory.dispatch("db_close", [], {}, null, transaction);
        expect(memory.samples()).toEqual([]); expect(memory.events()).toEqual([]);
        const later = await begin(memory); expect((await sql(memory, later, "SELECT * FROM training_runs")).rows).toEqual([]); await close(memory, later);
    });
    test("business writes rollback on exceptions including teacher insert second-statement failure", () => {
        const memory = fixture(); memory.connect(db => db.exec("CREATE TRIGGER fail_material BEFORE INSERT ON teacher_materials BEGIN SELECT RAISE(ABORT,'fixture second insert'); END"));
        expect(() => capture(memory, "fresh", "answer")).toThrow("fixture second insert"); expect(memory.samples()).toEqual([]);
        expect(memory.material_counts()).toEqual({ total: 0, pending: 0, eligible: 0 });
        expect(() => memory.connect(db => { db.query("INSERT INTO events VALUES ('temporary','fixture','{}',1)").run(); throw new Error("abort"); })).toThrow("abort");
        expect(memory.events()).toEqual([]);
    });
    test("lease operations stay responsive while business operations wait until close", async () => {
        const memory = fixture(), transaction = await begin(memory);
        await sql(memory, transaction, "INSERT INTO events VALUES ('queued','fixture','{}',1)");
        let finished = false; const waiting = memory.dispatch("events").then(result => { finished = true; return result; });
        await new Promise(resolve => setTimeout(resolve, 10)); expect(finished).toBe(false);
        await memory.dispatch("db_commit", [], {}, null, transaction); expect(finished).toBe(false);
        await memory.dispatch("db_close", [], {}, null, transaction); expect((await waiting)[0].id).toBe("queued");
    });
    test("local learning ledger work waits asynchronously for the same-root RPC lease and restores identity", async () => {
        const memory = fixture(); memory.model_identity = "previous";
        const transaction = await begin(memory, "rpc");
        await sql(memory, transaction, "INSERT INTO events VALUES ('rpc','fixture','{}',1)", [], "rpc");
        let entered = false;
        const waiting = memory.runWhenIdle("learning", () => {
            entered = true; expect(memory.model_identity).toBe("learning");
            add(memory, "learning question", "learning answer");
            return memory.events().map(row => row.id);
        });
        await new Promise<void>((done, fail) => setTimeout(() => {
            expect(entered).toBe(false);
            close(memory, transaction, "rpc").then(done, fail);
        }, 10));
        expect(await waiting).toEqual(["rpc"]); expect(memory.model_identity).toBe("previous");
        expect(await memory.dispatch("samples", [], {}, "rpc")).toEqual([]);
        expect(await memory.dispatch("samples", [], {}, "learning")).toHaveLength(1);
    });
    test("local ledger arbitration stays independent across roots and releases between short stages", async () => {
        const first = fixture(), second = fixture(), transaction = await begin(first);
        let entered = false;
        const waiting = first.runWhenIdle(null, () => { entered = true; first.event("learning", {}); });
        await second.runWhenIdle(null, () => second.event("independent", {}));
        expect(entered).toBe(false); expect(second.events()[0]!.kind).toBe("independent");
        await close(first, transaction); await waiting;
        // No lease or SQLite transaction spans the next model await.
        const later = await begin(first); expect((await sql(first, later, "SELECT kind FROM events")).rows).toEqual([["learning"]]);
        await close(first, later);
    });
    test("cancelled or closed local waiters do not run, and callback errors restore identity", async () => {
        const memory = fixture(), transaction = await begin(memory), controller = new AbortController();
        let entered = false;
        const cancelled = memory.runWhenIdle("learning", () => { entered = true; }, controller.signal).catch(error => error);
        controller.abort(); expect((await cancelled).name).toBe("AbortError"); expect(entered).toBe(false);
        await close(memory, transaction);
        memory.model_identity = "previous";
        await expect(memory.runWhenIdle("learning", () => { throw Error("fixture"); })).rejects.toThrow("fixture");
        await expect(memory.runWhenIdle("learning", () => Promise.resolve())).rejects.toThrow("不能返回 Promise");
        expect(memory.model_identity).toBe("previous");
        const later = await begin(memory), closed = memory.runWhenIdle(null, () => { entered = true; });
        memory.close(); await expect(closed).rejects.toThrow("关闭"); expect(entered).toBe(false);
    });
    test("cancelled queued begins and writes leave the current owner's transaction intact", async () => {
        const memory = fixture(), ownerSignal = new AbortController();
        const transaction = (await memory.dispatch("db_begin", [], {}, null, undefined, ownerSignal.signal)).transaction;
        // The completed begin request no longer owns cancellation of the acknowledged transaction.
        ownerSignal.abort();
        await sql(memory, transaction, "INSERT INTO events VALUES ('owner','fixture','{}',1)");
        const cancelled = new AbortController();
        const pending = [memory.dispatch("db_begin", [], {}, null, undefined, cancelled.signal),
            memory.dispatch("event", ["cancelled", {}], {}, null, undefined, cancelled.signal)].map(value => value.catch(error => error));
        cancelled.abort();
        expect((await Promise.all(pending)).map(error => error.name)).toEqual(["AbortError", "AbortError"]);
        expect((await sql(memory, transaction, "SELECT id FROM events")).rows).toEqual([["owner"]]);
        await close(memory, transaction);
        expect((await memory.dispatch("events")).map((row: any) => row.id)).toEqual(["owner"]);
    });
    test("cancellation after wakeup but before continuation cannot take the released lease", async () => {
        const memory = fixture(), transaction = await begin(memory), controller = new AbortController();
        const pending = memory.dispatch("db_begin", [], {}, null, undefined, controller.signal).catch(error => error);
        const released = memory.dispatch("db_close", [], {}, null, transaction);
        controller.abort(); await released;
        expect((await pending).name).toBe("AbortError");
        const next = await begin(memory); await close(memory, next);
        expect(await memory.dispatch("events")).toEqual([]);
        await expect(memory.dispatch("db_begin", [], {}, null, undefined, controller.signal)).rejects.toThrow();
        expect(await memory.dispatch("events")).toEqual([]);
    });
    test("abandoned lease idle expiry and service close roll back and release waiters", async () => {
        const memory = fixture(25), transaction = await begin(memory);
        await sql(memory, transaction, "INSERT INTO events VALUES ('orphan','fixture','{}',1)");
        expect(await memory.dispatch("events")).toEqual([]);
        await expect(sql(memory, transaction, "SELECT 1")).rejects.toThrow("已结束");
        const second = await begin(memory); await sql(memory, second, "INSERT INTO events VALUES ('close','fixture','{}',1)");
        const pending = memory.dispatch("events"); memory.close(); await expect(pending).rejects.toThrow("关闭");
        const db = new Database(memory.path, { readonly: true }); expect(db.query("SELECT * FROM events").all()).toEqual([]); db.close();
    });
    test("read-only lease creates no DB/schema and denies mutations", async () => {
        const missing = fixture(); await expect(begin(missing, null, true)).rejects.toThrow();
        expect(() => readFileSync(missing.path)).toThrow();
        const memory = fixture(); memory.event("fixture", {});
        const transaction = await begin(memory, null, true); expect((await sql(memory, transaction, "SELECT * FROM events")).rows).toHaveLength(1);
        await expect(sql(memory, transaction, "INSERT INTO events VALUES ('write','fixture','{}',1)")).rejects.toThrow();
        await memory.dispatch("db_close", [], {}, null, transaction); expect(memory.events()).toHaveLength(1);
    });
    test("file/directory failures remain errors; missing volumes never create shadow storage", () => {
        const memory = fixture(); mkdirSync(join(memory.root, "app")); mkdirSync(memory.path);
        expect(() => memory.usage()).toThrow();
        const volume = new Memory("/Volumes/luban-not-mounted-" + crypto.randomUUID() + "/storage");
        expect(() => volume.usage()).toThrow("存储盘未连接"); volume.close();
        if (process.getuid?.() !== 0) {
            const protectedMemory = fixture(); mkdirSync(join(protectedMemory.root, "app")); chmodSync(join(protectedMemory.root, "app"), 0o500);
            try { expect(() => protectedMemory.event("fixture", {})).toThrow(); }
            finally { chmodSync(join(protectedMemory.root, "app"), 0o700); }
        }
    });
    test("connection uses EXCLUSIVE without WAL and closes so independent Python reader can reopen", () => {
        const memory = fixture(); memory.connect(db => { expect(db.query("PRAGMA locking_mode").get()).toEqual({ locking_mode: "exclusive" }); expect(db.query("PRAGMA journal_mode").get()).toEqual({ journal_mode: "delete" }); });
        memory.event("reopen", {});
        expect(python("import sqlite3,json,sys;db=sqlite3.connect(sys.argv[1],timeout=.1);print(json.dumps(db.execute('PRAGMA integrity_check').fetchone()[0]));db.close()", [memory.path])).toBe("ok");
    });
    test("Python-created existing DB and all Bun writes round-trip unchanged through Python", async () => {
        const memory = fixture();
        const initial = python(`from luban.memory import Memory
import json,sys
m=Memory(sys.argv[1]);m.model_identity='base'
s=m.add_sample('原有😀','原答案',approved=True,training_allowed=True)
m.capture_teacher([{'role':'user','content':'原有😀'}],'原答案',{},True)
c=m.save_conversation('conversation',[{'role':'user','content':'😀'*45}])
m.record_usage('teacher',{'prompt_tokens':9007199254740993,'completion_tokens':3,'total_tokens':9007199254740996})
e=m.event('python',{'integer':9223372036854775807})
print(json.dumps({'samples':m.samples(),'materials':m.teacher_materials(),'usage':m.usage(),'conversation':m.conversation(c),'events':m.events()},ensure_ascii=False))`, [memory.root]);
        memory.model_identity = "base";
        expect(memory.samples()).toEqual(initial.samples); expect(memory.teacher_materials()).toEqual(initial.materials); expect(memory.usage()).toEqual(initial.usage);
        expect(memory.conversation("conversation")).toEqual(initial.conversation); expect(memory.events()).toEqual(initial.events);
        memory.event("bun", { integer: -9223372036854775808n }); add(memory, "Bun新增", "新答案");
        const after = python(`from luban.memory import Memory
import json,sys
m=Memory(sys.argv[1]);m.model_identity='base'
print(json.dumps({'samples':m.samples(),'events':m.events(),'usage':m.usage(),'counts':m.material_counts()},ensure_ascii=False))`, [memory.root]);
        expect(memory.samples()).toEqual(after.samples); expect(memory.events()).toEqual(after.events); expect(memory.usage()).toEqual(after.usage); expect(memory.material_counts()).toEqual(after.counts);
    });
    test("RPC kwargs binding rejects unknown/duplicate methods and identity does not leak", async () => {
        const memory = fixture(); const identifier = await memory.dispatch("add_sample", [], { prompt: "kwargs", answer: "answer", approved: true, training_allowed: true }, "base");
        expect((await memory.dispatch("samples", [], {}, "base"))[0].id).toBe(identifier);
        expect(await memory.dispatch("samples", [], {}, "other")).toEqual([]); expect(memory.model_identity).toBeNull();
        await expect(memory.dispatch("close")).rejects.toThrow("未知");
        await expect(memory.dispatch("samples", ["training"], { purpose: "validation" })).rejects.toThrow("重复");
        await expect(memory.dispatch("samples", [], { wrong: 1 })).rejects.toThrow("未知");
    });
});

test("missing teacher usage preserves call counts, marks totals unknown across reopen, and never invents zero tokens", () => {
    const memory = fixture();
    memory.record_usage("teacher", { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 });
    const unknown = { unknown: true, prompt_tokens: null, completion_tokens: null, total_tokens: null };
    memory.record_usage("teacher", unknown);
    expect(memory.usage().teacher).toEqual({ calls: 2, prompt_tokens: null, completion_tokens: null, total_tokens: null, estimated_calls: 0, unknown_calls: 1 });
    const reopened = new Memory(memory.root); memories.push(reopened);
    reopened.record_usage("teacher", { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 });
    expect(reopened.usage().teacher.total_tokens).toBeNull();
    expect(reopened.usage().teacher.calls).toBe(3);
    expect(reopened.usage().local.total_tokens).toBe(0);
    expect(python(`import json,sys
from luban.memory import Memory
m=Memory(sys.argv[1]); print(json.dumps(m.usage()))`, [memory.root])).toEqual(reopened.usage());
    for (const usage of [{ unknown: true }, { ...unknown, total_tokens: 0 }, { ...unknown, prompt_tokens: false }]) expect(() => reopened.record_usage("teacher", usage)).toThrow();
    expect(reopened.usage().teacher.calls).toBe(3);
});
