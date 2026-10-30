import { Database } from "bun:sqlite";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { randomBytes } from "node:crypto";
import { COUNTRY_CODES } from "./countries";
import { maskName } from "./name-filter";
import { createEvaluations, EvaluationError } from "./evaluation";
import { createEnterprise } from "./enterprise";
import { createAppAd } from "./app-ad";
import { createAppStarts } from "./app-starts";
import { createGrowth, GROWTH_DOMAINS, type GrowthDomain } from "./growth";
import { escape, profileMarkup } from "./profile";
import { domainNames, messages, locale, localizedCountry, text, type Language } from "./i18n";
import { resolveLanguage, validLanguage } from "./language";
import downloads from "./downloads.json";

const root = import.meta.dir;
// Browsable categories do not imply a supported scoring protocol.
const DOMAIN_LABELS = domainNames.zh;
type RankingDomain = keyof typeof DOMAIN_LABELS;
const supportsGrowth = (domain:RankingDomain) => domain==="all"||GROWTH_DOMAINS.includes(domain as GrowthDomain);
const databasePath = process.env.LEADERBOARD_DB || `${root}/data/leaderboard.sqlite`;
mkdirSync(dirname(databasePath), { recursive: true });
const db = new Database(databasePath, { create: true });
db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
  CREATE TABLE IF NOT EXISTS entries (
    owner TEXT PRIMARY KEY, username TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
    evolution_count INTEGER NOT NULL CHECK(evolution_count BETWEEN 0 AND 1000000000),
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, visible INTEGER NOT NULL DEFAULT 1,
    parameter_total TEXT
  );
  CREATE TABLE IF NOT EXISTS upgrades (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, proof TEXT NOT NULL, received_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS upgrade_owner ON upgrades(owner);
  CREATE TABLE IF NOT EXISTS display_entries (
    id TEXT PRIMARY KEY, username TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
    country_code TEXT NOT NULL, evolution_count INTEGER NOT NULL CHECK(evolution_count BETWEEN 0 AND 1000000000),
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS ranking ON entries(evolution_count DESC, created_at ASC, owner ASC);`);
// Preserve old profiles but quarantine editable v1 counts until a v2 report.
if (!db.query<{ name: string }, []>("PRAGMA table_info(entries)").all().some(column => column.name === "visible")) {
  db.exec("ALTER TABLE entries ADD COLUMN visible INTEGER NOT NULL DEFAULT 0");
}
if (!db.query<{ name: string }, []>("PRAGMA table_info(entries)").all().some(column => column.name === "country_code")) {
  db.exec("ALTER TABLE entries ADD COLUMN country_code TEXT");
}
if (!db.query<{ name: string }, []>("PRAGMA table_info(entries)").all().some(column => column.name === "parameter_total")) {
  db.exec("ALTER TABLE entries ADD COLUMN parameter_total TEXT");
}
for (const [name, type] of [["active_evaluation", "TEXT"], ["token_total", "INTEGER"], ["growth_model_version", "TEXT"]]) {
  if (!db.query<{ name: string }, []>("PRAGMA table_info(entries)").all().some(column => column.name === name)) db.exec(`ALTER TABLE entries ADD COLUMN ${name} ${type}`);
}
for (const [name, type] of [["token_total", "INTEGER"], ["benchmark", "REAL"], ["generalization", "REAL"], ["growth_profile", "TEXT"]]) {
  if (!db.query<{ name: string }, []>("PRAGMA table_info(display_entries)").all().some(column => column.name === name)) db.exec(`ALTER TABLE display_entries ADD COLUMN ${name} ${type}`);
}
const evaluations = createEvaluations(db);
const growth = createGrowth(db);
// Keep the management credential outside the public payload and database.
const adminKeyPath = process.env.ENTERPRISE_ADMIN_KEY_FILE || `${dirname(databasePath)}/enterprise-admin-key`;
function adminKey() {
  if (process.env.ENTERPRISE_ADMIN_KEY) return process.env.ENTERPRISE_ADMIN_KEY;
  try { return readFileSync(adminKeyPath, "utf8").trim(); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    try { writeFileSync(adminKeyPath, randomBytes(32).toString("hex") + "\n", { flag: "wx", mode: 0o600 }); }
    catch (writeError) { if ((writeError as NodeJS.ErrnoException).code !== "EEXIST") throw writeError; }
    return readFileSync(adminKeyPath, "utf8").trim();
  }
}
const enterprise = createEnterprise(db, { adminKey: adminKey(), origin: process.env.ENTERPRISE_ORIGIN });
const adKeyPath = process.env.APP_AD_ADMIN_KEY_FILE || `${dirname(databasePath)}/app-ad-admin-key`;
function adKey() {
  if (process.env.APP_AD_ADMIN_KEY) return process.env.APP_AD_ADMIN_KEY;
  try { return readFileSync(adKeyPath, "utf8").trim(); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    try { writeFileSync(adKeyPath, randomBytes(32).toString("hex") + "\n", { flag: "wx", mode: 0o600 }); }
    catch (writeError) { if ((writeError as NodeJS.ErrnoException).code !== "EEXIST") throw writeError; }
    return readFileSync(adKeyPath, "utf8").trim();
  }
}
const appManagementKey = adKey();
const appAd = createAppAd(db, appManagementKey);
const appStarts = createAppStarts(db, appManagementKey);

type Entry = { username: string; country_code: string | null; token_total: number | null; parameter_total: string | null; evolution_count: number; rank: number; benchmark: number | null; generalization: number | null; benchmark_delta: number | null; generalization_delta: number | null; suite_id: string | null; model_version: string | null; evaluated_at: string | null };
type StoredEntry = Entry & {owner:string;growth_model_version:string|null;display_profile:string|null};
type Sort = "evolution_count" | "benchmark" | "generalization" | "growth";
const counts = `SELECT e.*, (SELECT COUNT(*) FROM upgrades u WHERE u.owner=e.owner) AS accepted_count FROM entries e WHERE visible=1`;
// Operator-authored display rows are isolated from installations and upgrade receipts.
// A real installation using the same name takes precedence over a display row.
const displayRows = `SELECT * FROM display_entries d WHERE NOT EXISTS
  (SELECT 1 FROM entries e WHERE e.name_key=d.name_key AND e.visible=1)`;
const combined = `SELECT e.owner,e.username,e.country_code,e.token_total,e.parameter_total,e.accepted_count AS evolution_count,e.created_at,
  r.benchmark,r.generalization,r.benchmark_delta,r.generalization_delta,r.suite_id,r.model_version,r.completed_at AS evaluated_at,e.growth_model_version,NULL AS display_profile
  FROM (${counts}) e LEFT JOIN evaluation_receipts r ON r.id=e.active_evaluation AND r.owner=e.owner
    AND r.suite_id='${evaluations.metadata.suite_id}' AND r.suite_sha256='${evaluations.metadata.suite_sha256}' AND r.policy_hash='${evaluations.metadata.policy_hash}'
  UNION ALL SELECT 'display:' || id,username,country_code,token_total,NULL,evolution_count,created_at,benchmark,generalization,NULL,NULL,NULL,NULL,NULL,NULL,growth_profile FROM (${displayRows})`;
const fields = "owner,growth_model_version,display_profile,username,country_code,token_total,parameter_total,evolution_count,rank,benchmark,generalization,benchmark_delta,generalization_delta,suite_id,model_version,evaluated_at";
const ranked = (sort: Sort, domain:RankingDomain="mathematics") => {
  const conditions=`r.owner=c.owner AND r.model_version=c.growth_model_version AND json_extract(r.summary,'$.suite_id')='${growth.metadata.suite_id}' AND json_extract(r.summary,'$.policy_hash')='${growth.metadata.policy_hash}' AND json_extract(r.summary,'$.template_sha256')='${growth.metadata.template_sha256}'`;
  const selected=domain==="all"?`(SELECT r.domain FROM growth_publications p JOIN growth_receipts r ON r.id=p.receipt_id AND r.owner=p.owner WHERE ${conditions} ORDER BY r.created_at DESC,r.domain ASC LIMIT 1)`:`'${domain}'`;
  const displayFilter=domain==="all"?"json_extract(value,'$.growth') IS NOT NULL":"json_extract(value,'$.domain')='"+domain+"'";
  const summary=`COALESCE(g.summary,(SELECT json_extract(value,'$.growth') FROM json_each(c.display_profile) WHERE ${displayFilter} ORDER BY json_extract(value,'$.growth.completed_at') DESC,json_extract(value,'$.domain') ASC LIMIT 1))`;
  const scoped=`SELECT c.*,${summary} AS growth_summary,json_extract(${summary},'$.net_delta') AS growth_value FROM (${combined}) c
    LEFT JOIN growth_publications p ON p.owner=c.owner AND p.domain=${selected}
    LEFT JOIN growth_receipts g ON g.id=p.receipt_id AND g.owner=c.owner AND g.model_version=c.growth_model_version AND g.domain=p.domain
      AND json_extract(g.summary,'$.suite_id')='${growth.metadata.suite_id}' AND json_extract(g.summary,'$.policy_hash')='${growth.metadata.policy_hash}' AND json_extract(g.summary,'$.template_sha256')='${growth.metadata.template_sha256}'`;
  const filter=sort==="growth"?"WHERE growth_value IS NOT NULL AND json_extract(growth_summary,'$.eligible')=1":sort==="evolution_count"?"":`WHERE ${sort} IS NOT NULL`;
  return `SELECT *,ROW_NUMBER() OVER (ORDER BY ${sort==="growth"?"growth_value":sort} DESC,created_at ASC,owner ASC) AS rank FROM (${scoped}) ${filter}`;
};
const totalQuery = db.query<{ total: number; updated_at: string | null }, []>(
  `SELECT COUNT(*) AS total, MAX(updated_at) AS updated_at FROM
  (SELECT updated_at FROM entries WHERE visible=1 UNION ALL SELECT updated_at FROM (${displayRows}))`);
const myQuery = db.query<StoredEntry, [string]>(`SELECT ${fields} FROM (${ranked("evolution_count")}) WHERE owner=?`);
function publicEntry(entry:StoredEntry,domain:RankingDomain="mathematics") {
  const {owner,growth_model_version,display_profile,...publicFields}=entry;
  const growth_profile:ReturnType<typeof growth.profile>=display_profile?JSON.parse(display_profile):growth.profile(owner,growth_model_version);
  const current=domain==="all"?growth_profile.filter(value=>value.growth!==null).toSorted((a,b)=>b.growth!.completed_at.localeCompare(a.growth!.completed_at)||a.domain.localeCompare(b.domain))[0]:growth_profile.find(value=>value.domain===domain);
  return {...publicFields,username:maskName(publicFields.username),growth:current?.growth??null,growth_profile,growth_history:display_profile?[]:growth.history(owner)};
}

const proofFields = ["base_sha256", "weights_sha256", "baseline_validation_loss", "validation_loss", "replay_before", "replay_loss", "training_samples", "validation_samples", "completed_at"] as const;
type Proof = { [K in typeof proofFields[number]]: K extends "base_sha256" | "weights_sha256" ? string : K extends "replay_before" | "replay_loss" ? number | null : number } & { kind?: "core" };
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
function validatedProof(input: unknown): Proof {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("升级记录格式无效");
  const p = input as Record<string, unknown>;
  if ((Object.hasOwn(p,"kind") && p.kind !== "core") || Object.keys(p).length !== proofFields.length + (p.kind === "core" ? 1 : 0) || proofFields.some(field => !Object.hasOwn(p, field))) throw new Error("升级记录字段不完整");
  if (![p.base_sha256, p.weights_sha256].every(value => typeof value === "string" && /^[0-9a-f]{64}$/.test(value))) throw new Error("升级记录校验值无效");
  if (!finite(p.baseline_validation_loss) || p.baseline_validation_loss <= 0 || !finite(p.validation_loss) || p.validation_loss < 0 || p.validation_loss >= p.baseline_validation_loss * 0.995) throw new Error("升级记录未达到独立检验改善要求");
  if (p.replay_before !== null || p.replay_loss !== null) {
    if (!finite(p.replay_before) || p.replay_before < 0 || !finite(p.replay_loss) || p.replay_loss < 0 || p.replay_loss > p.replay_before * 1.05 + 0.01) throw new Error("升级记录未达到重放保持要求");
  }
  if (![p.training_samples, p.validation_samples].every(value => typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= (p.kind === "core" ? 10000 : 16))) throw new Error("升级记录需要独立训练与检验材料");
  if (!finite(p.completed_at) || p.completed_at <= 0 || p.completed_at > Date.now() / 1000 + 300) throw new Error("升级完成时间无效");
  return { ...Object.fromEntries(proofFields.map(field => [field, p[field]])), ...(p.kind === "core" ? {kind:"core" as const} : {}) } as Proof;
}

class ReportError extends Error {
  constructor(message: string, readonly status = 400) { super(message); }
}
const receiveReport = db.transaction((owner: string, username: string, country: string | null | undefined, proofs: Proof[], receipt: string | null, tokenTotal: number | null | undefined, parameterTotal: string | null | undefined, growthReceipt:ReturnType<typeof growth.validatePublication>|null, growthModel:string|null) => {
  const now = new Date().toISOString();
  db.query(`INSERT INTO entries (owner,username,name_key,evolution_count,created_at,updated_at,visible)
    VALUES (?,?,?,0,?,?,1) ON CONFLICT(owner) DO UPDATE SET username=excluded.username,
    name_key=excluded.name_key, updated_at=excluded.updated_at, visible=1`).run(owner, username, username.toLowerCase(), now, now);
  if (country !== undefined) db.query("UPDATE entries SET country_code=? WHERE owner=?").run(country, owner);
  db.query("UPDATE entries SET active_evaluation=? WHERE owner=?").run(receipt, owner);
  db.query("UPDATE entries SET growth_model_version=? WHERE owner=?").run(growthModel,owner);
  if(growthReceipt) growth.publish(owner,growthReceipt);
  if (tokenTotal != null) db.query("UPDATE entries SET token_total=MAX(COALESCE(token_total,0),?) WHERE owner=?").run(tokenTotal, owner);
  if (parameterTotal != null) {
    const previous = db.query<{parameter_total:string|null},[string]>("SELECT parameter_total FROM entries WHERE owner=?").get(owner)?.parameter_total;
    const next = previous === null || previous === undefined ? parameterTotal : (BigInt(parameterTotal) > BigInt(previous) ? parameterTotal : previous);
    db.query("UPDATE entries SET parameter_total=? WHERE owner=?").run(next, owner);
  }
  for (const proof of proofs) {
    const id = new Bun.CryptoHasher("sha256").update(`${proof.base_sha256}:${proof.weights_sha256}`).digest("hex");
    const serialized = JSON.stringify(proof);
    const previous = db.query<{ owner: string; proof: string }, [string]>("SELECT owner,proof FROM upgrades WHERE id=?").get(id);
    if (previous) {
      if (previous.owner !== owner) throw new ReportError("这份升级记录已归属另一安装，不能复制上榜", 409);
      if (previous.proof !== serialized) throw new ReportError("已接收的升级记录不能改写", 409);
    } else db.query("INSERT INTO upgrades VALUES (?,?,?,?)").run(id, owner, serialized, now);
  }
  return publicEntry(myQuery.get(owner)!);
});

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: {
    "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
  } });
}
function pageNumber(url: URL) {
  const raw = url.searchParams.get("page") || "1";
  if (!/^[1-9]\d{0,5}$/.test(raw)) throw new Error("页码无效");
  return Number(raw);
}
function sortOrder(url: URL): Sort {
  const raw = url.searchParams.get("sort") || "growth";
  if (!["evolution_count", "benchmark", "generalization", "growth"].includes(raw)) throw new EvaluationError("排序选项无效");
  return raw as Sort;
}
function selectedDomain(url:URL):RankingDomain {
  const domain=url.searchParams.get("domain")||"mathematics";
  if(!Object.hasOwn(DOMAIN_LABELS,domain)) throw new EvaluationError("成长领域无效");
  return domain as RankingDomain;
}
function leaderboard(page: number, sort: Sort = "growth", domain:RankingDomain="mathematics") {
  const totals = totalQuery.get()!;
  const ranking=ranked(sort,domain);
  const total = sort === "evolution_count" ? totals.total : db.query<{ total: number }, []>(`SELECT COUNT(*) AS total FROM (${ranking})`).get()!.total;
  return { entries: db.query<StoredEntry,[number,number]>(`SELECT ${fields} FROM (${ranking}) ORDER BY rank LIMIT ? OFFSET ?`).all(100,(page-1)*100).map(entry=>publicEntry(entry,domain)), ...totals, total, page, page_size: 100, sort, domain,
    domain_supported:supportsGrowth(domain), protocol_version: 2, measurement: "accepted_branch_upgrade_records", count_scope: "accepted_neural_upgrade_records", evaluation: evaluations.metadata, growth_protocol:growth.metadata };
}
function evolutionSummary() {
  const rows = db.query<{parameter_total:string|null;accepted_count:number},[]>(`SELECT e.parameter_total,
    (SELECT COUNT(*) FROM upgrades u WHERE u.owner=e.owner) AS accepted_count
    FROM entries e WHERE e.visible=1`).all();
  let parameters = 0n;
  let evolutions = 0n;
  let clients = 0;
  for (const row of rows) {
    if (row.parameter_total !== null && /^\d+$/.test(row.parameter_total)) {
      parameters += BigInt(row.parameter_total);
      if (BigInt(row.parameter_total) > 0n) clients += 1;
    }
    evolutions += BigInt(row.accepted_count || 0);
  }
  // Display scale only: factory expert width 32 × hidden 2048 × three matrices.
  // Repeated training and operator-authored leaderboard rows remain included;
  // this is deliberately separate from actual reported added_parameters.
  const rankingRows = db.query<{evolution_count:number},[]>(`SELECT evolution_count FROM (${combined})`).all();
  const rankingEvolutions = rankingRows.reduce((sum,row)=>sum+BigInt(row.evolution_count),0n);
  const parametersPerEvolution = 3 * 2048 * 32;
  return { added_parameters: parameters.toString(), completed_evolutions: evolutions.toString(), clients,
    ranking_evolutions: rankingEvolutions.toString(), ranking_entries: rankingRows.length,
    parameter_evolution_scale: (rankingEvolutions * BigInt(parametersPerEvolution)).toString(), parameters_per_evolution: parametersPerEvolution,
    updated_at: totalQuery.get()?.updated_at ?? null };
}
const format = (n: number, language:Language) => n.toLocaleString(locale(language));
const score = (value: number | null, delta: number | null, language:Language) => value === null ? `<span class="untested">${text(language,"untested")}</span>` : `${value.toFixed(2)}${delta === null ? "" : `<small class="delta">${delta > 0 ? "+" : ""}${delta.toFixed(2)}</small>`}`;

function home(language:Language) {
  const downloadButton = (url:string, label:string, primary=false) => `<${url ? 'a' : 'span'} class="button ${primary ? 'button-primary' : 'button-secondary'} button-large" ${url ? `href="${escape(url)}" rel="noopener noreferrer"` : 'aria-disabled="true"'}>${escape(label)}</${url ? 'a' : 'span'}>`;
  const t=(key:keyof typeof messages.zh)=>escape(text(language,key));
  const languageChoice=`<form class="language-choice" action="/" method="get" aria-label="${t("languageLabel")}">${(["zh","en"] as const).map(key=>`<button type="submit" name="lang" value="${key}" aria-pressed="${language===key}">${t(key==="zh"?"languageChinese":"languageEnglish")}</button>`).join("")}</form>`;
  return `<!doctype html>
<html lang="${locale(language)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light">
<title>${t("pageTitle")}</title><meta name="description" content="${t("metaDescription")}"><meta name="keywords" content="${t("metaKeywords")}">
<meta name="robots" content="index,follow"><link rel="canonical" href="https://ai.maihh.net/?lang=${language}">
<link rel="alternate" hreflang="zh-CN" href="https://ai.maihh.net/?lang=zh"><link rel="alternate" hreflang="en" href="https://ai.maihh.net/?lang=en"><link rel="alternate" hreflang="x-default" href="https://ai.maihh.net/">
<meta property="og:type" content="website"><meta property="og:title" content="${t("ogTitle")}"><meta property="og:description" content="${t("ogDescription")}"><meta property="og:url" content="https://ai.maihh.net/?lang=${language}"><meta property="og:locale" content="${language==="zh"?"zh_CN":"en_US"}"><meta property="og:image" content="https://ai.maihh.net/assets/luban-zero-logo.png">
<link rel="icon" type="image/svg+xml" href="/favicon.svg"><link rel="stylesheet" href="/style.css?v=home-evolution-v4"><script src="/evolution.js?v=scale-v2" defer></script></head>
<body><a class="skip-link" href="#main">${t("skipToContent")}</a><div class="shell"><header><a href="/?lang=${language}" aria-label="${t("homeLabel")}"><img src="/assets/luban-zero-logo.png" width="166" height="52" alt="${t("logoAlt")}"></a><nav class="site-nav" aria-label="${t("homeLabel")}"><a href="#product">${t("navProduct")}</a><a href="#why">${t("navWhy")}</a><a href="/leaderboard?lang=${language}">${t("navLeaderboard")}</a><a class="nav-report" href="https://github.com/MaiHHConnect/lubanllm/blob/main/paper/Luban_Technical_Report.md">${t("navTechnicalReport")}</a><a class="nav-cta" href="#download">${t("navDownload")}</a></nav><div class="header-tools"><span class="model">luban0</span>${languageChoice}</div></header>
<main id="main" class="home-page"><section id="evolution" class="evolution-section" aria-labelledby="evolution-title"><div class="evolution-copy"><p class="eyebrow">${t("evolutionEyebrow")}</p><h1 id="evolution-title">${t("evolutionTitle")}</h1><p>${t("evolutionLead")}</p></div><div class="evolution-stat" aria-live="polite"><div class="evolution-number"><strong id="evolution-scale" class="evolution-value">—</strong><span class="evolution-unit">${t("evolutionScaleUnit")}</span></div><div class="evolution-details"><span><b id="evolution-completed">—</b><small>${t("evolutionCompletedLabel")}</small></span><span><b id="evolution-clients">—</b><small>${t("evolutionClientsLabel")}</small></span><a href="/leaderboard?sort=evolution_count&amp;lang=${language}">${t("navLeaderboard")} <span aria-hidden="true">↗</span></a></div><p class="evolution-method">${t("evolutionMethod")}</p><p id="evolution-status" class="evolution-status">${t("evolutionUpdating")}</p><time id="evolution-updated" class="evolution-updated" datetime=""></time></div></section>
<section id="product" class="hero"><div class="hero-copy"><p class="eyebrow">${t("heroEyebrow")}</p><h2>${t("heroTitle")}</h2><p class="hero-lead">${t("heroLead")}</p><div class="hero-actions"><a class="button button-primary" href="#evolution">${t("heroPrimary")}</a><a class="button button-quiet" href="#why">${t("heroSecondary")} <span aria-hidden="true">↓</span></a></div><p class="hero-status"><span class="status-dot" aria-hidden="true"></span>${t("heroStatus")}</p></div><div class="hero-visual" aria-hidden="true"><div class="orb-grid"></div><div class="orb"><span class="orb-core"></span><span class="orb-ring ring-one"></span><span class="orb-ring ring-two"></span><span class="orb-ring ring-three"></span></div><span class="visual-label label-top">LOCAL / PRIVATE</span><span class="visual-label label-bottom">LEARN · VERIFY · KEEP</span></div><div class="hero-proof"><div><strong>01</strong><span>${t("heroProofLocal")}</span></div><div><strong>02</strong><span>${t("heroProofLearn")}</span></div><div><strong>03</strong><span>${t("heroProofVerify")}</span></div></div></section>
<section id="why" class="story-section problem-section"><div class="section-intro"><p class="eyebrow">${t("problemEyebrow")}</p><h2>${t("problemTitle")}</h2><p>${t("problemLead")}</p></div><div class="story-grid"><article><span class="story-number">01</span><h3>${t("problemOneTitle")}</h3><p>${t("problemOneBody")}</p></article><article><span class="story-number">02</span><h3>${t("problemTwoTitle")}</h3><p>${t("problemTwoBody")}</p></article><article><span class="story-number">03</span><h3>${t("problemThreeTitle")}</h3><p>${t("problemThreeBody")}</p></article></div></section>
<section class="story-section solution-section"><div class="section-intro"><p class="eyebrow">${t("solutionEyebrow")}</p><h2>${t("solutionTitle")}</h2><p>${t("solutionLead")}</p></div><div class="solution-list"><article><span class="solution-index">01</span><div><h3>${t("solutionOneTitle")}</h3><p>${t("solutionOneBody")}</p></div><span class="solution-arrow" aria-hidden="true">↗</span></article><article><span class="solution-index">02</span><div><h3>${t("solutionTwoTitle")}</h3><p>${t("solutionTwoBody")}</p></div><span class="solution-arrow" aria-hidden="true">↗</span></article><article><span class="solution-index">03</span><div><h3>${t("solutionThreeTitle")}</h3><p>${t("solutionThreeBody")}</p></div><span class="solution-arrow" aria-hidden="true">↗</span></article></div></section>
<section class="story-section now-section"><div class="section-intro"><p class="eyebrow">${t("nowEyebrow")}</p><h2>${t("nowTitle")}</h2><p>${t("nowLead")}</p></div><div class="story-grid now-grid"><article><span class="story-number">01</span><h3>${t("nowOneTitle")}</h3><p>${t("nowOneBody")}</p></article><article><span class="story-number">02</span><h3>${t("nowTwoTitle")}</h3><p>${t("nowTwoBody")}</p></article><article><span class="story-number">03</span><h3>${t("nowThreeTitle")}</h3><p>${t("nowThreeBody")}</p></article></div></section>
<section class="story-section keep-section"><div class="section-intro"><p class="eyebrow">${t("keepEyebrow")}</p><h2>${t("keepTitle")}</h2><p>${t("keepLead")}</p></div><div class="keep-grid"><article><span class="keep-mark">↗</span><h3>${t("keepOneTitle")}</h3><p>${t("keepOneBody")}</p></article><article><span class="keep-mark">◎</span><h3>${t("keepTwoTitle")}</h3><p>${t("keepTwoBody")}</p></article><article><span class="keep-mark">□</span><h3>${t("keepThreeTitle")}</h3><p>${t("keepThreeBody")}</p></article></div></section>
<section id="download" class="download-section"><div><p class="eyebrow">${t("downloadEyebrow")}</p><h2>${t("downloadTitle")}</h2><p>${t("downloadBody")}</p><p class="download-meta">${t("downloadMeta")}</p>${downloads.direct || downloads.quark || downloads.github ? '' : `<p role="status">${t("downloadPending")}</p>`}</div><div class="download-actions">${downloadButton(downloads.direct,text(language,"downloadComplete"),true)}${downloadButton(downloads.quark,text(language,"downloadPrimary"))}${downloadButton(downloads.github,text(language,"downloadGithub"))}</div></section></main><footer class="home-footer"><div class="footer-meta">© ${t("footerBrand")} <span>luban0</span></div><div id="research" class="research-note"><p><strong>${t("researchHeading")}</strong></p><p><a href="https://zenodo.org/records/23585073">${t("researchTitle")}</a></p><p>${t("researchStatus")} · DOI: <a href="https://doi.org/10.5281/zenodo.23585073">10.5281/zenodo.23585073</a></p></div></footer></div></body></html>`;
}

function leaderboardPage(page: number, sort: Sort, domain:RankingDomain, language:Language) {
  const data = leaderboard(page, sort, domain), labels=domainNames[language], t=(key:keyof typeof messages.zh)=>escape(text(language,key));
  const rows = data.entries.map(entry => `<tr><td class="rank ${entry.rank <= 3 ? "leading" : ""}"><span>${String(entry.rank).padStart(2, "0")}</span></td><td><div class="participant"><button class="username profile-trigger" type="button" popovertarget="profile-${entry.rank}" aria-label="${t("profileViewLabel").replace("{name}",escape(entry.username))}">${escape(entry.username)}</button>${entry.country_code ? `<span class="country">${escape(localizedCountry(entry.country_code,language))}</span>` : ""}${domain==="all"&&entry.growth?`<span class="country">${escape(labels[entry.growth.domain])}</span>`:""}</div></td><td class="tokens">${entry.token_total === null ? "—" : format(entry.token_total,language)}</td><td class="count">${format(entry.evolution_count,language)}<span>${t("evolutionUnit")}</span></td><td class="score gain">${entry.growth===null?"—":`${entry.growth.net_delta>0?"+":""}${entry.growth.net_delta.toFixed(2)}`}</td><td class="score">${entry.growth?.retention==null?"—":`${entry.growth.retention.toFixed(2)}%`}</td><td class="score">${score(entry.benchmark,null,language)}</td></tr>`).join("");
  const pages = Math.ceil((data.total || 0) / 100), suffix = `&amp;sort=${sort}&amp;domain=${domain}&amp;lang=${language}`;
  const paging = pages > 1 ? `<nav class="pagination" aria-label="${t("paginationLabel")}">${page > 1 ? `<a href="/leaderboard?page=${page - 1}${suffix}">${t("previousPage")}</a>` : "<span></span>"}<span>${page} / ${pages}</span>${page < pages ? `<a href="/leaderboard?page=${page + 1}${suffix}">${t("nextPage")}</a>` : "<span></span>"}</nav>` : "";
  const sorting = `<nav class="sorting" aria-label="${t("sortingLabel")}">${([ ["growth","sortGrowth"], ["benchmark","benchmarkLabel"], ["evolution_count","evolutionCount"] ] as const).map(([key,label]) => `<a href="/leaderboard?sort=${key}&amp;domain=${domain}&amp;lang=${language}"${sort === key ? ' aria-current="page"' : ""}>${t(label)}</a>`).join("")}</nav>`;
  const options=Object.entries(labels).map(([key,label])=>`<option value="${key}"${domain===key?" selected":""}>${escape(label)}</option>`).join("");
  const domains=`<form class="domain-filter" action="/leaderboard" method="get"><input type="hidden" name="sort" value="${sort}"><input type="hidden" name="lang" value="${language}"><label for="growth-domain">${t("growthDomain")}</label><select id="growth-domain" name="domain">${options}</select><button type="submit">${t("view")}</button></form>`;
  const languageChoice=`<form class="language-choice" action="/leaderboard" method="get" aria-label="${t("languageLabel")}"><input type="hidden" name="sort" value="${sort}"><input type="hidden" name="domain" value="${domain}"><input type="hidden" name="page" value="${page}">${(["zh","en"] as const).map(key=>`<button type="submit" name="lang" value="${key}" aria-pressed="${language===key}">${t(key==="zh"?"languageChinese":"languageEnglish")}</button>`).join("")}</form>`;
  const empty=sort==="evolution_count"?t("emptyRanking"):sort==="growth"?t("emptyGrowth"):t("emptyBenchmark");
  return `<!doctype html>
<html lang="${locale(language)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light">
<title>${t("leaderboardPageTitle")}</title>
<meta name="description" content="${t("leaderboardMetaDescription")}"><meta name="keywords" content="${t("metaKeywords")}">
<meta name="robots" content="${page === 1 ? "index,follow" : "noindex,follow"}"><link rel="canonical" href="https://ai.maihh.net/leaderboard?lang=${language}">
<link rel="alternate" hreflang="zh-CN" href="https://ai.maihh.net/leaderboard?lang=zh"><link rel="alternate" hreflang="en" href="https://ai.maihh.net/leaderboard?lang=en"><link rel="alternate" hreflang="x-default" href="https://ai.maihh.net/leaderboard">
<meta property="og:type" content="website"><meta property="og:title" content="${t("leaderboardPageTitle")}"><meta property="og:description" content="${t("leaderboardMetaDescription")}"><meta property="og:url" content="https://ai.maihh.net/leaderboard?lang=${language}"><meta property="og:locale" content="${language==="zh"?"zh_CN":"en_US"}"><meta property="og:image" content="https://ai.maihh.net/assets/luban-zero-logo.png">
<link rel="icon" type="image/svg+xml" href="/favicon.svg"><link rel="stylesheet" href="/style.css?v=language-v1"></head>
<body><a class="skip-link" href="#main">${t("skipToContent")}</a><div class="shell">
<header><a href="/?lang=${language}" aria-label="${t("homeLabel")}"><img src="/assets/luban-zero-logo.png" width="166" height="52" alt="${t("logoAlt")}"></a><nav class="site-nav" aria-label="${t("homeLabel")}"><a href="/?lang=${language}#product">${t("navProduct")}</a><a href="/?lang=${language}#why">${t("navWhy")}</a><a class="nav-cta" href="/leaderboard?lang=${language}">${t("navLeaderboard")}</a><a class="nav-report" href="https://github.com/MaiHHConnect/lubanllm/blob/main/paper/Luban_Technical_Report.md">${t("navTechnicalReport")}</a></nav><div class="header-tools"><span class="model">luban0</span>${languageChoice}</div></header>
<main id="main" class="leaderboard-page">
<section id="evidence" class="evidence-section ranking-only"><div class="heading"><div><p class="eyebrow">${t("rankingEyebrow")}</p><h2>${t("rankingGlobal")}${t("rankingHeading")}</h2></div><p class="participants">${format(data.total || 0,language)}<span>${t("participantUnit")}</span></p></div>
${domains}${sorting}<section class="board" aria-label="${t("boardLabel")}"><table><thead><tr><th scope="col" class="rank">${t("rankColumn")}</th><th scope="col">${t("usernameColumn")}</th><th scope="col" class="tokens">${t("tokenTotal")}</th><th scope="col" class="count">${t("evolutionCount")}</th><th scope="col" class="score gain">${t("growthColumn")}</th><th scope="col" class="score">${t("retentionLabel")}</th><th scope="col" class="score">${t("benchmarkLabel")}</th></tr></thead><tbody>${rows || `<tr><td colspan="7" class="empty"><strong>${empty}</strong></td></tr>`}</tbody></table>${paging}</section>${data.entries.map(entry=>profileMarkup(entry,`profile-${entry.rank}`,labels,language)).join("")}
${domain==="all"&&sort==="growth"?`<p class="note">${t("allDomainsNote")}</p>`:""}<p class="note">${t("evidenceNote")}</p><p class="note">${t("scoringNote")}</p></section></main><footer>© ${t("footerBrand")} <span>luban0</span></footer></div></body></html>`;
}

async function readJson(request: Request, limit = 65536) {
  if (!request.headers.get("content-type")?.startsWith("application/json")) throw new EvaluationError("请求需要 JSON", 415);
  const raw = await request.text();
  if (new TextEncoder().encode(raw).length > limit) throw new EvaluationError("请求内容过大", 413);
  try { return JSON.parse(raw); } catch { throw new EvaluationError("JSON 格式无效"); }
}
function authenticatedOwner(request: Request) {
  const origin = request.headers.get("origin");
  if (origin && origin !== "https://ai.maihh.net") throw new EvaluationError("不允许的请求来源", 403);
  const token = request.headers.get("authorization")?.match(/^Bearer ([0-9a-f]{64})$/)?.[1];
  if (!token) throw new EvaluationError("需要有效的应用排行凭据", 401);
  return new Bun.CryptoHasher("sha256").update(token).digest("hex");
}

export const server = Bun.serve({
  hostname: process.env.LEADERBOARD_HOST || "127.0.0.1",
  port: Number(process.env.PORT || 18170),
  maxRequestBodySize: 3 * 1024 * 1024,
  async fetch(request) {
    const url = new URL(request.url);
    const method = request.method === "HEAD" ? "GET" : request.method;
    let response: Response;
    try {
      if (url.pathname === "/api/app-start" || url.pathname === "/api/admin/app-starts") {
        response = (await appStarts.handle(request))!;
      } else if (url.pathname === "/api/app-ad" || url.pathname.startsWith("/api/app-ad/image/") || url.pathname === "/api/admin/app-ad") {
        response = (await appAd.handle(request))!;
      } else if (method === "GET" && url.pathname === "/app-ad/luban-intro-v1.png") {
        const file = Bun.file(`${root}/public/app-ad/luban-intro-v1.png`);
        if (!await file.exists()) return json({ error: "广告图片尚未发布 / Advertisement image not published" }, 404);
        response = new Response(request.method === "HEAD" ? null : file, { headers: { "Content-Type": "image/png", "Content-Length": String(file.size), "Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*" } });
      } else if (url.pathname.startsWith("/api/enterprise/")) {
        return (await enterprise.handle(request)) || json({ error: "未找到接口" }, 404);
      } else if (method === "GET" && ["/qy", "/qy/"].includes(url.pathname)) {
        response = json({ service: "enterprise_authorization", ok: true });
        response.headers.set("X-Robots-Tag", "noindex, nofollow");
      } else if (method === "GET" && url.pathname === "/api/evolution/summary") {
        response = json(evolutionSummary());
      } else if (method === "GET" && url.pathname === "/") {
        const chosen=url.searchParams.get("lang");
        if(chosen!==null&&!validLanguage(chosen)) throw new EvaluationError("语言选项无效 / Invalid language");
        const language=await resolveLanguage(request,server.requestIP(request)?.address||null);
        const legacyRanking = ["page", "sort", "domain"].some(key => url.searchParams.has(key));
        response = new Response(legacyRanking ? leaderboardPage(pageNumber(url),sortOrder(url),selectedDomain(url),language) : home(language), { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Language":locale(language), "Cache-Control": "private, no-store", "Vary":"Cookie, Accept-Language" } });
        if(validLanguage(chosen)) response.headers.set("Set-Cookie",`luban_lang=${chosen}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax`);
      } else if (method === "GET" && ["/leaderboard", "/leaderboard/"].includes(url.pathname)) {
        const chosen=url.searchParams.get("lang");
        if(chosen!==null&&!validLanguage(chosen)) throw new EvaluationError("语言选项无效 / Invalid language");
        const language=await resolveLanguage(request,server.requestIP(request)?.address||null);
        response = new Response(leaderboardPage(pageNumber(url),sortOrder(url),selectedDomain(url),language), { headers: { "Content-Type": "text/html; charset=utf-8", "Content-Language":locale(language), "Cache-Control": "private, no-store", "Vary":"Cookie, Accept-Language" } });
        if(validLanguage(chosen)) response.headers.set("Set-Cookie",`luban_lang=${chosen}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax`);
      } else if (method === "GET" && url.pathname === "/api/leaderboard") {
        response = json(leaderboard(pageNumber(url),sortOrder(url),selectedDomain(url)));
      } else if (request.method==="POST" && /^\/api\/leaderboard\/growth\/(challenge|before|after|retention)$/.test(url.pathname)) {
        const owner=authenticatedOwner(request),action=url.pathname.split("/").pop()!;
        const body=await readJson(request,action==="challenge"?4096:262144);
        response=json(action==="challenge"?growth.challenge(owner,body):action==="before"?growth.before(owner,body):action==="after"?growth.after(owner,body):growth.retention(owner,body));
      } else if (request.method === "POST" && ["/api/leaderboard/evaluation/challenge", "/api/leaderboard/evaluation"].includes(url.pathname)) {
        const owner = authenticatedOwner(request);
        const body = await readJson(request, url.pathname.endsWith("/challenge") ? 4096 : 262144);
        response = json(url.pathname.endsWith("/challenge") ? evaluations.challenge(owner,body) : evaluations.submit(owner,body));
      } else if (["PUT", "DELETE"].includes(request.method) && url.pathname === "/api/leaderboard/me") {
        const owner = authenticatedOwner(request);
        if (request.method === "DELETE") {
          // Retain receipt ownership so withdrawing/rejoining cannot replay a gain.
          db.query("UPDATE entries SET visible=0 WHERE owner = ?").run(owner);
          response = json({ ok: true });
        } else {
          const body = await readJson(request);
          if (!body || typeof body !== "object" || Array.isArray(body) || typeof body.username !== "string") return json({ error: "用户名格式无效" }, 400);
          const rawName = body.username.normalize("NFKC").trim();
          if (![...rawName].length || [...rawName].length > 16 || /\p{C}/u.test(rawName)) return json({ error: "用户名需为 1–16 个字符，不能包含控制字符" }, 400);
          const username = maskName(rawName);
          if ([...username].length > 16) return json({error:"敏感词替换后用户名超过16个字符，请缩短后重试"},400);
          if (Object.keys(body).some(key => !["username", "upgrades", "country_code", "model_version", "evaluation_receipt_id", "token_total", "parameter_total", "growth_receipt_id", "growth_model_version"].includes(key)) || !Object.hasOwn(body, "upgrades") || !Array.isArray(body.upgrades) || body.upgrades.length > 50) return json({ error: "次数由升级记录自动统计，请升级应用后上报；不接受手填次数或分数" }, 400);
          const country = body.country_code;
          if (Object.hasOwn(body,"country_code") && country !== null && (typeof country !== "string" || !COUNTRY_CODES.includes(country))) return json({error:"国家/地区选择无效"},400);
          if (Object.hasOwn(body,"token_total") && body.token_total !== null && (typeof body.token_total !== "number" || !Number.isSafeInteger(body.token_total) || body.token_total < 0 || body.token_total > 1e15)) throw new EvaluationError("累计 Token 需要有效的非负整数");
          if (Object.hasOwn(body,"parameter_total") && body.parameter_total !== null && (typeof body.parameter_total !== "string" || !/^(?:0|[1-9][0-9]{0,24})$/.test(body.parameter_total))) throw new EvaluationError("累计进化参数需要有效的非负整数");
          if (Object.hasOwn(body,"model_version") && (typeof body.model_version !== "string" || !/^[0-9a-f]{64}$/.test(body.model_version))) throw new EvaluationError("当前模型版本无效");
          if(Object.hasOwn(body,"growth_model_version")&&(typeof body.growth_model_version!=="string"||!/^[0-9a-f]{64}$/.test(body.growth_model_version))) throw new EvaluationError("完整本地模型指纹无效");
          const growthReceipt=Object.hasOwn(body,"growth_receipt_id")?growth.validatePublication(owner,body.growth_model_version,body.growth_receipt_id):null;
          let receipt=Object.hasOwn(body,"evaluation_receipt_id")?evaluations.validatePublication(owner,body.model_version,body.evaluation_receipt_id).receipt_id:null;
          if(!Object.hasOwn(body,"evaluation_receipt_id")&&growthReceipt&&body.model_version) {
            receipt=db.query<{id:string},[string,string]>("SELECT r.id FROM entries e JOIN evaluation_receipts r ON r.id=e.active_evaluation WHERE e.owner=? AND r.model_version=?").get(owner,body.model_version)?.id||null;
          }
          let proofs: Proof[];
          try { proofs = body.upgrades.map(validatedProof); } catch (error) { return json({ error: error instanceof Error ? error.message : "升级记录无效" }, 400); }
          try {
            response = json({ entry: receiveReport.immediate(owner, username, country, proofs, receipt, body.token_total, body.parameter_total, growthReceipt,body.growth_model_version||null), protocol_version: 2 });
          } catch (error: any) {
            if (error instanceof ReportError) return json({ error: error.message }, error.status);
            if (error.code === "SQLITE_CONSTRAINT_UNIQUE") return json({ error: "这个用户名已被使用，请换一个" }, 409);
            throw error;
          }
        }
      } else if (["GET", "HEAD"].includes(request.method) && /^\/updates\/(latest\.json|Luban-(?:[0-9.]+-macos-(?:arm64|x64)|[0-9.]+-from-[0-9.]+-macos-(?:arm64|x64)-delta)\.zip)$/.test(url.pathname)) {
        const file = Bun.file(`${root}/public${url.pathname}`);
        if (!await file.exists()) return json({ error: "更新文件尚未发布" }, 404);
        response = new Response(request.method === "HEAD" ? null : file, { headers: {
          "Content-Type": url.pathname.endsWith(".json") ? "application/json; charset=utf-8" : "application/zip",
          "Content-Length": String(file.size),
          "Cache-Control": url.pathname.endsWith(".json") ? "no-store" : "public, max-age=31536000, immutable",
        } });
      } else if (method === "GET" && url.pathname === "/health") {
        response = json({ ok: totalQuery.get() !== null });
      } else if (method === "GET" && url.pathname === "/robots.txt") {
        response = new Response("User-agent: *\nAllow: /\nDisallow: /qy\nDisallow: /api/\nDisallow: /health\nSitemap: https://ai.maihh.net/sitemap.xml\n", { headers: { "Content-Type": "text/plain; charset=utf-8" } });
      } else if (method === "GET" && url.pathname === "/sitemap.xml") {
        response = new Response('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://ai.maihh.net/?lang=zh</loc></url><url><loc>https://ai.maihh.net/?lang=en</loc></url><url><loc>https://ai.maihh.net/leaderboard?lang=zh</loc></url><url><loc>https://ai.maihh.net/leaderboard?lang=en</loc></url></urlset>', { headers: { "Content-Type": "application/xml" } });
      } else if (method === "GET" && ["/style.css", "/favicon.svg", "/assets/luban-zero-logo.png", "/evolution.js"].includes(url.pathname)) {
        const file = Bun.file(`${root}/public/${url.pathname.split("/").pop()}`);
        response = new Response(file, { headers: { "Cache-Control": "public, max-age=3600" } });
      } else response = json({ error: "未找到页面" }, 404);
    } catch (error) {
      if (error instanceof EvaluationError) response = json({error:error.message},error.status);
      else if (error instanceof Error && error.message === "页码无效") response = json({ error: error.message }, 400);
      else {
        console.error("Leaderboard request failed", error instanceof Error ? error.name : "unknown");
        response = json({ error: "排行榜暂时不可用，请稍后重试" }, 503);
      }
    }
    response.headers.set("X-Content-Type-Options", "nosniff");
    response.headers.set("Referrer-Policy", "same-origin");
    response.headers.set("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");
    if (url.pathname.startsWith("/api/") && !/^\/api\/app-ad\/image\/[0-9a-f]{64}\.png$/.test(url.pathname)) response.headers.set("Cache-Control", "no-store");
    return request.method === "HEAD" ? new Response(null, { status: response.status, headers: response.headers }) : response;
  },
});
console.log(`Luban0 leaderboard listening on ${server.hostname}:${server.port}`);
