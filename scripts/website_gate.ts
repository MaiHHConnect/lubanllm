// Website release gate: render both languages in an isolated temporary database.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { Database } from "bun:sqlite";

const target=resolve(Bun.argv[2]||join(import.meta.dir,"../website"));
const temporary=mkdtempSync(join(tmpdir(),"luban-website-gate-"));
process.env.PORT="0";process.env.LEADERBOARD_HOST="127.0.0.1";
process.env.LEADERBOARD_DB=join(temporary,"gate.sqlite");
delete process.env.LEADERBOARD_TRUST_PROXY;
let server:{port:number;stop:(force:boolean)=>void}|undefined;
function assert(value:unknown,message:string):asserts value {if(!value)throw new Error(`WEBSITE GATE FAILED: ${message}`)}
function englishComplete(html:string) {
  // The Chinese language's native name is intentional in the language control.
  const checked=html.replace(/(<button\b[^>]*name="lang"[^>]*value="zh"[^>]*>)中文(<\/button>)/g,"$1Chinese$2");
  assert(!/\p{Script=Han}/u.test(checked),"English template contains untranslated Chinese");
  assert(!/\{(?:name|count|correct|before|after|improved|regressed)\}/.test(html),"Unfilled UI placeholders");
}
try {
  const i18n=await import(`${target}/i18n.ts`);
  i18n.assertBilingualCatalog();
  const downloads=await Bun.file(`${target}/downloads.json`).json();
  assert(Object.keys(downloads).sort().join() === 'direct,github,quark,version' && /^\d+\.\d+\.\d+$/.test(downloads.version),"Invalid download configuration");
  assert(typeof downloads.direct === "string" && (!downloads.direct || downloads.direct === `https://ai.maihh.net/downloads/Luban-${downloads.version}-macos-arm64-full.zip`),"Invalid complete installer URL");
  assert(typeof downloads.quark === 'string' && typeof downloads.github === 'string',"Download URLs must be strings");
  assert(!downloads.quark || /^https:\/\/pan\.quark\.cn\/s\/[a-zA-Z0-9]+$/.test(downloads.quark),"Invalid Quark download link");
  assert(!downloads.github || /^https:\/\/github\.com\/MaiHHConnect\/lubanllm\/releases\/tag\/v\d+\.\d+\.\d+(?:-beta\.\d+)?$/.test(downloads.github),"Invalid GitHub download link");
  assert(!downloads.github || downloads.github === `https://github.com/MaiHHConnect/lubanllm/releases/tag/v${downloads.version}` || downloads.github.startsWith(`https://github.com/MaiHHConnect/lubanllm/releases/tag/v${downloads.version}-beta.`),"GitHub release version must match downloads");
  const {profileMarkup}=await import(`${target}/profile.ts`);
  const growth={domain:"mathematics",question_count:24,baseline_correct:12,current_correct:18,baseline_score:50,current_score:75,wrong_to_correct:6,correct_to_wrong:0,net_delta:25,gap_recovery:50,retention:100,eligible:true,completed_at:"2026-10-10T00:00:00Z",retention_check:null};
  const entry={username:"GateUser",country_code:"US",token_total:123456,evolution_count:7,benchmark:75,growth,growth_profile:[{domain:"mathematics",growth}],growth_history:[growth]};
  for(const language of ["zh","en"] as const) {
    for(const variant of [entry,{...entry,growth:null,growth_profile:[],growth_history:[]},{...entry,growth:{...growth,eligible:false,retention:null,completed_at:"invalid",retention_check:{retention:50}},growth_history:[{...growth,completed_at:"invalid"}]}]) {
      const html=profileMarkup(variant,"gate",i18n.domainNames[language],language);
      if(language==="en") englishComplete(html);
      assert(html.includes(i18n.messages[language].profileOverview)&&html.includes(i18n.messages[language].profileHistory),"Growth profile localization missing");
      assert(!html.includes("<script"),"Unexpected script in profile");
    }
  }
  server=(await import(`${target}/server.ts`)).server;
  const db=new Database(process.env.LEADERBOARD_DB);
  const insert=db.query("INSERT INTO display_entries(id,username,name_key,country_code,evolution_count,created_at,updated_at,token_total,benchmark,generalization,growth_profile) VALUES(?,?,?,?,?,?,?,?,?,?,?)");
  db.transaction(()=>{for(let n=0;n<101;n++)insert.run(`gate${n}`,`Gate${n}`,`gate${n}`,"US",n,"2026-10-10","2026-10-10",123456,75,75,JSON.stringify([{domain:"mathematics",growth}]))}).immediate();
  db.close();
  const base=`http://127.0.0.1:${server.port}`;
  const summary=await (await fetch(`${base}/api/evolution/summary`)).json();
  assert(summary.added_parameters === "0" && summary.completed_evolutions === "0" && summary.clients === 0,"Evolution summary API must expose stable counters");
  assert(summary.ranking_evolutions === "5050" && summary.ranking_entries === 101,"Homepage must aggregate every leaderboard row, including page two");
  assert(summary.parameters_per_evolution === 196608 && summary.parameter_evolution_scale === "992870400","Parameter-evolution scale must preserve the disclosed factory-expert calculation");
  let pages=0;
  for(const language of ["zh","en"] as const) {
    const homeResponse=await fetch(`${base}/?lang=${language}`),homeHtml=await homeResponse.text();
    assert(homeResponse.ok&&homeHtml.includes(`<html lang="${i18n.locale(language)}">`),"Localized home page must open");
    assert(homeHtml.includes('src="/evolution.js?v=scale-v2"'),"Home evolution script missing");
    assert(homeHtml.includes(i18n.messages[language].downloadComplete)&&homeHtml.includes(i18n.messages[language].downloadPrimary)&&homeHtml.includes(i18n.messages[language].downloadGithub),"Two localized download choices missing");
    for(const url of [downloads.direct,downloads.quark,downloads.github]) if(url) assert(homeHtml.includes(`href="${url}"`),"Published download URL missing");
    for(const [channel,label] of [["direct","downloadComplete"],["quark","downloadPrimary"],["github","downloadGithub"]] as const) {
      if(!downloads[channel]) assert(new RegExp(`<span[^>]*aria-disabled="true"[^>]*>${i18n.messages[language][label].replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}<\\/span>`).test(homeHtml),"Unpublished download must remain disabled");
    }
    assert(homeHtml.includes(i18n.messages[language].downloadPending) === !Boolean(downloads.direct || downloads.quark || downloads.github),"Pending notice must match channel availability");
    assert(homeResponse.headers.get("content-security-policy")?.includes("script-src 'self'")&&homeResponse.headers.get("content-security-policy")?.includes("connect-src 'self'")&&!/<script(?![^>]+\bsrc=)/.test(homeHtml),"Website script policy changed");
    const evolutionScript=await fetch(`${base}/evolution.js`);
    assert(evolutionScript.ok&&evolutionScript.headers.get("content-type")?.includes("javascript"),"Evolution script route missing");
    const summary=await (await fetch(`${base}/api/evolution/summary`)).json();
    assert(typeof summary.added_parameters==="string"&&/^\d+$/.test(summary.added_parameters),"Evolution summary parameter count missing");
    if(language==="en") englishComplete(homeHtml);
    pages++;
    for(const domain of Object.keys(i18n.domainNames[language])) {
      for(const sort of ["growth","benchmark","evolution_count"]) {
        const response=await fetch(`${base}/leaderboard?lang=${language}&domain=${domain}&sort=${sort}`),html=await response.text();
        assert(response.ok,"Localized page must open");
        assert(response.headers.get("content-language")===i18n.locale(language),"Content-Language missing");
        assert(response.headers.get("cache-control")?.includes("no-store"),"Personalized language must not enter shared caches");
        assert(html.includes(`<html lang="${i18n.locale(language)}">`),"Wrong document language");
        assert(html.includes(`lang=${language}`)&&html.includes('hreflang="en"')&&html.includes('hreflang="zh-CN"'),"Language-aware navigation/SEO missing");
        assert(html.includes('class="leaderboard-page"')&&html.includes('ranking-only'),"Leaderboard route must use the focused ranking layout");
        assert(!html.includes('class="hero"')&&!html.includes('class="story-section"')&&!html.includes('class="download-section"'),"Leaderboard route must not include homepage marketing sections");
        assert(response.headers.get("content-security-policy")?.includes("script-src 'self'")&&!html.includes("<script"),"Leaderboard script policy changed");
        if(language==="en") englishComplete(html);
        pages++;
      }
    }
    const html=await (await fetch(`${base}/leaderboard?lang=${language}&sort=evolution_count&page=2`)).text();
    assert(html.includes(`href="/leaderboard?page=1&amp;sort=evolution_count&amp;domain=mathematics&amp;lang=${language}"`),"Pagination loses language");
    if(language==="en")englishComplete(html);
    pages++;
  }
  const counterDb=new Database(process.env.LEADERBOARD_DB);
  counterDb.query("INSERT INTO entries(owner,username,name_key,evolution_count,created_at,updated_at,visible,parameter_total) VALUES(?,?,?,?,?,?,?,?)").run("real-gate","Gate0","gate0",999,"2026-10-22","2026-10-22",1,"196608");
  counterDb.query("INSERT INTO entries(owner,username,name_key,evolution_count,created_at,updated_at,visible) VALUES(?,?,?,?,?,?,?)").run("hidden-gate","HiddenGate","hiddengate",999,"2026-10-22","2026-10-22",0);
  for(const id of ["one","two"]) counterDb.query("INSERT INTO upgrades(id,owner,proof,received_at) VALUES(?,?,?,?)").run(id,"real-gate","fixture","2026-10-22");
  counterDb.close();
  const mixed=await (await fetch(`${base}/api/evolution/summary`)).json();
  assert(mixed.ranking_entries===101&&mixed.ranking_evolutions==="5052","Homepage must use accepted receipts, suppress duplicate display names, and exclude hidden entries");
  assert(mixed.added_parameters==="196608"&&mixed.completed_evolutions==="2"&&mixed.clients===1,"Actual reported counters must remain separate from the display calculation");
  console.log(JSON.stringify({gate:"passed",languages:["zh","en"],catalog_keys:Object.keys(i18n.messages.zh).length,domains:17,page_checks:pages,profile_variants:6,temporary_data_only:true,model_calls:0}));
}finally {server?.stop(true);rmSync(temporary,{recursive:true,force:true})}
