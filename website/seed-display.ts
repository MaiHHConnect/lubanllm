import { Database } from "bun:sqlite";
import { COUNTRY_CODES } from "./countries";
import { maskName } from "./name-filter";
import { GROWTH_DOMAINS, type GrowthSummary } from "./growth";

// Synthetic operator-authored handles; never fabricate installations or training evidence.
const groups: [string, string][] = [
  ["US", `evan.reed Mason87 avaparks noahh liam_h mia.wren
    lucasb emma_lee oliver23 sophia.r ethanwells amelia.dev
    james_cedar charlotte.k benji84 harperwest henrybytes evelyn.j
    alex.nova abbyfox daniel_s loganpixel ella.moss scarlett.r
    jackson_loop gracecloud sebastian_k chloelane jack_v lilycove
    owensmith zoey.reed samriver nora.dev aidenpark rileyj
    wyatt92 hazelsky leo.brook ellietrace julianm stella.f
    isaac_v violetwren gabriel.p lucymiles anthony88 aurora.k
    dylanbridge audreylake luke_p bella.r nathantide claire.dev
    calebstone skylar_n ryan.field annaelm adambyte mayawells
    eli.cedar naomi_b aaronridge leahloop hudsonlane alicec
    miles.sage evathread dominic77 sadiewest connorr ruby_cove
    kai_v ivyparks jordan_s quinnriver carterelm piper_n
    colemoss sarahbytes tylerb clara.reed blakecove erin.k
    finnbranch mollycodes zach_w june.pixel maxtrace arlolane
    theo.byte lena_dev devonpark tesscloud wesridge rosenode
    asherc jadewren kylepatch iris_s bennettfox`],
  ["CA", "mapleowen lucanorth emily.cove felix93 sophielake aiden_t theo_maple clara.n julien42 norapines"],
  ["GB", "alfie_b freya_lane archie87 florence.k harrybytes isla.cove oscarreed poppy_n tobybrook"],
  ["DE", "leon1989 clara_berlin finnbyte emil94 lenak jonas_r"],
  ["AU", "oliver_gum matilda.k jackcove willow93 luca_surf evien"],
  ["FR", "lucas.paris chloe_d hugo92 ines.m"],
  ["JP", "haru_86 yukinode sora77 ren.kumo"],
  ["KR", "minjun_95 seoyun_n jiho.k"],
  ["IN", "arjun24 ananya_d rohanbytes diya.s"],
  ["BR", "joaov luiza_n pedro91"],
  ["NL", "daan84 lotte_v"], ["SE", "elias_n alma87"],
  ["SG", "wei_l rachel_n"], ["CN", "云杉小站 林间字节"],
  ["ES", "mateo.g lucia93"], ["IT", "luca_rossi giulia_m"],
  ["CH", "niko.alpine"], ["NZ", "kiwiollie"], ["MX", "diego_m"],
  ["AE", "omar_h"], ["ZA", "sipho87"],
];

export const displayEntries = groups.flatMap(([country_code, handles]) => handles.trim().split(/\s+/).map(username => {
  const id = new Bun.CryptoHasher("sha256").update(`ranking-display-20261009:${country_code}:${username}`).digest("hex");
  const sample = Number.parseInt(id.slice(0, 8), 16) / 0xffffffff;
  const created_at = new Date(Date.UTC(2026, 9, 2) + Number.parseInt(id.slice(8, 16), 16) % (7 * 86400000)).toISOString();
  const evolution_count = 1 + Math.floor(72 * sample ** 3);
  const percentile = (correct:number, total=24) => Math.round(10000 * correct / total) / 100;
  const growth_profile = GROWTH_DOMAINS.map(domain => {
    const hash = new Bun.CryptoHasher("sha256").update(`display-metrics-v1:${id}:${domain}`).digest("hex");
    const n = (offset:number) => Number.parseInt(hash.slice(offset,offset+4),16);
    if (n(0) % 13 === 0) return {domain,growth:null};
    const baseline_correct = 7 + n(4) % 12;
    const wrong_to_correct = n(8) % Math.min(7,25-baseline_correct);
    const correct_to_wrong = n(12) % 3;
    const current_correct = baseline_correct + wrong_to_correct - correct_to_wrong;
    const growth:GrowthSummary = {domain,question_count:24,baseline_correct,current_correct,
      baseline_score:percentile(baseline_correct),current_score:percentile(current_correct),
      wrong_to_correct,correct_to_wrong,net_delta:percentile(wrong_to_correct-correct_to_wrong),
      gap_recovery:percentile(wrong_to_correct,24-baseline_correct),retention:percentile(baseline_correct-correct_to_wrong,baseline_correct),
      eligible:true,completed_at:new Date(Date.UTC(2026,9,9) + n(16)*1000).toISOString(),retention_check:null};
    return {domain,growth};
  });
  return { id, username: maskName(username), country_code, evolution_count, created_at,
    token_total:Math.floor(12000 + evolution_count * (18000 + Number.parseInt(id.slice(16,20),16)*4) + sample*700000),
    benchmark:Number.parseInt(id.slice(20,24),16)%11===0?null:percentile(6+Number.parseInt(id.slice(24,28),16)%13),
    generalization:Number.parseInt(id.slice(28,32),16)%11===0?null:percentile(8+Number.parseInt(id.slice(32,36),16)%15),growth_profile };
}));

export function seedDisplayEntries(db: Database) {
  if (displayEntries.length !== 167 || new Set(displayEntries.map(e => e.username.toLowerCase())).size !== 167)
    throw new Error("展示用户必须恰好167条且名称唯一");
  for (const entry of displayEntries) {
    if (!COUNTRY_CODES.includes(entry.country_code) || [...entry.username].length > 16)
      throw new Error(`展示用户资料无效：${entry.username}`);
  }
  // Schema is created by the website server; require an explicit, existing database.
  const insert = db.query(`INSERT INTO display_entries(id,username,name_key,country_code,evolution_count,created_at,updated_at)
    VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING`);
  const update = db.query(`UPDATE display_entries SET token_total=?,benchmark=?,generalization=?,growth_profile=?,updated_at=? WHERE id=?`);
  return db.transaction(() => {
    let added = 0;
    for (const e of displayEntries) {
      added += insert.run(e.id,e.username,e.username.toLowerCase(),e.country_code,e.evolution_count,e.created_at,e.created_at).changes;
      update.run(e.token_total,e.benchmark,e.generalization,JSON.stringify(e.growth_profile),"2026-10-10T00:00:00.000Z",e.id);
    }
    return added;
  }).immediate();
}

if (import.meta.main) {
  const path = Bun.argv[2];
  if (!path || !await Bun.file(path).exists()) throw new Error("请指定已初始化的排行榜数据库路径");
  const db = new Database(path);
  try {
    console.log(JSON.stringify({ added: seedDisplayEntries(db), total: displayEntries.length,
      countries: Object.fromEntries(groups.map(([code, names]) => [code, names.trim().split(/\s+/).length])) }));
  } finally { db.close(); }
}
