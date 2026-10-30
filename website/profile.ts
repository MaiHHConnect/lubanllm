import { locale, localizedCountry, text, type Language } from "./i18n";
import type { GrowthDomain, GrowthSummary } from "./growth";

export function escape(value: string) {
  return value.replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!);
}

const number = (value: number | null, language: Language) => value === null ? "—" : value.toLocaleString(locale(language));
const decimal = (value: number, language: Language) => value.toLocaleString(locale(language), { minimumFractionDigits: 2, maximumFractionDigits: 2, useGrouping: false });
const percent = (value: number | null, language: Language) => value === null ? "—" : `${decimal(value, language)}%`;
const delta = (value: number, language: Language) => `${value > 0 ? "+" : ""}${decimal(value, language)}`;
const date = (value: string, language: Language) => {
  const time = new Date(value);
  return Number.isFinite(time.getTime()) ? new Intl.DateTimeFormat(locale(language), {timeZone:"Asia/Shanghai",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false}).format(time) : text(language, "timeUnrecorded");
};

export function profileMarkup(entry: {
  username: string; country_code: string | null; token_total: number | null; evolution_count: number;
  benchmark: number | null; growth: GrowthSummary | null;
  growth_profile: {domain:GrowthDomain;growth:GrowthSummary|null}[]; growth_history:GrowthSummary[];
}, id: string, labels: Record<string,string>, language: Language = "zh") {
  const t = (key: keyof typeof import("./i18n").messages.zh) => text(language, key);
  const count = (value: number | null) => number(value, language);
  const measured=entry.growth_profile.flatMap(value=>value.growth?[value.growth]:[]);
  const current=entry.growth??measured.toSorted((a,b)=>b.completed_at.localeCompare(a.completed_at)||a.domain.localeCompare(b.domain))[0];
  const changes=current?`<div class="profile-change"><p>${escape(labels[current.domain])} · ${t("profileEvaluation").replace("{count}", count(current.question_count))}</p><strong>${t("profileNewCorrect").replace("{count}", `<b>${count(current.wrong_to_correct)}</b>`)}</strong><dl class="profile-stats"><div><dt>${t("baselineCorrect")}</dt><dd>${count(current.baseline_correct)}<small> / ${count(current.question_count)}${t("questionUnit")}</small></dd></div><div><dt>${t("currentCorrect")}</dt><dd>${count(current.current_correct)}<small> / ${count(current.question_count)}${t("questionUnit")}</small></dd></div><div><dt>${t("abilityChange")}</dt><dd>${delta(current.net_delta, language)}<small>${t("percentagePoints")}</small></dd></div><div><dt>${t("regression")}</dt><dd>${count(current.correct_to_wrong)}<small>${t("questionUnit")}</small></dd></div><div><dt>${t("retentionLabel")}</dt><dd>${percent(current.retention, language)}</dd></div>${current.retention_check?`<div><dt>${t("retentionRecheck")}</dt><dd>${percent(current.retention_check.retention as number|null, language)}</dd></div>`:""}</dl>${current.eligible?"":`<p class="profile-empty">${t("profileExcluded")}</p>`}</div>`:`<p class="profile-empty">${t("profileEmpty")}</p>`;
  const domains=measured.map(value=>`<li><span>${escape(labels[value.domain])}</span><span>${t("domainCorrect").replace("{correct}", count(value.current_correct)).replace("{count}", count(value.question_count))}</span><strong>${delta(value.net_delta, language)}<small>${t("percentagePoints")}</small></strong></li>`).join("");
  const history=entry.growth_history.map(value=>`<li><div><time datetime="${escape(value.completed_at)}">${escape(date(value.completed_at, language))}</time><span>${escape(labels[value.domain])}</span><strong>${delta(value.net_delta, language)}<small>${t("percentagePoints")}</small></strong></div><p>${t("profileHistoryLine").replace("{before}", count(value.baseline_correct)).replace("{after}", count(value.current_correct)).replace("{count}", count(value.question_count)).replace("{improved}", count(value.wrong_to_correct)).replace("{regressed}", count(value.correct_to_wrong))}</p></li>`).join("");
  return `<section class="profile-card" id="${id}" popover role="dialog" aria-labelledby="${id}-title"><div class="profile-top"><div><p>${t("profileTitle")}</p><h2 id="${id}-title">${escape(entry.username)}</h2>${entry.country_code?`<span>${escape(localizedCountry(entry.country_code, language))}</span>`:""}</div><button class="profile-close" type="button" popovertarget="${id}" popovertargetaction="hide" aria-label="${t("profileClose")}" autofocus>×</button></div><section class="profile-section"><h3>${t("profileOverview")}</h3><dl class="profile-stats profile-overview"><div><dt>${t("evolutionCount")}</dt><dd>${count(entry.evolution_count)}<small>${t("evolutionUnit")}</small></dd></div><div><dt>${t("tokenTotal")}</dt><dd>${count(entry.token_total)}</dd></div><div><dt>${t("benchmarkLabel")}</dt><dd>${entry.benchmark===null?t("untested"):decimal(entry.benchmark, language)}</dd></div></dl>${changes}${domains?`<ul class="profile-domains" aria-label="${t("profileDomains")}">${domains}</ul>`:""}</section><section class="profile-section"><h3>${t("profileHistory")}</h3>${history?`<ol class="profile-history">${history}</ol>`:`<p class="profile-empty">${t("profileHistoryEmpty")}</p>`}</section></section>`;
}
