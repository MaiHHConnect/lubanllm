import { isIP } from "node:net";
import type { Language } from "./i18n";

const chineseRegions = new Set(["CN", "HK", "MO", "TW"]);
const cache = new Map<string, { country: string | null; expires: number }>();
const pending = new Map<string, Promise<string | null>>();
export const validLanguage = (value: unknown): value is Language => value === "zh" || value === "en";

export function publicIp(value: string): boolean {
  if (/^::ffff:/i.test(value)) return publicIp(value.slice(7));
  if (isIP(value) === 4) {
    const [a,b,c]=value.split(".").map(Number);
    return !(a===0||a===10||a===127||a>=224||a===169&&b===254||a===172&&b>=16&&b<=31||a===192&&b===168||a===100&&b>=64&&b<=127||a===198&&[18,19].includes(b)||a===192&&b===0||a===198&&b===51&&c===100||a===203&&b===0&&c===113);
  }
  return isIP(value) === 6 && !/^(::(?:1)?$|f[cd]|fe[89ab]|ff|2001:db8:)/i.test(value);
}

async function countryLookup(ip: string): Promise<string | null> {
  if (!publicIp(ip)) return null;
  // Only country is retained in a bounded volatile cache, keyed by an IP hash.
  const key=new Bun.CryptoHasher("sha256").update(ip).digest("hex"),saved=cache.get(key);
  if(saved&&saved.expires>Date.now()) return saved.country;
  if(pending.has(key)) return pending.get(key)!;
  if(pending.size>=32) return null;
  const lookup=(async()=> {
    let country:string|null=null;
    try {
      const response=await fetch(`https://api.country.is/${encodeURIComponent(ip)}`,{signal:AbortSignal.timeout(1000),redirect:"error"});
      if(response.ok) {
        const body=await response.text();
        if(body.length<=4096) {
          const value=JSON.parse(body);
          if(typeof value.country==="string"&&/^[A-Z]{2}$/.test(value.country)) country=value.country;
        }
      }
    } catch { /* Optional geolocation must never prevent the page from opening. */ }
    if(cache.size>=1024) cache.delete(cache.keys().next().value!);
    cache.set(key,{country,expires:Date.now()+(country?6*3600000:300000)});
    return country;
  })();
  pending.set(key,lookup);
  try { return await lookup; } finally { pending.delete(key); }
}

export function browserLanguage(header:string|null):Language {
  const choices=(header||"").split(",").map((part,index)=> {
    const [tag,...params]=part.trim().split(";");
    const q=params.find(p=>p.trim().startsWith("q="))?.trim().slice(2);
    const weight=q===undefined?1:Number(q);
    return {language:tag.toLowerCase().split("-")[0],weight:Number.isFinite(weight)&&weight>=0&&weight<=1?weight:0,index};
  }).filter(choice=>choice.weight>0&&validLanguage(choice.language)).sort((a,b)=>b.weight-a.weight||a.index-b.index);
  return (choices[0]?.language as Language)||"en";
}

export async function resolveLanguage(request:Request,address:string|null,lookup=countryLookup):Promise<Language> {
  const explicit=new URL(request.url).searchParams.get("lang");
  if(validLanguage(explicit)) return explicit;
  const cookie=request.headers.get("cookie")?.split(";").map(value=>value.trim()).find(value=>value.startsWith("luban_lang="))?.slice(11);
  if(validLanguage(cookie)) return cookie;
  const ip=process.env.LEADERBOARD_TRUST_PROXY==="1"?request.headers.get("x-real-ip")||address:address;
  const country=ip&&publicIp(ip)?await lookup(ip):null;
  return country?chineseRegions.has(country)?"zh":"en":browserLanguage(request.headers.get("accept-language"));
}
