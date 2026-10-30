import { expect, test } from "bun:test";
import { english } from "../frontend/locales/en";

async function browserLanguage(saved: unknown, preferred: string, action = "", storageFails = false) {
  const modulePath = new URL("../frontend/i18n.ts", import.meta.url).pathname;
  const script = `
    let stored=${JSON.stringify(saved)}, reloads=0, callback, before=0, errors=[];
    const select={value:'',disabled:false,addEventListener(type,fn){callback=fn}};
    globalThis.window={localStorage:{getItem(){return stored},setItem(key,value){if(${storageFails})throw Error('denied');stored=value}},location:{reload(){reloads++}}};
    Object.defineProperty(globalThis,'navigator',{value:{language:${JSON.stringify(preferred)}},configurable:true});
    globalThis.document={getElementById(){return select}};
    const {language,locale,t,initializeLanguage}=await import(${JSON.stringify(modulePath)});
    initializeLanguage(async()=>{before++;${action === 'busy' ? "throw Error('busy')" : ""}},error=>errors.push(error.message));
    const user='首页 <script>alert(1)</script> {1}';
    const translated=t(['测试成功 · ',' · ',' 秒 · 回复：','。配置尚未保存。'],'model',1,user);
    ${action ? "select.value=language()==='zh'?'en':'zh';callback();await new Promise(resolve=>setTimeout(resolve,0));" : ""}
    console.log(JSON.stringify({language:language(),locale:locale(),title:t('教师训练'),translated,user,stored,reloads,before,errors,disabled:select.disabled,selected:select.value,unknown:t('custom-user-text')}));
  `;
  const child = Bun.spawn([process.execPath, "--eval", script], { stdout: "pipe", stderr: "pipe" });
  const output = await new Response(child.stdout).text();
  const error = await new Response(child.stderr).text();
  expect(await child.exited, error).toBe(0);
  return JSON.parse(output);
}

test("first-use language follows Chinese variants, while every other system language defaults to English", async () => {
  for (const preferred of ["zh", "zh-CN", "zh-Hant-TW", "ZH-hk"]) expect((await browserLanguage(null, preferred)).language).toBe("zh");
  for (const preferred of ["en-US", "fr-FR", "ja-JP", "", "zhfoo"]) expect((await browserLanguage(null, preferred)).language).toBe("en");
  expect((await browserLanguage("invalid", "fr-FR")).language).toBe("en");
  expect((await browserLanguage("zh", "en-US")).language).toBe("zh");
  expect((await browserLanguage("en", "zh-CN")).language).toBe("en");
});

test("English templates preserve user content and substitute once without translating answers or interpreting markup", async () => {
  const result = await browserLanguage("en", "zh-CN");
  expect(result.locale).toBe("en-US"); expect(result.title).toBe("Teacher training");
  expect(result.translated).toBe(`Test successful · model · 1 seconds · Reply: ${result.user}. Settings have not been saved.`);
  expect(result.unknown).toBe("custom-user-text");
});

test("switching saves the preference only after draft protection; busy and storage failure retain the current language", async () => {
  const saved = await browserLanguage("zh", "zh-CN", "switch");
  expect(saved.stored).toBe("en"); expect(saved.reloads).toBe(1); expect(saved.before).toBe(1);
  const busy = await browserLanguage("en", "en-US", "busy");
  expect(busy.stored).toBe("en"); expect(busy.reloads).toBe(0); expect(busy.errors).toEqual(["busy"]); expect(busy.disabled).toBe(false); expect(busy.selected).toBe("en");
  const denied = await browserLanguage("zh", "zh-CN", "switch", true);
  expect(denied.stored).toBe("zh"); expect(denied.reloads).toBe(0); expect(denied.errors.length).toBe(1); expect(denied.disabled).toBe(false);
});

test("English catalog retains every interpolation slot and learning cards use the requested label without single-character badges", async () => {
  for (const [key, value] of Object.entries(english)) {
    expect(value.trim().length, key).toBeGreaterThan(0);
    expect(value.match(/\{\d+\}/g)?.sort() || [], key).toEqual(key.match(/\{\d+\}/g)?.sort() || []);
  }
  const html = await Bun.file(new URL("../frontend/index.html", import.meta.url)).text();
  expect(html).not.toContain("feeding-choice-symbol"); expect(html).not.toContain("API 训练");
  expect(html).toContain("<strong>教师训练</strong>");
  expect(html).toMatch(/<input[^>]*data-no-translate[^>]*id="reset-phrase"/);
});
