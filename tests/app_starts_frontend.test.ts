import { expect, test } from "bun:test";

test("startup uses a persisted random installation ID, no credentials or content, and reports once per page", async () => {
  const env = globalThis as any;
  const before = { fetch: env.fetch, localStorage: env.localStorage };
  const saved = new Map<string, string>(); const calls: any[] = [];
  env.localStorage = { getItem: (k: string) => saved.get(k) ?? null, setItem: (k: string,v: string) => saved.set(k,v) };
  env.fetch = async (...args: any[]) => { calls.push(args); return new Response(null,{status:204}); };
  try {
    const source = await Bun.file(new URL('../frontend/app-starts.ts',import.meta.url)).text();
    const load = () => new Function(new Bun.Transpiler({loader:'ts'}).transformSync(source).replace('export async function', 'async function') + '\nreturn { reportAppStart };')();
    const { reportAppStart } = load();
    await reportAppStart(); await reportAppStart();
    expect(calls.length).toBe(1);
    expect(calls[0][0]).toBe('https://ai.maihh.net/api/app-start');
    const options = calls[0][1]; const payload = JSON.parse(options.body);
    expect(Object.keys(payload).sort()).toEqual(['event_id','install_id','schema']);
    expect(payload.install_id).toBe(saved.get('luban.anonymous-install'));
    expect(payload.event_id).not.toBe(payload.install_id);
    expect(options.credentials).toBe('omit'); expect(options.redirect).toBe('error');
    const second = load();
    await second.reportAppStart();
    expect(JSON.parse(calls[1][1].body).install_id).toBe(payload.install_id);
    env.localStorage.getItem = () => { throw Error('storage unavailable'); };
    await load().reportAppStart(); expect(calls.length).toBe(2);
  } finally { Object.assign(env,before); }
});
