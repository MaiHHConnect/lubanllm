import { expect, test } from "bun:test";
import { browserLanguage, publicIp, resolveLanguage } from "./language";

test("language preference overrides IP and browser without geolocation requests",async()=> {
  const lookup=()=>{throw new Error("Preference must not look up IP")};
  expect(await resolveLanguage(new Request("https://ai.maihh.net/?lang=en",{headers:{Cookie:"luban_lang=zh"}}),"8.8.8.8",lookup)).toBe("en");
  expect(await resolveLanguage(new Request("https://ai.maihh.net/",{headers:{Cookie:"x=y; luban_lang=zh; other=en"}}),"8.8.8.8",lookup)).toBe("zh");
});
test("country chooses Chinese regions, other countries English; failed lookup uses browser preference",async()=> {
  const request=new Request("https://ai.maihh.net/",{headers:{"Accept-Language":"en-US,en;q=.9"}});
  for(const country of ["CN","HK","MO","TW"])expect(await resolveLanguage(request,"8.8.8.8",async()=>country)).toBe("zh");
  expect(await resolveLanguage(new Request("https://ai.maihh.net/",{headers:{"Accept-Language":"zh-CN"}}),"8.8.8.8",async()=>"US")).toBe("en");
  expect(await resolveLanguage(request,"8.8.8.8",async()=>null)).toBe("en");
  expect(browserLanguage("en;q=0.3,zh-CN;q=0.8")).toBe("zh");
  expect(browserLanguage("fr,zh;q=0,en;q=0.8")).toBe("en");
  expect(browserLanguage("zh;q=2,en;q=0.1")).toBe("en");
});
test("proxy IP is accepted only under explicitly enabled trusted origin configuration",async()=> {
  const previous=process.env.LEADERBOARD_TRUST_PROXY;
  const request=new Request("https://ai.maihh.net/",{headers:{"X-Real-IP":"8.8.8.8","Accept-Language":"en"}});
  try {
    delete process.env.LEADERBOARD_TRUST_PROXY;
    expect(await resolveLanguage(request,"127.0.0.1",()=>{throw new Error("Do not trust forged proxy headers")})).toBe("en");
    process.env.LEADERBOARD_TRUST_PROXY="1";
    expect(await resolveLanguage(request,"127.0.0.1",async ip=>{expect(ip).toBe("8.8.8.8");return "CN"})).toBe("zh");
  }finally {if(previous===undefined)delete process.env.LEADERBOARD_TRUST_PROXY;else process.env.LEADERBOARD_TRUST_PROXY=previous}
});
test("private, reserved and malformed addresses are never sent to the lookup service",async()=> {
  for(const ip of ["127.0.0.1","10.0.0.1","172.16.1.2","192.168.0.1","169.254.1.2","100.64.0.1","224.0.0.1","192.0.2.1","198.51.100.1","203.0.113.1","::1","::","fc00::1","fe80::1","ff00::1","2001:db8::1","::ffff:127.0.0.1","::FFFF:127.0.0.1","not-an-IP"]) {
    expect(publicIp(ip)).toBe(false);
    expect(await resolveLanguage(new Request("https://ai.maihh.net/",{headers:{"Accept-Language":"zh"}}),ip,()=>{throw new Error("Private IP must not leave origin")})).toBe("zh");
  }
  for(const ip of ["8.8.8.8","1.1.1.1","2001:4860:4860::8888","::ffff:8.8.8.8"])expect(publicIp(ip)).toBe(true);
});
