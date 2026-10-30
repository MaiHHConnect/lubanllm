import { expect, test } from "bun:test";
import { main, needsFinalPaper, validateDownloadLink } from "../scripts/distribute";

test("只有显式内测标签与prerelease一致时才独立于正式论文发布", () => {
  expect(needsFinalPaper("v0.3.217-beta.1", true)).toBe(false);
  expect(needsFinalPaper("v0.3.217", false)).toBe(true);
  expect(() => needsFinalPaper("v0.3.217-beta.1", false)).toThrow();
  expect(() => needsFinalPaper("v0.3.217", true)).toThrow();
  for (const tag of ["untagged-abc", "v0.3.217-beta.0", "v0.3.217-preview", "v0.3.217-beta.1/other"])
    expect(() => needsFinalPaper(tag, true)).toThrow();
});

test("官网下载地址只接受本项目发行和夸克公开分享", () => {
  expect(validateDownloadLink("quark", "https://pan.quark.cn/s/abc123")).toBe("https://pan.quark.cn/s/abc123");
  expect(validateDownloadLink("github", "https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.211-beta.1")).toContain("v0.3.211-beta.1");
  for (const link of ["http://pan.quark.cn/s/abc", "https://pan.quark.cn/s/abc?token=private", "https://user:pass@pan.quark.cn/s/abc", "https://evil.test/s/abc"])
    expect(() => validateDownloadLink("quark", link)).toThrow();
  for (const link of ["https://github.com/else/repo/releases/tag/v0.3.211", "https://github.com/MaiHHConnect/lubanllm/releases/tag/untagged-abc", "https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.211#token"])
    expect(() => validateDownloadLink("github", link)).toThrow();
});

test("参数错误在任何上传、登录或构建操作前拒绝", async () => {
  for (const args of [["github", "--app", "/tmp/.env"], ["quark", "--directory", "/tmp"], ["updates", "--app", "/tmp/app.zip"], ["links", "--quark", "https://pan.quark.cn/s/abc"], ["unknown"]])
    await expect(main(args)).rejects.toThrow();
});
