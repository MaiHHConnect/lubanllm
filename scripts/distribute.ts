import { lstat, readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { parseArgs } from "node:util";
import { assertReleasePayload } from "./release-payload";
import { readRelease } from "./release";

const root = resolve(import.meta.dir, ".."), repository = "MaiHHConnect/lubanllm";
const python = resolve(root, ".venv/bin/python");

async function run(argv: string[], capture = false) {
  const child = Bun.spawn(argv, { cwd: root, stdin: "ignore", stdout: capture ? "pipe" : "inherit", stderr: capture ? "pipe" : "inherit" });
  const [stdout, , code] = await Promise.all([
    capture ? new Response(child.stdout).text() : Promise.resolve(""),
    capture ? new Response(child.stderr).text() : Promise.resolve(""), child.exited]);
  // Never echo external CLI stderr: authentication failures can contain authorization URLs/tokens.
  if (code !== 0) throw Error(`${basename(argv[0]!)} 执行失败；未继续发布（退出码 ${code}）`);
  return stdout;
}

export function validateDownloadLink(kind: "quark" | "github", input: string) {
  const url = new URL(input);
  if (url.protocol !== "https:" || url.username || url.password || url.port || url.search || url.hash
      || (kind === "quark" ? url.hostname !== "pan.quark.cn" || !/^\/s\/[a-zA-Z0-9]+$/.test(url.pathname)
        : url.hostname !== "github.com" || !/^\/MaiHHConnect\/lubanllm\/releases\/tag\/v\d+\.\d+\.\d+(?:-beta\.\d+)?$/.test(url.pathname)))
    throw Error("下载链接必须是本项目GitHub版本页或夸克公开分享，不能包含凭据/查询参数");
  return url.href;
}

export function needsFinalPaper(tag: string, prerelease: boolean) {
  if (!/^v\d+\.\d+\.\d+(?:-beta\.[1-9]\d*)?$/.test(tag)
      || tag.includes("-beta.") !== prerelease) throw Error("内测标签必须对应prerelease，正式版本不得冒用内测通道");
  return !prerelease;
}

async function publicationGate(tag: string, prerelease: boolean) {
  await run([python, "scripts/paper_gate.py", ...(needsFinalPaper(tag, prerelease) ? ["--release"] : [])]);
}

async function regular(path: string) {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw Error("只允许明确列出的普通文件，不接受目录/符号链接");
}

async function checkedApp(app: string, notes?: string) {
  const release = await readRelease();
  if (basename(app) !== `Luban-${release.version}-macos-arm64.zip` && basename(app) !== `Luban-${release.version}-macos-arm64-full.zip`) throw Error("安装包文件名必须匹配当前release.json版本和arm64架构");
  await regular(app);
  if (notes) {
    if (basename(notes) !== "RELEASE-NOTES.md") throw Error("说明文件必须为RELEASE-NOTES.md");
    await regular(notes);
  }
  await run([python, "scripts/release-secrets.py", app, ...(notes ? [notes] : [])]);
  const receipt = await assertReleasePayload(app);
  if (!receipt.standalone) throw Error("下载入口只接受独立安装包，不能上传依赖开发机的更新候选");
  await run(["bun", "run", "website:gate"]);
  return { release, digest: "archiveSha256" in receipt ? receipt.archiveSha256 : undefined };
}

async function quark(argv: string[]) {
  const output = await run(["quarkclouddrive", ...argv], true);
  const rows = output.trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
  const result = rows.findLast(row => row.type === "result");
  if (!result || result.code !== 0) throw Error("夸克操作失败；未生成公开下载入口，已上传部分请通过CLI任务记录检查");
  if (result.notice?.text) console.log(result.notice.text);
  return result.data;
}

export async function main(args: string[]) {
  const { positionals, values } = parseArgs({ args, allowPositionals: true, strict: true, options: {
    app: { type: "string" }, notes: { type: "string" }, tag: { type: "string" },
    publish: { type: "boolean", default: false }, initial: { type: "boolean", default: false },
    quark: { type: "string" }, github: { type: "string" }, help: { type: "boolean" },
  } });
  const [command] = positionals;
  if (values.help || !command) {
    console.log("bun run distribute <check|github|quark|updates|links> --app /发行/Luban-VERSION-macos-arm64.zip [--notes /发行/RELEASE-NOTES.md] [--tag vVERSION-beta.1] [--publish]\n默认GitHub草稿、夸克私有上传；公开内测要求研究草稿及工程门禁，正式版本与应用更新仍要求正式研究门禁。links --quark URL --github URL 只在两端公开可访问后更新官网配置。");
    return;
  }
  if (positionals.length !== 1 || !["check", "github", "quark", "updates", "links"].includes(command)) throw Error("未知分发命令");
  if (command === "updates") {
    if (values.app || values.notes || values.tag || values.publish || values.quark || values.github) throw Error("updates只支持--initial，更新包由既有脚本构建和验证");
    await run(["bun", "scripts/release.ts", "upload", ...(values.initial ? ["--initial"] : [])]);
    return;
  }
  if (command === "links") {
    if (!values.quark || !values.github || values.app || values.notes || values.tag || values.publish || values.initial) throw Error("links只接受--quark和--github");
    const quarkURL = validateDownloadLink("quark", values.quark), githubURL = validateDownloadLink("github", values.github);
    const tag = new URL(githubURL).pathname.split("/").at(-1)!;
    const remote = JSON.parse(await run(["gh", "api", `repos/${repository}/releases/tags/${tag}`], true));
    const release = await readRelease();
    if (!new RegExp(`^v${release.version.replaceAll(".", "\\.")}(?:-beta\\.[1-9]\\d*)?$`).test(tag)) throw Error("下载标签必须对应当前版本");
    if (remote.draft || !remote.assets?.some((asset: { name: string }) => asset.name === `Luban-${release.version}-macos-arm64.zip`)) throw Error("GitHub版本尚未公开或缺少当前安装包");
    const detail = await quark(["share-detail", "--url", quarkURL]);
    if (!detail?.files?.some((file: {filename: string}) => file.filename === `Luban-${release.version}-macos-arm64-full.zip`)) throw Error("夸克分享中缺少当前完整安装包");
    await publicationGate(tag, remote.prerelease === true);
    const path = resolve(root, "website/downloads.json"), previous = await readFile(path, "utf8");
    try {
      await writeFile(path, JSON.stringify({ version: release.version, quark: quarkURL, github: githubURL, direct: JSON.parse(previous).direct ?? "" }, null, 2) + "\n");
      await run(["bun", "run", "website:gate"]);
    } catch (error) { await writeFile(path, previous); throw error; }
    console.log("官网双下载配置已更新；部署仍使用website/deploy/update.py。");
    return;
  }
  if (!values.app || values.initial || values.quark || values.github) throw Error("需要--app，不接受其他目标参数");
  const app = resolve(values.app), notes = values.notes ? resolve(values.notes) : undefined;
  const { release, digest } = await checkedApp(app, notes);
  if (command === "check") { if (values.publish || values.tag) throw Error("check不接受发布参数"); return; }
  const tag = values.tag ?? `v${release.version}-beta.1`;
  if (!new RegExp(`^v${release.version.replaceAll(".", "\\.")}(?:-beta\\.[1-9]\\d*)?$`).test(tag)) throw Error("tag必须对应当前版本");
  const prerelease = tag.includes("-beta.");
  if (values.publish) await publicationGate(tag, prerelease);
  if (command === "github") {
    if (!notes) throw Error("GitHub发行必须提供双语--notes");
    if (app.endsWith("-full.zip")) throw Error("GitHub单资产小于2GiB；完整包请上传官网和夸克，GitHub版本说明链接完整包");
    const releases = JSON.parse(await run(["gh", "api", `repos/${repository}/releases`], true));
    const existing = releases.find((item: { tag_name: string }) => item.tag_name === tag);
    if (existing && !existing.draft) throw Error("已公开版本禁止覆盖，请使用新版本");
    if (existing) needsFinalPaper(tag, existing.prerelease === true);
    const existingApp = existing?.assets?.find((item: { name: string }) => item.name === basename(app));
    if (existingApp && (!digest || existingApp.digest !== `sha256:${digest}`)) throw Error("同名资产摘要不同或无法验证，禁止覆盖；请提升版本");
    if (!existing) await run(["gh", "release", "create", tag, "--repo", repository, "--draft", ...(prerelease ? ["--prerelease"] : []), "--title", `Luban ${release.version}${prerelease ? " beta" : ""}`, "--notes-file", notes]);
    if (!existingApp) await run(["gh", "release", "upload", tag, app, "--repo", repository]);
    if (existing) await run(["gh", "release", "edit", tag, "--repo", repository, "--notes-file", notes]);
    if (values.publish) await run(["gh", "release", "edit", tag, "--repo", repository, "--draft=false"]);
    console.log(JSON.stringify({ github: `https://github.com/${repository}/releases/tag/${tag}`, draft: !values.publish }));
  } else {
    const uploaded = await quark(["upload", app, ...(notes ? [notes] : [])]);
    console.log(JSON.stringify({ quark_upload: true, files: uploaded.fileNames, fids: uploaded.fids }));
    if (values.publish) {
      if (!Array.isArray(uploaded.fids) || !uploaded.fids.length) throw Error("夸克没有返回文件ID，停止分享");
      const shared = await quark(["share", ...uploaded.fids, "--title", `Luban ${release.version} macOS${prerelease ? " beta" : ""}`, "--url-type", "1", "--expired-type", "1"]);
      console.log(JSON.stringify({ quark: validateDownloadLink("quark", shared.share_url) }));
    }
  }
}

if (import.meta.main) main(Bun.argv.slice(2)).catch(error => { console.error((error as Error).message); process.exitCode = 1; });
