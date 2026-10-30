/** Native syntax parser only: generated Python is never executed. */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { cleanRuntimeEnvironment } from "./native-process.ts";

let parser: string | undefined;
function executable() {
  if (parser) return parser;
  const paths = [join(import.meta.dir, "Resources/native/bin/syntax-parser"),
    join(import.meta.dir, "../native/bin/syntax-parser"),
    join(import.meta.dir, "native/luban-python-upper"), join(import.meta.dir, "../build/native/luban-python-upper")];
  const found = paths.find(path => existsSync(path));
  if (!found) throw new Error("原生语法检验程序缺失，请重新构建应用");
  // The build/release payload gate owns artifact admission, as for the other
  // shipped native executables. Do not hash/reopen it for each validation row.
  parser = found;
  return parser;
}

export function pythonUpper(answer: string): boolean {
  const match = /^```(?:python|py|json)?[ \t]*\n([\s\S]*?)\n```$/i.exec(answer);
  const source = match ? match[1]! : answer;
  if (source.includes("\0") || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(source))
    throw new Error("原生语法检验输入无效");
  const input = Buffer.from(source);
  if (input.length > 65536) throw new Error("原生语法检验输入超过上限");
  const child = Bun.spawnSync([executable()], { stdin: input, stdout: "pipe", stderr: "pipe",
    env: cleanRuntimeEnvironment(), timeout: 2000 });
  if (child.exitCode !== 0 || child.signalCode || child.stderr.length || !/^(true|false)\n$/.test(child.stdout.toString()))
    throw new Error("原生语法检验程序失败：" + (child.signalCode ?? child.exitCode));
  return child.stdout.toString() === "true\n";
}
