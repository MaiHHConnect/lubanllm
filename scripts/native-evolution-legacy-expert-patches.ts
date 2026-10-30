/** Match Python's persisted expert loading; new Learning publication still requires its report. */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
type Replace = (body: string, before: string, after: string) => string;

export function patchLegacyExpertCollector(body: string, replace: Replace) {
  return replace(body, "files[name]=readSafe(store,name,16<<20);",
    "try{files[name]=readSafe(store,name,16<<20);}catch(error:any){if(error.code!=='ENOENT')throw error;}");
}
export function patchLegacyExpertValidation(body: string, replace: Replace) {
  return replace(body, `    need(reports.length === 1, "专家缺少唯一采纳报告");
    const report = json(files, reports[0]!);
    need(report.promoted === true && report.active_expert === active.id, "专家采纳报告绑定不一致"); expected.add(reports[0]!);`,
    `    need(reports.length <= 1, "专家采纳报告不唯一");
    if (reports.length) {
      const report = json(files, reports[0]!);
      need(report.promoted === true && report.active_expert === active.id, "专家采纳报告绑定不一致"); expected.add(reports[0]!);
    }`);
}
export async function applyNativeEvolutionLegacyExpertPatches(input: { bodies: Map<string, string>; target(path: string): string; replace: Replace; root: string }) {
  const { bodies, target, replace, root } = input;
  for (const [relative, patch] of [
    ["native-learning-product/model-files.ts", patchLegacyExpertCollector],
    ["native-product-runtime/backend/evolution-snapshot-files.ts", patchLegacyExpertValidation],
  ] as const) {
    const path = target(join(root, ".native-migration/research", relative));
    bodies.set(path, patch(bodies.get(path) ?? await readFile(path, "utf8"), replace));
  }
}
