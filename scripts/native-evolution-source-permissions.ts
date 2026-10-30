/** Preserve numerical headers while admitting read-only exFAT accepted artifacts by held FD. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

export async function writeEvolutionSourcePermissions(root: string, output: string) {
  const research = join(root, ".native-migration/research"), records: any[] = [];
  async function overlay(relative: string, name: string, edit: (change: (before: string, after: string) => void) => void) {
    const source = join(research, relative), original = await readFile(source, "utf8");
    let body = original; const edits: { before: string; after: string }[] = [];
    const change = (before: string, after: string) => { assert.equal(body.split(before).length, 2, before); edits.push({ before, after }); body = body.replace(before, after); };
    for (const match of [...body.matchAll(/^#include "([^"]+)"/gm)]) change(match[0], "#include " + JSON.stringify(resolve(dirname(source), match[1]!)));
    edit(change);
    let inverse = body; for (const item of edits.toReversed()) { assert.equal(inverse.split(item.after).length, 2); inverse = inverse.replace(item.after, item.before); }
    assert.equal(inverse, original);
    const path = join(output, name); await writeFile(path, body); records.push({ source, path, original, body, edits, inverse_exact: true }); return path;
  }
  const permission = join(root, "native/AcceptedSourcePermissions.h");
  const expert = await overlay("native-unified-final-role-v1/ExpertResource.h", "EvolutionExpertResource.h", change => {
    change("#pragma once", "#pragma once\n#include " + JSON.stringify(permission));
    change('need(held->original.st_uid==getuid()&&(held->original.st_mode&07777)==0600&&hashFile(held->path)==hash,"Private expert complete file SHA differs");',
      'need(lubanAcceptedSourcePermissions(held->fd,held->original),"Expert source ownership/permissions differ");need(hashFile(held->path)==hash,"Expert complete file SHA differs");');
    change('need(source->original.st_uid==getuid()&&(source->original.st_mode&07777)==0600&&hashFile(path)==fileSHA,"Wake source SHA/mode differs");',
      'need(lubanAcceptedSourcePermissions(source->fd,source->original),"Wake source ownership/permissions differ");need(hashFile(path)==fileSHA,"Wake source complete file SHA differs");');
  });
  const router = await overlay("native-unified-router-resource/RouterResource.h", "EvolutionRouterResource.h", change => {
    change("#pragma once", "#pragma once\n#include " + JSON.stringify(permission));
    change('need(next->original.st_uid==getuid()&&(next->original.st_mode&07777)==0600&&hashFile(next->path)==sha,"Private complete router source differs");',
      'need(lubanAcceptedSourcePermissions(next->fd,next->original),"Router source ownership/permissions differ");need(hashFile(next->path)==sha,"Router complete file SHA differs");');
  });
  const graph = await overlay("native-evolution-retained-producer-v1/Graph.h", "EvolutionSourceGraph.h", change => {
    change("#include " + JSON.stringify(join(research, "native-unified-final-role-v1/ExpertResource.h")), "#include " + JSON.stringify(expert));
  });
  const routerCommands = await overlay("native-unified-final-role-v1/RouterCommands.h", "EvolutionRouterCommands.h", change => {
    change("#include " + JSON.stringify(join(research, "native-unified-router-resource/RouterResource.h")), "#include " + JSON.stringify(router));
  });
  return { permission, expert, router, graph, routerCommands, records };
}
