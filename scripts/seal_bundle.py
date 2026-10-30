#!/usr/bin/env python3
"""Record hashes after updating the manuscript and rerunning evidence.

Sealing never changes draft status, fabricates results, or approves publication.
"""
import json
from paper_gate import BUNDLE, ROOT, Snapshot, digest
from mindmap_gate import check as check_mindmap

snapshot = Snapshot()
errors = check_mindmap(snapshot)
if errors:
    raise SystemExit("\n".join(errors))
bundle = json.loads(snapshot.read(BUNDLE))
bundle["files"] = {p: digest(snapshot.read(p)) for p in sorted(snapshot.paths())}
bundle["paper_sha256"] = digest(snapshot.read(bundle["paper"]))
for evidence in bundle["evidence"]:
    evidence["sha256"] = digest(snapshot.read(evidence["path"]))
(ROOT / BUNDLE).write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
print("Bundle sealed. Status remains: " + bundle["status"])
