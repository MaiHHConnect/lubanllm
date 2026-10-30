# Bundled Python runtime licenses

The product runtime uses CPython 3.12.12, python-build-standalone build 20260203,
and the dependency versions explicitly listed in `scripts/python-runtime-lock.json`.
The packager uses only standard library files and those distributions' installed
RECORD entries. It excludes environment activation files, user files, bytecode,
editable-install metadata and entry-point scripts. No developer venv is shipped.

License sources, retrieved unchanged on 2026-10-21 through the GitHub Contents API:

- `licenses/standalone-*`: matching files at
  https://github.com/astral-sh/python-build-standalone/tree/20260203
  (the `standalone-` prefix is the only filename change).
- `licenses/tokenizers-LICENSE`: the upstream Tokenizers 0.22.2 license at
  https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE,
  supplementing that wheel's omitted license file.
- All other dependency licenses/notices remain at their original wheel-relative
  paths in the runtime. CPython's own installed `lib/python3.12/LICENSE.txt`
  is retained as well.

Only package copies are relocated, stripped of debug symbols, and ad-hoc signed.
The local source interpreter, installed dependencies and model files are not changed.
The package verifier checks the source-derived allowlist, complete license content,
every Mach-O dependency and signature. A separate isolated interpreter probe
checks Python imports and CPU data operations. This is local relocatability
evidence; it does not claim Apple notarization or testing on every supported Mac.
