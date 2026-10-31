# Evidence release index

Git tracks source only (`results_*/`, `analysis_*/`, `REPORT*.md`, caches and
checkpoints are gitignored). A numeric claim is auditable only through an
**evidence release**: a `release_<tag>.tar.gz` plus its SHA-256 manifest,
published to a read-only location, containing per-seed records, configs,
environment, checkpoint hashes, failure/coverage records and the freeze lock.

No formal evidence release is bundled in this source snapshot. Every future
release must add a row here (and should be recorded in `DECISION_LOG.md`):

| tag | date | Git commit | archive URL / object key | archive SHA-256 | manifest SHA-256 | verifier |
|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — |

Build and verify:

```bash
python archive_release.py plan  v13 --dirs results_lm_test analysis_v12
python archive_release.py make  v13 --dirs results_lm_test analysis_v12 --include-ckpt
python archive_release.py verify archive/release_v13.tar.gz
```

`archive_release.py` packs `DECISION_LOG.md`, `confirmation_frozen.json`,
`RELEASES.md`, `pyproject.toml`, `requirements.txt`, `requirements.lock` (if
present), `design_protocol.json` (if present), cache/tokenizer metadata,
every `run_*.log` at the repo root (if present), the freeze lock, the measurement source files and a per-file
SHA-256 manifest. The manifest also carries an `identity` block (the source
tree digest, `CODE_SEMANTICS`, dependency contract and execution fingerprint);
`verify` recomputes every file hash and the source-tree digest from the packed
bytes, independently of Git and the current worktree. Archives without
`identity` are legacy: their file hashes still verify, but their source
binding is explicitly reported as unverifiable.

A release without a verifier URL or without the `verify` output should be
treated as **not auditable**, regardless of the numbers in its report.
`python audit_gates.py release` parses this table: with no complete row the
repository is reported as `no_release` (source-only), and every generated
report number is explicitly not independently auditable.
