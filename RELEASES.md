# Evidence release index

Git tracks source only (`results_*/`, `analysis_*/`, `REPORT*.md`, caches and
checkpoints are gitignored). A numeric claim is auditable only through an
**evidence release**: a `release_<tag>.tar.gz` plus its SHA-256 manifest,
published to a read-only location, containing per-seed records, configs,
environment, checkpoint hashes, failure/coverage records and the freeze lock.

No formal evidence release is bundled in this source snapshot. Every future
release must add a row here (and should be recorded in `DECISION_LOG.md`):

> **Status note (2026-11-26).** The v12 confirmatory results are RETIRED as a
> claim basis (owner decision; `DECISION_LOG.md` review 16, design_notes
> §33.1): no future release row may cite v12 numbers as evidence for
> CSA/HCA/Dynamic effectiveness or equal-compute superiority. Confirmatory
> evidence can only come from the follow-up protocol
> (`FOLLOWUP_CONFIRMATORY_PROTOCOL.md`); until that study is run, frozen,
> and released, this index remains empty by design.

| tag | date | Git commit | archive URL / object key | archive SHA-256 | manifest SHA-256 | verifier |
|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — |

Build, verify, publish and record:

```bash
python archive_release.py plan  v13 --dirs results_lm_test analysis_v12
python archive_release.py make  v13 --dirs results_lm_test analysis_v12 --include-ckpt
python archive_release.py verify archive/release_v13.tar.gz
# upload the tar.gz + .sha256 + .MANIFEST.sha256 to a read-only release/object
# store, then bind the published URL and both digests into this index:
python archive_release.py record v13 archive/release_v13.tar.gz \
    --url https://github.com/183600/dyn_csa_experiment/releases/download/evidence-v13/release_v13.tar.gz \
    --verifier "archive_release.py verify @ <verifier-log-url>"
```

`record` refuses to write a row unless `verify` passes, both digests are
64-hex, the commit is 7–40 hex, the date is ISO, and the URL starts with
http(s)://, s3:// or gs://.  It inserts the row into the index table; it never
rewrites an existing tag/URL/digest.  Uploading is deliberately an external action, so
the tool needs no GitHub credentials.

`archive_release.py` packs `DECISION_LOG.md`, `confirmation_frozen.json`,
`RELEASES.md`, `pyproject.toml`, `requirements.txt`, `requirements.lock` (if
present), cache/tokenizer metadata, every `run_*.log` at the repo root (if
present), the freeze lock and a per-file SHA-256 manifest; `verify` recomputes
every hash independently.

A release without a verifier URL or without the `verify` output should be
treated as **not auditable**, regardless of the numbers in its report.
`python audit_gates.py release` parses this table: with no complete row the
repository is reported as `no_release` (source-only), and every generated
report number is explicitly not independently auditable.
