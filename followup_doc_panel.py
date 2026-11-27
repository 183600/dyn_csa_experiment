#!/usr/bin/env python3
"""Frozen stratified document-panel sampler for the follow-up confirmatory study.

FOLLOWUP_CONFIRMATORY_PROTOCOL.md ("Evaluation population") requires, before
any model evaluation on the test split:

    "Use the complete held-out test split where feasible.  If a sampled
     document panel is required, sample at least 64 distinct documents,
     stratified by document length, and freeze the document IDs before
     model evaluation."

and the design record carries two booleans about exactly that
(``evaluation.document_ids_frozen_before_test_access`` /
``freeze.test_document_ids_frozen``).  Until now there was NO producer for a
frozen document panel: the booleans were hand-settable exactly like the
pre-review-16 ``power_simulation_completed`` flag was.  This tool is the
missing producer, and ``validate_followup_design.py`` now binds those
booleans to its artifact (a design that declares frozen document IDs must
point at a real, non-synthetic, big-enough, stratified panel this tool
froze).

Metadata-only access, by construction: the sampler reads ONLY the test-split
sidecars ``test_doc_ids_<tag>.npy`` and ``test_doc_lengths_<tag>.npy``
(document identity + length).  It never loads the token-id stream
(``test_ids_<tag>.npy``) — the held-out CONTENT stays unread until the frozen
study itself evaluates it.  The artifact records every file it read with its
SHA-256 so the metadata access is auditable, not silent.

If train/val document-id sidecars sit in the same cache directory, the panel
is cross-checked against them and ANY overlap is a hard failure (a document
that leaked into training or the tuning split can never enter the frozen
evaluation panel); missing sidecars are disclosed, never assumed clean.

Sampling design (deterministic, recorded): documents are stratified by
length into ``--strata`` equal-count strata over the length-sorted roster;
``--min-docs`` are allocated proportionally to stratum sizes (largest
remainder, every stratum keeps at least one document); within a stratum the
sample is drawn without replacement under the recorded ``--seed``.  When the
whole split has no more documents than the request, the panel takes the
entire split and the artifact says so (the "complete held-out split" case).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

TOOL_NAME = 'followup_doc_panel.py'
SCHEMA_VERSION = 1
DEFAULT_MIN_DOCS = 64
DEFAULT_STRATA = 4


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_sidecars(sidecar_dir: Path):
    """Read ONLY the doc-id and doc-length sidecars (never the token ids).

    Returns (doc_ids [uint64], doc_lengths [int64], read_files {name: sha256}).
    Fail-closed on: missing files, ambiguous tags, length mismatch,
    non-positive lengths, duplicate ids.
    """
    import numpy as np
    ids_files = sorted(sidecar_dir.glob('test_doc_ids_*.npy'))
    lens_files = sorted(sidecar_dir.glob('test_doc_lengths_*.npy'))
    problems = []
    if not ids_files:
        problems.append('no test_doc_ids_*.npy sidecar found — the test '
                        'split cache must be built (load_wikitest) before a '
                        'panel can be frozen from it')
    if not lens_files:
        problems.append('no test_doc_lengths_*.npy sidecar found')
    if problems:
        raise RuntimeError('; '.join(problems) +
                          f' (searched: {sidecar_dir})')
    if len(ids_files) > 1 or len(lens_files) > 1:
        # Multiple tag variants: refuse to guess which split is meant.
        raise RuntimeError(
            f'ambiguous test sidecars: {len(ids_files)} doc-id files '
            f'{[p.name for p in ids_files]} / {len(lens_files)} length files '
            f'{[p.name for p in lens_files]} — remove the stale cache or '
            f'pass the exact --sidecar-dir holding the single split')
    ids_path, lens_path = ids_files[0], lens_files[0]
    doc_ids = np.load(ids_path)
    doc_lengths = np.load(lens_path)
    if doc_ids.ndim != 1 or doc_lengths.ndim != 1:
        raise RuntimeError('sidecars must be 1-D arrays '
                           f'(ids {doc_ids.shape}, lengths {doc_lengths.shape})')
    if doc_ids.shape[0] != doc_lengths.shape[0]:
        raise RuntimeError(f'sidecar row mismatch: {doc_ids.shape[0]} ids vs '
                           f'{doc_lengths.shape[0]} lengths')
    if doc_ids.size == 0:
        raise RuntimeError('the test split sidecar has zero documents')
    if not np.issubdtype(doc_ids.dtype, np.integer):
        raise RuntimeError(f'doc-id sidecar dtype {doc_ids.dtype} is not integer')
    if not np.issubdtype(doc_lengths.dtype, np.integer):
        raise RuntimeError(f'length sidecar dtype {doc_lengths.dtype} is not integer')
    if (doc_lengths <= 0).any():
        raise RuntimeError('non-positive document lengths in the sidecar')
    if len(np.unique(doc_ids)) != doc_ids.shape[0]:
        raise RuntimeError('duplicate document ids in the sidecar')
    read_files = {p.name: _sha256_file(p) for p in (ids_path, lens_path)}
    return doc_ids, doc_lengths, read_files


def _overlap_with(other_files, panel_ids):
    """Intersect the panel with train/val doc-id sidecars when they exist.

    Returns (overlap_count, checked_files) — missing sidecars are NOT an
    error (their absence is disclosed by the caller); any real overlap is.
    """
    import numpy as np
    overlap = 0
    checked = []
    for path in sorted(other_files):
        try:
            other = np.load(path)
        except Exception:
            continue
        checked.append(path.name)
        inter = np.intersect1d(panel_ids, other)
        overlap += int(inter.size)
    return overlap, checked


def stratified_panel(doc_lengths, *, min_docs, n_strata, rng):
    """Length-stratified sample: returns per-row indices into the roster.

    Equal-count strata over the length-sorted roster; proportional
    allocation by largest remainder (each stratum keeps >= 1 document);
    without-replacement draws inside each stratum.  When the roster is not
    larger than the request, every document is taken (the full split).
    """
    n = int(doc_lengths.shape[0])
    if n_strata < 1:
        raise ValueError('n_strata must be >= 1')
    if n <= min_docs:
        # The complete split IS the panel.
        order = list(range(n))
        return order, [{'stratum': 0, 'roster': n, 'sampled': n,
                        'full_split': True}], True
    n_strata = min(int(n_strata), n)
    length_sorted = sorted(range(n), key=lambda i: int(doc_lengths[i]))
    # Equal-count strata over the length-sorted roster.
    bounds = [round(k * n / n_strata) for k in range(n_strata + 1)]
    strata = [length_sorted[bounds[k]:bounds[k + 1]]
              for k in range(n_strata)]
    strata = [s for s in strata if s]
    # Proportional allocation with largest-remainder rounding.
    raw = [len(s) * min_docs / n for s in strata]
    alloc = [max(1, int(round(r))) for r in raw]
    deficit = min_docs - sum(alloc)
    if deficit > 0:
        # Largest fractional remainder first.
        order_by_frac = sorted(range(len(strata)),
                               key=lambda k: raw[k] - int(raw[k]),
                               reverse=True)
        for k in order_by_frac[:deficit]:
            alloc[k] += 1
    elif deficit < 0:
        # Trim from the largest strata, never below one document.
        order_by_size = sorted(range(len(strata)), key=lambda k: -len(strata[k]))
        for k in order_by_size[:-deficit]:
            if alloc[k] > 1:
                alloc[k] -= 1
    picked = []
    strat_info = []
    for k, s in enumerate(strata):
        take = min(alloc[k], len(s))
        perm = rng.permutation(len(s))[:take]
        chosen = [s[i] for i in perm]
        picked.extend(chosen)
        lens_k = sorted(int(doc_lengths[i]) for i in s)
        strat_info.append({
            'stratum': k,
            'roster': len(s),
            'sampled': take,
            'length_min': lens_k[0],
            'length_median': lens_k[len(lens_k) // 2],
            'length_max': lens_k[-1],
        })
    picked = sorted(picked)
    if len(picked) < min_docs:
        raise RuntimeError(f'internal allocation error: picked {len(picked)} '
                           f'< requested {min_docs}')
    return picked, strat_info, False


def freeze_panel(study_id: str, sidecar_dir: Path, *, min_docs: int,
                 n_strata: int, seed: int, synthetic: bool,
                 repo: Path | None = None) -> dict:
    import numpy as np
    doc_ids, doc_lengths, read_files = _load_sidecars(sidecar_dir)
    rng = np.random.default_rng(int(seed))
    picked, strat_info, full_split = stratified_panel(
        doc_lengths, min_docs=int(min_docs), n_strata=int(n_strata), rng=rng)
    panel_ids = np.asarray(doc_ids)[picked]
    panel_lengths = np.asarray(doc_lengths)[picked]
    # Cross-check against train/val sidecars when they exist: an overlapping
    # document can never enter the frozen evaluation panel.
    train_side = sorted(sidecar_dir.glob('train_doc_ids_*.npy'))
    val_side = sorted(sidecar_dir.glob('val_doc_ids_*.npy'))
    overlap, checked = _overlap_with(train_side + val_side, panel_ids)
    if overlap:
        raise RuntimeError(f'{overlap} panel document ids also appear in the '
                           f'train/validation sidecars ({checked}) — leaked '
                           f'documents must never enter the frozen panel')
    canonical = hashlib.sha256(
        np.ascontiguousarray(panel_ids.astype(np.uint64))).hexdigest()
    artifact = {
        'schema_version': SCHEMA_VERSION,
        'tool': TOOL_NAME,
        'status': 'ok',
        'study_id': str(study_id),
        'generated_at_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'synthetic': bool(synthetic),
        'sidecar_dir': str(sidecar_dir),
        'read_files': read_files,
        'access_scope': 'document ids and document lengths only — the '
                        'held-out token stream (test_ids_*.npy) is NOT read '
                        'by this tool; the content stays unread until the '
                        'frozen study evaluates it',
        'rng_seed': int(seed),
        'n_strata': int(n_strata),
        'requested_min_docs': int(min_docs),
        'full_split_taken': bool(full_split),
        'split_documents': int(doc_ids.shape[0]),
        'panel_documents': int(panel_ids.shape[0]),
        'panel_sha256_uint64_le': canonical,
        'doc_ids': [int(x) for x in panel_ids],
        'doc_lengths': [int(x) for x in panel_lengths],
        'strata': strat_info,
        'overlap_check': {
            'train_val_sidecars_checked': checked,
            'overlapping_documents': overlap,
        },
        'notes': [
            'A frozen panel is a pre-registration act, not an evaluation: '
            'no model output exists here and none is implied.',
            'Stratification is by document length over the length-sorted '
            'roster with proportional allocation (every stratum keeps at '
            'least one document); the within-stratum draw is without '
            'replacement under the recorded seed.',
            'The 64-document floor is a population gate, not a sufficiency '
            'claim: the study still needs its own document-level bootstrap '
            'and power analysis for the effects it tests.',
        ],
    }
    if synthetic:
        artifact['notes'].insert(0, 'SYNTHETIC sidecars: this artifact was '
                                    'produced from generated sidecars (tests '
                                    'only); it can never certify a real '
                                    'frozen panel.')
    return artifact


def _write_synthetic_sidecars(out_dir: Path, n_docs: int = 96, seed: int = 5):
    """Deterministic synthetic sidecars so tests exercise the real sampler."""
    import numpy as np
    rng = np.random.default_rng(seed)
    # Lengths: a spread that actually stratifies (log-normal-ish, int).
    lengths = rng.integers(50, 5000, size=n_docs).astype(np.int64)
    doc_ids = (rng.integers(0, 2 ** 63 - 1, size=n_docs).astype(np.uint64)
               + np.uint64(1))
    np.save(out_dir / 'test_doc_ids_synthetic.npy', doc_ids)
    np.save(out_dir / 'test_doc_lengths_synthetic.npy', lengths)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study-id', default=None,
                        help='study id the panel is frozen for (binds the '
                             'artifact to the design record)')
    parser.add_argument('--design', type=Path, default=None,
                        help='optional design JSON to read the study id from')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent,
                        help='repository root (used to resolve the default '
                             'cache directory the same way exp_lib does)')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='wt103_cache directory holding the test sidecars '
                             '(default: the repo cache discipline)')
    parser.add_argument('--sidecar-dir', type=Path, default=None,
                        help='directory holding ONLY the sidecar pair to freeze '
                             'from (overrides --cache-dir)')
    parser.add_argument('--min-docs', type=int, default=DEFAULT_MIN_DOCS,
                        help=f'minimum distinct documents (default {DEFAULT_MIN_DOCS})')
    parser.add_argument('--strata', type=int, default=DEFAULT_STRATA,
                        help=f'number of length strata (default {DEFAULT_STRATA})')
    parser.add_argument('--seed', type=int, default=20261127,
                        help='rng seed recorded in the artifact')
    parser.add_argument('--out', type=Path,
                        default=Path('analysis_followup/doc_panel.json'),
                        help='output artifact path (parent dirs are created)')
    parser.add_argument('--synthetic', action='store_true',
                        help='freeze from generated synthetic sidecars '
                             '(tests only; the artifact is marked synthetic '
                             'and can never certify a real panel)')
    args = parser.parse_args()
    study_id = args.study_id
    if study_id is None and args.design is not None:
        try:
            study_id = json.loads(args.design.read_text(encoding='utf-8')).get('study_id')
        except Exception:
            study_id = None
    if not study_id or not str(study_id).strip():
        print('INVALID: --study-id (or a readable --design with a study_id) '
              'is required — an unbound panel certifies nothing', file=sys.stderr)
        return 2
    if args.min_docs < 1:
        print('INVALID: --min-docs must be >= 1', file=sys.stderr)
        return 2
    if args.strata < 1:
        print('INVALID: --strata must be >= 1', file=sys.stderr)
        return 2
    if args.synthetic:
        sidecar_dir = args.out.parent / '_synthetic_sidecars'
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        _write_synthetic_sidecars(sidecar_dir)
    elif args.sidecar_dir is not None:
        sidecar_dir = args.sidecar_dir
    elif args.cache_dir is not None:
        sidecar_dir = args.cache_dir
    else:
        # The repo's cache discipline: wt103_cache next to the repo (WT103_CACHE_DIR
        # handled by exp_lib; here we only need the sidecar directory).
        sidecar_dir = Path(args.repo) / 'wt103_cache'
    try:
        artifact = freeze_panel(str(study_id), sidecar_dir,
                                min_docs=int(args.min_docs),
                                n_strata=int(args.strata), seed=int(args.seed),
                                synthetic=bool(args.synthetic), repo=args.repo)
    except Exception as exc:
        print(f'PANEL NOT FROZEN: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1) + '\n', encoding='utf-8')
    print(f'[doc-panel] study={artifact["study_id"]} '
          f'split_docs={artifact["split_documents"]} '
          f'panel_docs={artifact["panel_documents"]} '
          f'strata={artifact["n_strata"]} full_split={artifact["full_split_taken"]} '
          f'synthetic={artifact["synthetic"]}')
    print(f'[doc-panel] panel sha256(uint64le)={artifact["panel_sha256_uint64_le"]}')
    for s in artifact['strata']:
        print(f'[doc-panel]   stratum {s["stratum"]}: roster={s["roster"]} '
              f'sampled={s["sampled"]} len[{s["length_min"]}..{s["length_median"]}..{s["length_max"]}]')
    print(f'[doc-panel] artifact: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
