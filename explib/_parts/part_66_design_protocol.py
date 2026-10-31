"""Prospective design protocol: one pre-run registration for the whole study.

The repository already has strong *pieces* of experimental design: a frozen
manifest with arms/seeds/alpha, a power projection, a randomized run order, a
run-disposition policy for failed cells, a held-out access budget and several
result-level audits.  Those pieces live in different files and are checked by
different commands.  Nothing requires them to describe the **same** experiment
before the first cell starts, and nothing detects a design that was assembled
after the fact from pieces that individually look valid.

This fragment adds the missing design-stage object: a canonical
``design_protocol`` with

* one section per design decision (question/endpoints, arms/comparisons,
  sample size/power, randomization/allocation, stopping, analysis, data and
  operations);
* a content hash that excludes only timestamps and revision metadata, so the
  scientific content is immutable once registered;
* a completeness/consistency check that is strict for confirmatory studies and
  explicitly weaker for exploratory/draft studies;
* a timing check against the first recorded event, because a protocol written
  after the first cell is a reconstruction, not a pre-registration.

It deliberately does **not** add a test, p-value, Holm slot or run.  It reads
JSON only and changes no measurement, seed, endpoint or frozen decision.  A
missing protocol is reported as ``not_verifiable_locally``, never as a pass.
"""

import copy
import datetime as _dp_dt
import hashlib
import json
import math
import numbers
import os
import time


DESIGN_PROTOCOL_KIND = 'design_protocol'
DESIGN_PROTOCOL_SCHEMA = 1
DESIGN_PROTOCOL_SECTIONS = (
    'study', 'research_question', 'arms', 'comparisons', 'sample_size',
    'randomization', 'stopping', 'analysis', 'data', 'operations',
)

DESIGN_PROTOCOL_POLICY = {
    'version': 1,
    'strict_levels': ('confirmatory',),
    'levels': ('confirmatory', 'exploratory', 'draft'),
    'required_sections': DESIGN_PROTOCOL_SECTIONS,
    'role': (
        'one prospective, hash-bound registration of the design decisions '
        'that the result-level gates assume were made before the run.  It is '
        'a completeness/consistency/timing gate, not a new statistical test '
        'and not an independent replication.'),
    'rule': (
        'confirmatory: every section is required, the resolution floor must '
        'be attainable at the declared min_n/family, the correction and '
        'disposition policy must be pre-declared, and the protocol must '
        'precede the first recorded event.  exploratory/draft: missing '
        'pieces are warnings and the study may not be promoted without them.'),
    'note': (
        'absence of a protocol is not evidence that a study was not '
        'pre-registered; it means this tree cannot verify that it was.  The '
        'protocol never proves that the analyst did not look at the data.'),
}

# A declaration is not a pre-registration unless the executable analysis is
# part of it.  The manifest can stay byte-identical while the code that turns
# a stored by-length cell into a p-value, a Holm decision or a direction
# verdict changes.  The freeze lock binds the source only at the first test
# read, so an edit made between registration and that read would be invisible.
# The protocol therefore carries a digest of the analysis closure:
#
#   * the core measurement library (``explib/_parts``);
#   * the confirmation driver that computes the frozen endpoint
#     (``explib/confirmation``);
#   * the loader and the historical import shim that assemble those fragments.
#
# The whole source-tree digest is also recorded for provenance, but only this
# closure is a hard invalidation: editing an unrelated panel driver or the
# dynamic draft must not silently invalidate a fixed-chunk registration.
DESIGN_PROTOCOL_ANALYSIS_PREFIXES = (
    'explib/_parts/',
    'explib/confirmation/',
)
DESIGN_PROTOCOL_ANALYSIS_FILES = (
    'explib/__init__.py',
    '_split_loader.py',
    'exp_lib.py',
)
DESIGN_PROTOCOL_IMPLEMENTATION_POLICY = {
    'version': 1,
    'analysis_prefixes': DESIGN_PROTOCOL_ANALYSIS_PREFIXES,
    'analysis_files': DESIGN_PROTOCOL_ANALYSIS_FILES,
    'rule': (
        'confirmatory: the protocol must bind a digest of the analysis '
        'closure and the code-semantics stamp; a changed analysis closure is '
        'a new registration, not an edit.  exploratory/draft: a missing '
        'binding is a warning because it cannot support a confirmatory claim.'),
    'whole_tree_rule': (
        'the whole measurement source-tree digest is recorded as provenance; '
        'drift outside the analysis closure is a warning, because a different '
        'panel driver or draft manifest does not implement this endpoint.'),
    'boundary': (
        'a digest makes code drift visible; it is not a signature and cannot '
        'stop someone who rewrites both the code and the registration.  The '
        'registered analysis is still the one that must be reported.'),
}


def _dp_sha_text(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def _dp_canonical(obj):
    """Canonical JSON used for every design hash (sorted, compact, strict)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _dp_sha_obj(obj):
    return _dp_sha_text(_dp_canonical(obj))


def _dp_sha256_text(value):
    """Return a lower-case SHA-256 hex digest, or ``None`` when malformed."""
    text = _dp_text(value)
    if text is None:
        return None
    text = text.lower()
    if len(text) != 64 or any(ch not in '0123456789abcdef' for ch in text):
        return None
    return text


def _dp_analysis_relpaths(files):
    """Select the analysis closure from a source-tree file map.

    The selection is deliberately a whitelist, not "whatever changed": an
    unrelated exploratory driver or the dynamic draft may be edited while a
    fixed-chunk study is pending, and that must not look like a change to the
    endpoint that computes the frozen decision.
    """
    if not isinstance(files, dict):
        return []
    out = []
    for rel in files:
        if not isinstance(rel, str):
            continue
        if rel in DESIGN_PROTOCOL_ANALYSIS_FILES:
            out.append(rel)
        elif any(rel.startswith(p) for p in DESIGN_PROTOCOL_ANALYSIS_PREFIXES):
            out.append(rel)
    return sorted(set(out))


def _dp_subset_source_sha(files, relpaths):
    """Recompute the source-tree digest over a declared subset of files.

    Uses the same ``relpath NUL bytes NUL sha256 LF`` construction as
    ``source_tree_manifest`` so the subset digest has exactly the meaning of
    the full-tree digest, just over a narrower, declared scope.
    """
    if not isinstance(files, dict) or not relpaths:
        return None
    rows = []
    for rel in sorted(set(relpaths)):
        entry = files.get(rel)
        if not isinstance(entry, dict):
            return None
        digest = _dp_sha256_text(entry.get('sha256'))
        size = _dp_int(entry.get('bytes'), minimum=0)
        if digest is None or size is None:
            return None
        rows.append(f'{rel}\0{size}\0{digest}\n')
    return _dp_sha_text(''.join(rows))


def _dp_implementation_binding(source_tree=None):
    """Copy the executable-analysis identity into the design protocol.

    ``source_tree`` is the output of ``source_tree_manifest``.  Injecting it
    makes the timing/conformance tests deterministic;
    production callers omit it and the current tree is hashed.  Missing or
    partial input returns ``None``: the validator then reports a missing
    binding instead of the builder inventing a convenient digest.
    """
    if source_tree is None:
        manifest = globals().get('source_tree_manifest')
        if not callable(manifest):
            return None
        try:
            source_tree = manifest()
        except Exception:
            return None
    if not isinstance(source_tree, dict):
        return None
    files = source_tree.get('files')
    relpaths = _dp_analysis_relpaths(files)
    analysis_sha = _dp_subset_source_sha(files, relpaths)
    if analysis_sha is None:
        return None
    registry = source_tree.get('split_registry')
    registry = registry if isinstance(registry, dict) else None
    unreadable = source_tree.get('unreadable')
    return {
        'code_semantics': _dp_text(CODE_SEMANTICS),  # noqa: F821
        'analysis_source_algorithm': _dp_text(source_tree.get('algorithm')),
        'analysis_source_sha256': analysis_sha,
        'analysis_n_files': len(relpaths),
        'analysis_prefixes': list(DESIGN_PROTOCOL_ANALYSIS_PREFIXES),
        'analysis_files': list(DESIGN_PROTOCOL_ANALYSIS_FILES),
        'full_source_tree_sha256': _dp_sha256_text(source_tree.get('sha256')),
        'full_source_tree_n_files': _dp_int(source_tree.get('n_files'),
                                           minimum=0),
        'split_registry_sha256': (_dp_sha256_text(registry.get('sha256'))
                                  if registry else None),
        'split_registry_status': (_dp_text(registry.get('status'))
                                  if registry else None),
        'split_registry_n_parts': (_dp_int(registry.get('n_parts'),
                                          minimum=0) if registry else None),
        'source_unreadable': list(unreadable) if isinstance(
            unreadable, list) else [],
        'scope': ('core measurement library + confirmation endpoint driver + '
                  'loader/import shim; the whole-tree digest is provenance, '
                  'not part of the hard invalidation rule'),
        'policy': dict(DESIGN_PROTOCOL_IMPLEMENTATION_POLICY),
    }


def _dp_implementation_audit(protocol, *, source_tree=None):
    """Compare a registered analysis closure with the current tree.

    Returns a compact block with both digests, the exact problems and the
    whole-tree warnings kept separate.  A missing registration is a hard
    problem for a confirmatory protocol and a warning otherwise; a missing
    current tree is ``not_verifiable`` rather than a pass.
    """
    out = {'kind': 'design_protocol_implementation',
           'status': 'not_verifiable',
           'registered': None, 'current': None,
           'problems': [], 'warnings': []}
    analysis = protocol.get('analysis') if isinstance(
        protocol.get('analysis'), dict) else {}
    registered = analysis.get('implementation')
    if not isinstance(registered, dict):
        out['status'] = 'missing'
        out['warnings'].append(
            'analysis.implementation is not registered; the protocol binds '
            'the design declarations but not the code that computes them')
        return out
    out['registered'] = {
        'code_semantics': registered.get('code_semantics'),
        'analysis_source_sha256': registered.get('analysis_source_sha256'),
        'analysis_n_files': registered.get('analysis_n_files'),
        'full_source_tree_sha256': registered.get('full_source_tree_sha256'),
        'split_registry_sha256': registered.get('split_registry_sha256'),
    }
    if _dp_sha256_text(registered.get('analysis_source_sha256')) is None:
        out['problems'].append(
            'analysis.implementation.analysis_source_sha256 is missing or not '
            'a 64-character SHA-256 digest')
    if _dp_sha256_text(registered.get('full_source_tree_sha256')) is None:
        out['problems'].append(
            'analysis.implementation.full_source_tree_sha256 is missing or '
            'not a 64-character SHA-256 digest')
    if _dp_int(registered.get('analysis_n_files'), minimum=1) is None:
        out['problems'].append(
            'analysis.implementation.analysis_n_files must be an integer >= 1')
    code_semantics = _dp_text(analysis.get('code_semantics'))
    if (code_semantics is not None
            and _dp_text(registered.get('code_semantics')) != code_semantics):
        out['problems'].append(
            'analysis.implementation.code_semantics does not equal '
            'analysis.code_semantics')
    status = _dp_text(registered.get('split_registry_status'))
    if status is None:
        out['warnings'].append(
            'analysis.implementation does not record the split-registry '
            'status; executed-fragment order cannot be checked')
    elif status != 'ok':
        out['warnings'].append(
            f'analysis.implementation.split_registry_status={status!r}; '
            'the executed analysis fragments were not fully registered')
    unreadable = registered.get('source_unreadable')
    if isinstance(unreadable, list) and unreadable:
        out['problems'].append(
            f'analysis.implementation.source_unreadable is non-empty '
            f'({len(unreadable)} file(s)); the analysis closure is incomplete')
    if source_tree is None:
        manifest = globals().get('source_tree_manifest')
        if callable(manifest):
            try:
                source_tree = manifest()
            except Exception as exc:
                out['warnings'].append(
                    'the current analysis tree could not be hashed: '
                    f'{type(exc).__name__}: {exc}')
                source_tree = None
    if not isinstance(source_tree, dict):
        out['warnings'].append(
            'no current source-tree identity was supplied; implementation '
            'drift cannot be checked')
        out['status'] = 'fail' if out['problems'] else 'not_verifiable'
        return out
    current_files = source_tree.get('files')
    current_relpaths = _dp_analysis_relpaths(current_files)
    current_sha = _dp_subset_source_sha(current_files, current_relpaths)
    out['current'] = {
        'code_semantics': _dp_text(CODE_SEMANTICS),  # noqa: F821
        'analysis_source_sha256': current_sha,
        'analysis_n_files': len(current_relpaths),
        'full_source_tree_sha256': _dp_sha256_text(
            source_tree.get('sha256')),
        'split_registry_sha256': (
            _dp_sha256_text((source_tree.get('split_registry') or {}).get(
                'sha256'))
            if isinstance(source_tree.get('split_registry'), dict) else None),
    }
    if current_sha is None:
        out['problems'].append(
            'the current analysis closure could not be hashed (the source '
            'tree is partial or malformed)')
    elif current_sha != _dp_sha256_text(
            registered.get('analysis_source_sha256')):
        out['problems'].append(
            'analysis implementation changed after registration: the '
            'analysis-closure digest no longer matches.  A code change is a '
            'design change; register a revision with a reason instead of '
            'editing the implementation under the old protocol')
    # The digest is the hard identity, but the recorded closure size and
    # declared scope are part of the same registration.  A same-digest tree
    # with a different count means the block was edited after registration
    # (or constructed inconsistently); do not let that read as verified.
    _reg_n = _dp_int(registered.get('analysis_n_files'), minimum=1)
    if (_reg_n is not None and current_relpaths
            and _reg_n != len(current_relpaths)):
        out['problems'].append(
            f'analysis.implementation.analysis_n_files={_reg_n} != the '
            f'current closure size {len(current_relpaths)}')
    for _key, _want in (
            ('analysis_prefixes', list(DESIGN_PROTOCOL_ANALYSIS_PREFIXES)),
            ('analysis_files', list(DESIGN_PROTOCOL_ANALYSIS_FILES))):
        _got = registered.get(_key)
        if _got is None:
            continue
        if (not isinstance(_got, list)
                or [str(x) for x in _got] != _want):
            out['problems'].append(
                f'analysis.implementation.{_key} does not match the declared '
                'analysis-closure scope; the registration describes a '
                'different closure than the one being checked')
    registered_full = _dp_sha256_text(
        registered.get('full_source_tree_sha256'))
    current_full = _dp_sha256_text(source_tree.get('sha256'))
    if (registered_full is not None and current_full is not None
            and registered_full != current_full and not out['problems']):
        out['warnings'].append(
            'measurement source outside the analysis closure changed after '
            'registration; the endpoint implementation is unchanged, but '
            'review whether the change affects the study')
    registered_reg = _dp_sha256_text(registered.get('split_registry_sha256'))
    current_registry = source_tree.get('split_registry')
    current_reg = (_dp_sha256_text(current_registry.get('sha256'))
                   if isinstance(current_registry, dict) else None)
    if (registered_reg is not None and current_reg is not None
            and registered_reg != current_reg and not out['problems']):
        out['warnings'].append(
            'split-package registration changed after the protocol was '
            'written; no analysis-closure file changed, but the executed '
            'order of another panel is now different')
    if out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out


def _dp_is_number(value):
    return (not isinstance(value, (bool, str, bytes, bytearray))
            and isinstance(value, numbers.Real))


def _dp_number(value):
    if not _dp_is_number(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _dp_resolution_floor(family_size, n):
    """Exact family-wise sign-flip floor ``m * 2 / 2**n`` without overflow.

    ``2.0 ** n`` overflows for a few thousand seeds even though the true
    floor is simply zero at that resolution.  ``math.ldexp`` handles the
    exponent without building an enormous power and does not raise for the
    non-positive exponents used here; a family size too large for a float is
    reported as infinite, which still fails the alpha comparison.
    """
    try:
        m = float(family_size)
        n = int(n)
    except (TypeError, ValueError, OverflowError):
        return float('inf')
    # For a single seed m * 2**(1-n) underflows quickly, but a very large
    # family can keep the product representable well past n=1075 (e.g.
    # m=1e308 at n=1076 is still ~1e-15).  Let ldexp do the scaling and only
    # fall back to zero when the exponent itself is beyond a C int; a huge
    # family that overflows float is still reported as infinite.
    try:
        return math.ldexp(m, 1 - n)
    except OverflowError:
        return 0.0 if math.isfinite(m) else float('inf')


def _dp_int(value, *, minimum=None, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool,)) or not isinstance(value, numbers.Integral):
        return None
    out = int(value)
    if minimum is not None and out < minimum:
        return None
    return out


def _dp_text(value):
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _dp_bool(value):
    return value if isinstance(value, bool) else None


def _dp_sign(value):
    """Map the manifest's vocabulary to -1/0/1, or None when not directional."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value) if int(value) in (-1, 0, 1) else None
    text = str(value).strip().lower()
    if text in ('negative', 'lower', 'decrease', 'negative_delta', '<'):
        return -1
    if text in ('positive', 'higher', 'increase', 'positive_delta', '>'):
        return 1
    if text in ('zero', 'none', 'non_directional', 'non-directional', 'two-sided'):
        return 0
    return None


def _dp_non_directional(value):
    """True only for an explicit no-direction declaration, not a typo."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return int(value) == 0
    text = str(value).strip().lower()
    return text in ('zero', 'none', 'non_directional', 'non-directional',
                    'two-sided', 'two sided')


def _dp_normal_correction(value):
    text = _dp_text(value)
    if text is None:
        return None
    key = text.lower().replace('_', ' ').replace('-', ' ')
    key = ' '.join(key.split())
    if key in ('holm', 'holm step down', 'holm stepdown'):
        return 'holm'
    if key in ('max t', 'maxt', 'westfall young', 'westfall young step down'):
        return 'max-t'
    if key in ('bonferroni',):
        return 'bonferroni'
    return key


def _dp_timestamp(value):
    """Parse the ISO-like wall-clock stamps used by the drivers."""
    text = _dp_text(value)
    if text is None:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return _dp_dt.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
            try:
                return _dp_dt.datetime.strptime(text, fmt)
            except (TypeError, ValueError):
                continue
    return None


def _dp_comparable_times(a, b):
    return (a is not None and b is not None
            and (a.tzinfo is None) == (b.tzinfo is None))


def design_protocol_content_sha(protocol):
    """Hash the scientific body, excluding only provenance/revision fields.

    ``created_at`` says when the registration was written, not what it says;
    ``revision``/``supersedes`` describe the history.  Everything else -- the
    hypotheses, endpoints, family, sample size, randomization, stopping rule,
    analysis rule, data split and disposition policy -- is covered.  A rewrite
    that changes any of those therefore changes the identity.
    """
    if not isinstance(protocol, dict):
        raise TypeError('design protocol must be an object')
    excluded = {'content_sha256', 'created_at', 'revision', 'supersedes',
                'reason', 'history'}
    body = {str(k): v for k, v in protocol.items() if str(k) not in excluded}
    return _dp_sha_obj(body)


def _dp_commitments(protocol):
    return {section: _dp_sha_obj(protocol.get(section))
            for section in DESIGN_PROTOCOL_SECTIONS}


def _dp_level(manifest, explicit=None):
    if explicit is not None:
        level = _dp_text(explicit)
        if level in DESIGN_PROTOCOL_POLICY['levels']:
            return level
        raise ValueError(f'unknown design level {explicit!r}')
    status = _dp_text((manifest or {}).get('status'))
    if status and status.lower() == 'draft':
        return 'draft'
    if isinstance((manifest or {}).get('scope'), dict):
        if (manifest.get('scope') or {}).get('confirmatory_claims'):
            return 'confirmatory'
    return 'exploratory'


def _dp_manifest_sha(manifest):
    return _dp_sha_obj(manifest)


def _dp_recipe(manifest):
    recipe = (manifest or {}).get('training_recipe')
    if isinstance(recipe, dict):
        return recipe
    binding = (manifest or {}).get('binding')
    if isinstance(binding, dict):
        sig = binding.get('training_recipe_signature')
        if isinstance(sig, dict):
            return sig
    sig = (manifest or {}).get('training_recipe_signature')
    return sig if isinstance(sig, dict) else {}


def _dp_normalise_test_access(manifest, test_access=None):
    """Extract and validate the held-out access budget.

    The shared ``test_access_policy`` validator is the single source of truth
    for the policy schema (unknown keys, method/max_looks agreement, boolean
    flags).  Reusing it here means a typo such as ``max_look`` cannot enter a
    protocol as a valid-looking declaration, and a malformed declaration is
    returned as ``invalid`` instead of escaping as an exception from
    ``build_design_protocol``.  The ``declared``/``role`` keys are protocol
    provenance, not policy fields, and are stripped before validation.
    """
    explicit = test_access is not None
    try:
        if explicit:
            if not isinstance(test_access, dict):
                return {'invalid': True,
                        'error': f'test_access must be an object, got {type(test_access).__name__}'}
            raw = test_access
            candidate = {k: v for k, v in raw.items()
                         if k not in ('declared', 'role')}
            policy = test_access_policy({'test_access_policy': candidate})  # noqa: F821
            declared = raw.get('declared', True) is True
        else:
            raw = None
            stat = (manifest or {}).get('statistics')
            for _source, block in (('statistics.test_access_policy', stat),
                                   ('test_access_policy', manifest)):
                if isinstance(block, dict) and 'test_access_policy' in block:
                    value = block.get('test_access_policy')
                    if not isinstance(value, dict):
                        return {'invalid': True,
                                'error': f'{_source} must be an object, got {type(value).__name__}'}
                    raw = value
                    break
            policy = test_access_policy(manifest)  # noqa: F821 - same module
            declared = policy.get('declared') is True
    except Exception as exc:
        return {'invalid': True, 'error': f'{type(exc).__name__}: {exc}'}
    raw = raw if isinstance(raw, dict) else {}
    return {
        'method': _dp_text(policy.get('method')) or 'single_look',
        'max_looks': _dp_int(policy.get('max_looks'), minimum=1),
        'declared': declared,
        'require_reason_for_reread':
            policy.get('require_reason_for_reread') is not False,
        'source': _dp_text(raw.get('source')),
    }


def _dp_normalise_disposition(manifest, disposition=None):
    raw = disposition
    if raw is None:
        stat = (manifest or {}).get('statistics')
        if isinstance(stat, dict) and isinstance(stat.get('disposition_policy'),
                                                   dict):
            raw = stat.get('disposition_policy')
        elif isinstance((manifest or {}).get('disposition_policy'), dict):
            raw = (manifest or {}).get('disposition_policy')
    if raw is None:
        return None
    if isinstance(raw, dict) and raw.get('kind') == 'run_disposition':
        raw = raw.get('policy')
    if not isinstance(raw, dict):
        return {'invalid': True, 'raw_type': type(raw).__name__}
    try:
        pol = validate_run_disposition_policy(raw)  # noqa: F821
    except Exception as exc:
        return {'invalid': True, 'error': f'{type(exc).__name__}: {exc}'}
    return {
        'retention': pol.get('retention'),
        'replacement': pol.get('replacement'),
        'max_attempts_per_cell': pol.get('max_attempts_per_cell'),
        'max_total_attempts': pol.get('max_total_attempts'),
        'allow_supersede': pol.get('allow_supersede'),
        'replaceable_failure_classes':
            list(pol.get('replaceable_failure_classes') or []),
        'non_replaceable_failure_classes':
            list(pol.get('non_replaceable_failure_classes') or []),
    }


def _dp_comparison(c):
    if not isinstance(c, dict):
        return None
    a = _dp_text(c.get('a'))
    b = _dp_text(c.get('b'))
    name = _dp_text(c.get('name')) or (
        f'{a} - {b}' if a and b else None)
    if name is None:
        return None
    raw_sign = c.get('expected_sign')
    sign = _dp_sign(raw_sign)
    # An explicit JSON null (as used for the sink-matched readout) is a
    # declaration of no direction.  An absent key is not: treating it as
    # non-directional would let an omitted expected_sign pass as if the
    # direction had been pre-registered.
    non_directional = (bool(_dp_non_directional(raw_sign))
                       if 'expected_sign' in c else False)
    return {
        'name': name,
        'a': a,
        'b': b,
        'endpoint': _dp_text(c.get('endpoint')),
        'predicted_sign': sign,
        'non_directional': non_directional,
        'mechanism_claim_allowed': c.get('mechanism_claim_allowed') is True,
        'hypothesis_direction': _dp_text(c.get('hypothesis_direction')),
        'caveat': _dp_text(c.get('interpretation_caveat')),
        'claim_note': _dp_text(c.get('claim_note')),
    }


def _dp_hypotheses(manifest, comparisons):
    raw = (manifest or {}).get('hypotheses')
    out = []
    if isinstance(raw, dict) and raw:
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                continue
            raw_sign = spec.get('predicted_sign')
            sign = _dp_sign(raw_sign)
            non_directional = (bool(_dp_non_directional(raw_sign))
                               if 'predicted_sign' in spec else False)
            out.append({
                'id': str(name),
                'statement': _dp_text(spec.get('statement')),
                'endpoint': _dp_text(spec.get('endpoint')),
                'predicted_sign': sign,
                'non_directional': non_directional,
                'falsifier': _dp_text(spec.get('falsifier')),
                'mechanism': _dp_text(spec.get('mechanism')),
                'requires_param_match': spec.get('requires_param_match') is True,
                'source': 'manifest.hypotheses',
            })
        return out
    # The fixed-chunk freeze expresses its questions as named primary
    # comparisons rather than a separate hypotheses block.  Derive the
    # question objects from those declarations and say so; this is a
    # transcription, not a new hypothesis.
    for c in comparisons:
        out.append({
            'id': c['name'],
            'statement': (c.get('hypothesis_direction')
                          or f'{c.get("a")} vs {c.get("b")}'),
            'endpoint': c.get('endpoint'),
            'predicted_sign': c.get('predicted_sign'),
            'non_directional': c.get('non_directional'),
            'falsifier': (c.get('claim_note') or c.get('caveat')
                          or 'the pre-registered direction is not observed '
                             'after the declared correction'),
            'mechanism': None,
            'requires_param_match': False,
            'source': 'primary_comparisons',
        })
    return out


def build_design_protocol(
        manifest, *, level=None, study_name=None, run_order=None,
        disposition=None, test_access=None, resource_budget=None,
        disjoint_declared=None, extras=None, created_at=None, revision=0,
        supersedes=None, reason=None, source_tree=None):
    """Compile one canonical protocol from the existing declarations.

    This does not create new scientific commitments: every value is copied
    from the manifest, the run-order plan, the disposition policy or the
    held-out access policy.  Missing inputs stay missing; the validation gate
    then reports them instead of filling in a convenient default.

    ``source_tree`` is the ``source_tree_manifest`` of the measurement tree.
    It is captured so the protocol binds the analysis closure that will
    compute the endpoint, not only the prose around it.  Production callers
    omit it and the current checkout is hashed; tests inject a fixture.
    """
    if not isinstance(manifest, dict):
        raise TypeError('manifest must be an object')
    level = _dp_level(manifest, level)
    implementation = _dp_implementation_binding(source_tree)
    stats = manifest.get('statistics')
    stats = stats if isinstance(stats, dict) else {}
    comps = []
    for raw in (manifest.get('primary_comparisons') or []):
        c = _dp_comparison(raw)
        if c is not None:
            comps.append(c)
    arms_raw = manifest.get('arms')
    arms = []
    if isinstance(arms_raw, (list, tuple)):
        for a in arms_raw:
            text = _dp_text(a)
            if text is not None and text not in arms:
                arms.append(text)
    seeds = manifest.get('seeds')
    # Do not infer the planned n from the length of the seed list: the
    # protocol records declarations, and a missing ``n_seeds`` is a missing
    # design decision that the validator must report rather than fill in.
    n_seeds = _dp_int(stats.get('n_seeds'), minimum=1)
    min_n = _dp_int(stats.get('min_seeds_for_confirmation'), minimum=1)
    alpha = _dp_number(stats.get('alpha', manifest.get('alpha')))
    family_size = len(comps) or None

    recipe = _dp_recipe(manifest)
    steps = _dp_int(recipe.get('steps'), minimum=1)
    batch = _dp_int(recipe.get('bs'), minimum=1)
    seq_len = _dp_int(recipe.get('sl'), minimum=1)
    planned_tokens = (steps * batch * seq_len
                      if steps and batch and seq_len else None)
    eval_every = _dp_int(recipe.get('ee'), minimum=1)
    patience = _dp_int(recipe.get('early_stop_patience'), minimum=1,
                       allow_none=True)
    eval_subset = _dp_int(recipe.get('es'), minimum=1, allow_none=True)

    order = run_order if isinstance(run_order, dict) else None
    design = order.get('design') if isinstance(order, dict) else None
    design = design if isinstance(design, dict) else {}
    randomization = None
    if isinstance(order, dict):
        randomization = {
            'design': _dp_text(design.get('name')) or 'randomized_complete_block_balanced',
            'design_seed': _dp_int(design.get('design_seed'), minimum=0),
            'block': _dp_text(design.get('block')) or 'seed',
            'position_balance': bool(design.get('position_balance')),
            'planned_cells': list(order.get('planned_cells') or []),
            'environment_slots': list(order.get('environment_slots') or []) or None,
            'source': 'run_order.json',
        }

    test_policy = _dp_normalise_test_access(manifest, test_access)
    dispo = _dp_normalise_disposition(manifest, disposition)

    eval_cfg = manifest.get('eval')
    eval_cfg = eval_cfg if isinstance(eval_cfg, dict) else {}
    binding = manifest.get('binding')
    binding = binding if isinstance(binding, dict) else {}

    _purpose = _dp_text(manifest.get('purpose'))
    _purpose_name = None
    if _purpose:
        _purpose_name = _purpose.split('.')[0].strip()[:160] or None
    protocol = {
        'kind': DESIGN_PROTOCOL_KIND,
        'schema': DESIGN_PROTOCOL_SCHEMA,
        'study': {
            'name': (_dp_text(study_name) or _dp_text(manifest.get('name'))
                     or _purpose_name or 'unnamed-study'),
            'level': level,
            'manifest_sha256': _dp_manifest_sha(manifest),
            'manifest_version': manifest.get('manifest_version'),
            'declared_status': manifest.get('status'),
            'frozen_at': manifest.get('frozen_at'),
            'code_semantics': _dp_text(manifest.get('code_semantics_expected'))
                              or _dp_text(CODE_SEMANTICS),  # noqa: F821
            'alpha': alpha,
        },
        'research_question': {
            'hypotheses': _dp_hypotheses(manifest, comps),
            'primary_family': [c['name'] for c in comps],
            'scope': copy.deepcopy(manifest.get('scope')) if isinstance(
                manifest.get('scope'), dict) else None,
        },
        'arms': arms,
        'comparisons': comps,
        'sample_size': {
            'planned_seeds': n_seeds,
            'seeds': list(seeds) if isinstance(seeds, (list, tuple)) else None,
            'min_n': min_n,
            'family_size': family_size,
            'n_eval_units': _dp_int(eval_cfg.get('n_seqs_clean'), minimum=1),
            'target_power': _dp_number(stats.get('min_power_for_test')),
            'power_q': _dp_number(stats.get('power_q_for_test')),
            'power_method': _dp_text(stats.get('power_note')),
            'power_simulated': bool(stats.get('power_simulation_required')),
            'power_gate_required': bool(
                stats.get('require_power_gate')
                or stats.get('power_simulation_required')),
            'power_simulation_required': bool(
                stats.get('power_simulation_required')),
            'resolution_floor': _dp_number(stats.get('min_attainable_p_holm')),
            'min_consistency': _dp_int(
                stats.get('min_consistency_for_first_gate'), minimum=1,
                allow_none=True),
            'source': 'manifest.statistics',
        },
        'randomization': randomization,
        'stopping': {
            'training_steps': steps,
            'tokens_per_step': (batch * seq_len
                                if batch and seq_len else None),
            'planned_tokens_seen': planned_tokens,
            'eval_every': eval_every,
            'eval_subset': eval_subset,
            'early_stopping': ({'patience': patience,
                                'min_delta': _dp_number(
                                    recipe.get('early_stop_min_delta', 0.0))}
                               if patience else None),
            'test_looks': test_policy.get('max_looks'),
            'analysis_looks': 1,
            'test_access_method': test_policy.get('method'),
            'test_access_declared': test_policy.get('declared'),
            'allow_data_dependent_stopping': False,
            'rule': 'fixed_plan_no_optional_stopping',
            'source': 'manifest.training_recipe + test_access_policy',
        },
        'analysis': {
            'test': _dp_text(stats.get('test')),
            'correction': _dp_normal_correction(stats.get('correction')),
            'alpha': alpha,
            'family': _dp_text(stats.get('family')),
            'family_size': family_size,
            'min_attainable_p_holm': _dp_number(
                stats.get('min_attainable_p_holm')),
            'code_semantics': _dp_text(manifest.get('code_semantics_expected'))
                              or _dp_text(CODE_SEMANTICS),  # noqa: F821
            'endpoint_rule': 'pre_registered_primary_then_descriptive_by_length',
            'missing_data_policy': 'report_as_is_no_topup_no_imputation',
            'decision_rule': _dp_text(stats.get('decision_rule')),
            'implementation': implementation,
            'source': 'manifest.statistics',
        },
        'data': {
            'selection_split': 'validation',
            'test_split': 'test',
            'disjoint_declared': (disjoint_declared
                                  if isinstance(disjoint_declared, bool)
                                  else None),
            'training_fingerprint': copy.deepcopy(
                binding.get('training_data_fingerprint')),
            'test_slice': copy.deepcopy(binding.get('test_slice')),
            'eval_lengths': list(eval_cfg.get('by_length_ppl') or []) or None,
            'n_eval_units': _dp_int(eval_cfg.get('n_seqs_clean'), minimum=1),
            'test_access': test_policy,
            'source': 'manifest.binding + manifest.eval',
        },
        'operations': {
            'disposition': dispo,
            'resource_budget': copy.deepcopy(resource_budget),
            'environment_plan': ('run_order.environment_slots'
                                 if randomization and randomization.get(
                                     'environment_slots') else None),
            'source': 'manifest.statistics.disposition_policy + caller',
        },
        'extras': copy.deepcopy(extras) if isinstance(extras, dict) else None,
    }
    protocol['commitments'] = _dp_commitments(protocol)
    protocol['created_at'] = created_at or time.strftime('%Y-%m-%dT%H:%M:%S')
    protocol['revision'] = _dp_int(revision, minimum=0) or 0
    protocol['supersedes'] = list(supersedes or [])
    protocol['reason'] = _dp_text(reason)
    protocol['content_sha256'] = design_protocol_content_sha(protocol)
    return protocol


def _dp_problem(out, msg):
    out['problems'].append(str(msg))


def _dp_warn(out, msg):
    out['warnings'].append(str(msg))


def validate_design_protocol(protocol, *, require_complete=None):
    """Check the protocol's structure, completeness and internal arithmetic.

    ``require_complete`` defaults to strict for a confirmatory study and
    permissive (warnings, not failures) for exploratory/draft studies.  The
    function never fills in a missing value: a missing design decision is
    reported as missing, which is the entire point of a pre-flight gate.
    """
    out = {
        'kind': 'design_protocol_validation',
        'status': 'ok',
        'level': None,
        'strict': None,
        'missing': [],
        'problems': [],
        'warnings': [],
        'checks': {},
    }
    if not isinstance(protocol, dict):
        _dp_problem(out, f'protocol must be an object, got {type(protocol).__name__}')
        out['status'] = 'fail'
        return out
    study = protocol.get('study')
    study = study if isinstance(study, dict) else {}
    level = _dp_text(study.get('level'))
    if level not in DESIGN_PROTOCOL_POLICY['levels']:
        _dp_problem(out, f"study.level must be one of {DESIGN_PROTOCOL_POLICY['levels']}, got {level!r}")
        level = None
    strict = (level in DESIGN_PROTOCOL_POLICY['strict_levels']
              if require_complete is None else bool(require_complete))
    out['level'] = level
    out['strict'] = bool(strict)

    def need(condition, message, *, missing=None):
        if condition:
            return True
        if strict:
            _dp_problem(out, message)
        else:
            _dp_warn(out, message)
            if missing:
                out['missing'].append(missing)
        return False

    if protocol.get('kind') != DESIGN_PROTOCOL_KIND:
        _dp_problem(out, f"protocol.kind must be {DESIGN_PROTOCOL_KIND!r}, got {protocol.get('kind')!r}")
    if protocol.get('schema') != DESIGN_PROTOCOL_SCHEMA:
        _dp_problem(out, f'protocol.schema must be {DESIGN_PROTOCOL_SCHEMA}, got {protocol.get("schema")!r}')

    name = _dp_text(study.get('name'))
    need(name is not None, 'study.name is missing', missing='study.name')
    sha = _dp_text(study.get('manifest_sha256'))
    if sha is None or len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha.lower()):
        _dp_problem(out, 'study.manifest_sha256 must be a 64-character SHA-256 hex digest')
    alpha = _dp_number(study.get('alpha'))
    if alpha is None or not (0.0 < alpha < 1.0):
        _dp_problem(out, f'study.alpha must be a finite probability in (0,1), got {study.get("alpha")!r}')
    need(_dp_text(study.get('code_semantics')) is not None,
         'study.code_semantics is missing', missing='study.code_semantics')

    # ---- research question -------------------------------------------------
    rq = protocol.get('research_question')
    rq = rq if isinstance(rq, dict) else {}
    hyps = rq.get('hypotheses')
    if not isinstance(hyps, list) or not hyps:
        need(False, 'research_question.hypotheses must be a non-empty list',
             missing='research_question.hypotheses')
        hyps = []
    hyp_ids = []
    for i, h in enumerate(hyps):
        tag = f'research_question.hypotheses[{i}]'
        if not isinstance(h, dict):
            _dp_problem(out, f'{tag} must be an object')
            continue
        hid = _dp_text(h.get('id'))
        if hid is None:
            _dp_problem(out, f'{tag}.id is missing')
        elif hid in hyp_ids:
            _dp_problem(out, f'{tag}.id {hid!r} is duplicated')
        else:
            hyp_ids.append(hid)
        need(_dp_text(h.get('statement')) is not None,
             f'{tag}.statement is missing', missing=f'{tag}.statement')
        need(_dp_text(h.get('endpoint')) is not None,
             f'{tag}.endpoint is missing', missing=f'{tag}.endpoint')
        sign = h.get('predicted_sign')
        nod = h.get('non_directional') is True
        if sign is None and not nod:
            need(False, f'{tag}.predicted_sign is missing and non_directional is not true',
                 missing=f'{tag}.predicted_sign')
        elif sign is not None and _dp_int(sign, allow_none=True) not in (-1, 0, 1):
            _dp_problem(out, f'{tag}.predicted_sign must be -1, 0, 1 or null')
        need(_dp_text(h.get('falsifier')) is not None,
             f'{tag}.falsifier is missing', missing=f'{tag}.falsifier')

    comps = protocol.get('comparisons')
    if not isinstance(comps, list) or not comps:
        need(False, 'comparisons must be a non-empty list', missing='comparisons')
        comps = []
    comp_names = []
    by_name = {}
    for i, c in enumerate(comps):
        tag = f'comparisons[{i}]'
        if not isinstance(c, dict):
            _dp_problem(out, f'{tag} must be an object')
            continue
        cname = _dp_text(c.get('name'))
        if cname is None:
            _dp_problem(out, f'{tag}.name is missing')
        elif cname in by_name:
            _dp_problem(out, f'{tag}.name {cname!r} is duplicated')
        else:
            comp_names.append(cname)
            by_name[cname] = c
        if _dp_text(c.get('a')) is None or _dp_text(c.get('b')) is None:
            _dp_problem(out, f'{tag} must name both arms (a/b)')
        need(_dp_text(c.get('endpoint')) is not None,
             f'{tag}.endpoint is missing', missing=f'{tag}.endpoint')
        sign = c.get('predicted_sign')
        nod = c.get('non_directional') is True
        if sign is None and not nod:
            need(False, f'{tag}.predicted_sign is missing and non_directional is not true',
                 missing=f'{tag}.predicted_sign')
        elif sign is not None and _dp_int(sign, allow_none=True) not in (-1, 0, 1):
            _dp_problem(out, f'{tag}.predicted_sign must be -1, 0, 1 or null')
        if not isinstance(c.get('mechanism_claim_allowed'), bool):
            _dp_problem(out, f'{tag}.mechanism_claim_allowed must be boolean')

    family = rq.get('primary_family')
    if not isinstance(family, list) or not family:
        need(False, 'research_question.primary_family must be a non-empty list',
             missing='research_question.primary_family')
        family = []
    else:
        family = [str(x) for x in family]
        if len(set(family)) != len(family):
            _dp_problem(out, 'research_question.primary_family contains duplicates')
        if set(family) != set(comp_names):
            _dp_problem(out, 'research_question.primary_family must name exactly the declared comparisons')
    out['checks']['n_hypotheses'] = len(hyps)
    out['checks']['n_comparisons'] = len(comps)

    # ---- arms --------------------------------------------------------------
    arms = protocol.get('arms')
    if not isinstance(arms, list) or not arms:
        need(False, 'arms must be a non-empty list', missing='arms')
        arms = []
    else:
        if not all(isinstance(x, str) for x in arms):
            _dp_problem(out, 'arms must contain only strings')
        arms = [str(x) for x in arms]
        if len(set(arms)) != len(arms):
            _dp_problem(out, 'arms contains duplicate names')
    arm_set = set(arms)
    for c in comps:
        if not isinstance(c, dict):
            continue
        for key in ('a', 'b'):
            arm = _dp_text(c.get(key))
            if arm is not None and arm not in arm_set:
                _dp_problem(out, f'comparison {c.get("name")!r} names undeclared {key} arm {arm!r}')
    out['checks']['n_arms'] = len(arms)

    # ---- sample size / power ----------------------------------------------
    ss = protocol.get('sample_size')
    ss = ss if isinstance(ss, dict) else {}
    planned = _dp_int(ss.get('planned_seeds'), minimum=2)
    min_n = _dp_int(ss.get('min_n'), minimum=2)
    fam_n = _dp_int(ss.get('family_size'), minimum=1)
    if planned is None:
        need(False, 'sample_size.planned_seeds must be an integer >= 2',
             missing='sample_size.planned_seeds')
    if min_n is None:
        need(False, 'sample_size.min_n must be an integer >= 2',
             missing='sample_size.min_n')
    if fam_n is None:
        need(False, 'sample_size.family_size must be an integer >= 1',
             missing='sample_size.family_size')
    elif comp_names and fam_n != len(comp_names):
        _dp_problem(out, f'sample_size.family_size={fam_n} != {len(comp_names)} declared comparisons')
    if planned is not None and min_n is not None and min_n > planned:
        _dp_problem(out, f'sample_size.min_n={min_n} exceeds planned_seeds={planned}')
    if family and fam_n is not None and fam_n != len(family):
        _dp_problem(out, f'sample_size.family_size={fam_n} != primary family size {len(family)}')
    raw_seeds = ss.get('seeds')
    if raw_seeds is not None:
        if not isinstance(raw_seeds, list):
            _dp_problem(out, 'sample_size.seeds must be a list of integer seed identities')
        else:
            seed_values = []
            for i, raw_seed in enumerate(raw_seeds):
                seed_value = _dp_int(raw_seed, minimum=0)
                if seed_value is None:
                    _dp_problem(out, f'sample_size.seeds[{i}] must be a non-negative integer')
                else:
                    seed_values.append(seed_value)
            if len(set(seed_values)) != len(seed_values):
                _dp_problem(out, 'sample_size.seeds contains duplicate seed identities; '
                                 'the same seed cannot count as two independent units')
            if planned is not None and len(seed_values) != planned:
                _dp_problem(out, f'sample_size.planned_seeds={planned} != '
                                 f'len(sample_size.seeds)={len(seed_values)}')
    target_power = _dp_number(ss.get('target_power'))
    if target_power is None:
        need(False, 'sample_size.target_power is missing',
             missing='sample_size.target_power')
    elif not (0.0 < target_power < 1.0):
        _dp_problem(out, 'sample_size.target_power must be in (0,1)')
    method = _dp_text(ss.get('power_method'))
    need(method is not None, 'sample_size.power_method is missing',
         missing='sample_size.power_method')
    if method is not None and ss.get('power_simulated') is not True:
        _dp_warn(out, 'sample_size.power_method is a projection, not a simulation on this panel\'s validation deltas; treat it as a design aid')
    for _flag in ('power_gate_required', 'power_simulation_required'):
        if _flag in ss and not isinstance(ss.get(_flag), bool):
            _dp_problem(out, f'sample_size.{_flag} must be boolean, got '
                             f'{ss.get(_flag)!r}')
    if min_n is not None and fam_n is not None and alpha is not None:
        floor_at_min = _dp_resolution_floor(fam_n, min_n)
        out['checks']['resolution_floor_at_min_n'] = floor_at_min
        if floor_at_min >= alpha:
            _dp_problem(out, f'the declared min_n={min_n}/family={fam_n} has exact Holm resolution floor {floor_at_min:.6g} >= alpha={alpha:.6g}; no effect size can be significant')
        floor_at_planned = (_dp_resolution_floor(fam_n, planned)
                            if planned is not None else None)
        if floor_at_planned is not None:
            out['checks']['resolution_floor_at_planned_n'] = floor_at_planned
        recorded_floor = _dp_number(ss.get('resolution_floor'))
        if recorded_floor is not None and floor_at_planned is not None:
            if not math.isclose(recorded_floor, floor_at_planned,
                                rel_tol=1e-12, abs_tol=0.0):
                _dp_warn(out, f'sample_size.resolution_floor={recorded_floor!r} != '
                              f'recomputed at planned n {floor_at_planned:.6g}')
    n_units = _dp_int(ss.get('n_eval_units'), minimum=1, allow_none=True)
    if n_units is not None:
        out['checks']['n_eval_units'] = n_units

    # ---- randomization / allocation --------------------------------------
    rnd = protocol.get('randomization')
    if not isinstance(rnd, dict):
        need(False, 'randomization is missing (no run-order plan)',
             missing='randomization')
        rnd = {}
    if _dp_text(rnd.get('design')) is None:
        need(False, 'randomization.design is missing', missing='randomization.design')
    seed = _dp_int(rnd.get('design_seed'), minimum=0)
    if seed is None:
        need(False, 'randomization.design_seed must be an integer >= 0',
             missing='randomization.design_seed')
    need(_dp_text(rnd.get('block')) is not None,
         'randomization.block is missing', missing='randomization.block')
    cells = rnd.get('planned_cells')
    if not isinstance(cells, list) or not cells:
        need(False, 'randomization.planned_cells must be a non-empty list',
             missing='randomization.planned_cells')
        cells = []
    else:
        if not all(isinstance(x, str) for x in cells):
            _dp_problem(out, 'randomization.planned_cells must contain only strings')
        text_cells = [str(x) for x in cells]
        if len(set(text_cells)) != len(text_cells):
            _dp_problem(out, 'randomization.planned_cells contains duplicates')
        if planned and arms:
            expected = planned * len(arms)
            if len(text_cells) != expected:
                _dp_warn(out, f'randomization.planned_cells covers {len(text_cells)} cells but planned_seeds*arms={expected}; verify the arm/seed grid')
        declared_seeds = ss.get('seeds')
        declared_seeds = (set(int(x) for x in declared_seeds
                              if _dp_int(x, allow_none=True) is not None)
                          if isinstance(declared_seeds, list) else set())
        for lab in text_cells:
            if '::seed' not in lab:
                _dp_warn(out, f'planned cell {lab!r} does not use the arm::seedN identity; its arm/seed membership cannot be checked here')
                continue
            arm, seed_text = lab.rsplit('::seed', 1)
            if arm_set and arm not in arm_set:
                _dp_problem(out, f'planned cell {lab!r} names undeclared arm {arm!r}')
            seed_value = _dp_int(seed_text, minimum=0)
            if seed_value is None and seed_text.isascii() and seed_text.isdigit():
                # ``str.isdigit()`` also accepts non-ASCII digits (e.g. '²')
                # that int() rejects; Python also caps decimal conversions at
                # a few thousand digits.  A malformed label is a reported
                # validation problem, never an exception that aborts the gate.
                try:
                    seed_value = int(seed_text)
                except (ValueError, OverflowError):
                    seed_value = None
            if seed_value is None:
                _dp_problem(out, f'planned cell {lab!r} has a non-integer seed')
            elif declared_seeds and seed_value not in declared_seeds:
                _dp_problem(out, f'planned cell {lab!r} uses seed {seed_value}, which is not in sample_size.seeds')
    if not isinstance(rnd.get('position_balance'), bool):
        _dp_warn(out, 'randomization.position_balance is not declared as a boolean')
    out['checks']['n_planned_cells'] = len(cells)

    # ---- stopping ----------------------------------------------------------
    st = protocol.get('stopping')
    st = st if isinstance(st, dict) else {}
    steps = _dp_int(st.get('training_steps'), minimum=1)
    if steps is None:
        need(False, 'stopping.training_steps must be an integer >= 1',
             missing='stopping.training_steps')
    ev = _dp_int(st.get('eval_every'), minimum=1)
    if ev is None:
        need(False, 'stopping.eval_every must be an integer >= 1',
             missing='stopping.eval_every')
    tl = _dp_int(st.get('test_looks'), minimum=1)
    if tl is None:
        need(False, 'stopping.test_looks must be an integer >= 1',
             missing='stopping.test_looks')
    al = _dp_int(st.get('analysis_looks'), minimum=1)
    if al is None:
        need(False, 'stopping.analysis_looks must be an integer >= 1',
             missing='stopping.analysis_looks')
    if st.get('allow_data_dependent_stopping') is not False:
        need(False, 'stopping.allow_data_dependent_stopping must be false for a prospective design',
             missing='stopping.allow_data_dependent_stopping')
    need(_dp_text(st.get('rule')) is not None,
         'stopping.rule is missing', missing='stopping.rule')

    # ---- analysis ----------------------------------------------------------
    an = protocol.get('analysis')
    an = an if isinstance(an, dict) else {}
    need(_dp_text(an.get('test')) is not None,
         'analysis.test is missing', missing='analysis.test')
    corr = _dp_normal_correction(an.get('correction'))
    if corr is None:
        need(False, 'analysis.correction is missing', missing='analysis.correction')
    elif strict and corr not in ('holm', 'max-t', 'bonferroni'):
        _dp_problem(out, f'analysis.correction={corr!r} is not a recognised family-wise correction')
    a_alpha = _dp_number(an.get('alpha'))
    if a_alpha is None or (alpha is not None and not math.isclose(a_alpha, alpha, rel_tol=0.0, abs_tol=1e-12)):
        _dp_problem(out, 'analysis.alpha must equal study.alpha')
    a_fam = _dp_int(an.get('family_size'), minimum=1)
    if a_fam is not None and fam_n is not None and a_fam != fam_n:
        _dp_problem(out, f'analysis.family_size={a_fam} != sample_size.family_size={fam_n}')
    need(_dp_text(an.get('code_semantics')) is not None,
         'analysis.code_semantics is missing', missing='analysis.code_semantics')
    need(_dp_text(an.get('missing_data_policy')) is not None,
         'analysis.missing_data_policy is missing',
         missing='analysis.missing_data_policy')
    impl = an.get('implementation')
    if not isinstance(impl, dict):
        need(False, 'analysis.implementation is missing: the protocol does '
                    'not bind the code that computes the endpoint',
             missing='analysis.implementation')
    else:
        _impl_sha = _dp_sha256_text(impl.get('analysis_source_sha256'))
        if _impl_sha is None:
            need(False, 'analysis.implementation.analysis_source_sha256 must '
                        'be a 64-character SHA-256 digest',
                 missing='analysis.implementation.analysis_source_sha256')
        _impl_full = _dp_sha256_text(impl.get('full_source_tree_sha256'))
        if _impl_full is None:
            need(False, 'analysis.implementation.full_source_tree_sha256 must '
                        'be a 64-character SHA-256 digest',
                 missing='analysis.implementation.full_source_tree_sha256')
        _impl_n = _dp_int(impl.get('analysis_n_files'), minimum=1)
        if _impl_n is None:
            need(False, 'analysis.implementation.analysis_n_files must be an '
                        'integer >= 1',
                 missing='analysis.implementation.analysis_n_files')
        _impl_code = _dp_text(impl.get('code_semantics'))
        if _impl_code is None:
            need(False, 'analysis.implementation.code_semantics is missing',
                 missing='analysis.implementation.code_semantics')
        elif (_dp_text(an.get('code_semantics')) is not None
              and _impl_code != _dp_text(an.get('code_semantics'))):
            _dp_problem(out, 'analysis.implementation.code_semantics does not '
                             'equal analysis.code_semantics')
        _reg_status = _dp_text(impl.get('split_registry_status'))
        if _reg_status is None:
            _dp_warn(out, 'analysis.implementation does not record the '
                          'split-registry status; executed-fragment order '
                          'cannot be checked')
        elif _reg_status != 'ok':
            _dp_warn(out, f'analysis.implementation.split_registry_status='
                          f'{_reg_status!r}; the executed analysis fragments '
                          'were not fully registered')
        _impl_unreadable = impl.get('source_unreadable')
        if isinstance(_impl_unreadable, list) and _impl_unreadable:
            _dp_problem(out, f'analysis.implementation.source_unreadable is '
                             f'non-empty ({len(_impl_unreadable)} file(s)); '
                             'the analysis closure is incomplete')
        for _key in ('analysis_prefixes', 'analysis_files'):
            _val = impl.get(_key)
            if _val is not None and (not isinstance(_val, list)
                                     or not all(isinstance(x, str)
                                                and x for x in _val)):
                _dp_problem(out, f'analysis.implementation.{_key} must be a '
                                 'list of non-empty strings')

    # ---- data --------------------------------------------------------------
    data = protocol.get('data')
    data = data if isinstance(data, dict) else {}
    need(_dp_text(data.get('selection_split')) is not None,
         'data.selection_split is missing', missing='data.selection_split')
    need(_dp_text(data.get('test_split')) is not None,
         'data.test_split is missing', missing='data.test_split')
    ta = data.get('test_access')
    if not isinstance(ta, dict):
        need(False, 'data.test_access is missing', missing='data.test_access')
        ta = {}
    elif ta.get('invalid'):
        _dp_problem(out, f"data.test_access is invalid: {ta.get('error') or 'not an object'}")
    ta_looks = _dp_int(ta.get('max_looks'), minimum=1)
    if ta_looks is None:
        need(False, 'data.test_access.max_looks must be an integer >= 1',
             missing='data.test_access.max_looks')
    elif ta_looks > int(TEST_ACCESS_POLICY['max_looks_limit']):  # noqa: F821
        _dp_problem(out, f'data.test_access.max_looks={ta_looks} exceeds the '
                         f'supported limit {TEST_ACCESS_POLICY["max_looks_limit"]}')  # noqa: F821
    ta_method = _dp_text(ta.get('method'))
    if ta_method is None:
        need(False, 'data.test_access.method is missing',
             missing='data.test_access.method')
    elif ta_method not in ('single_look', 'bonferroni'):
        _dp_problem(out, f'data.test_access.method={ta_method!r} is not one of '
                         "('single_look', 'bonferroni')")
    elif ta_method == 'single_look' and ta_looks is not None and ta_looks != 1:
        _dp_problem(out, 'data.test_access.method=single_look requires max_looks=1; '
                         'declare method=bonferroni for a repeated-look budget')
    if ta and not isinstance(ta.get('require_reason_for_reread'), bool):
        need(False, 'data.test_access.require_reason_for_reread must be boolean',
             missing='data.test_access.require_reason_for_reread')
    if ta.get('declared') is not True:
        need(False, 'the held-out access budget is the historical single-look '
                    'default, not an explicit statistics.test_access_policy '
                    'declaration', missing='data.test_access.declared')
    if ta_looks is not None and tl is not None and ta_looks != tl:
        _dp_problem(out, f'stopping.test_looks={tl} != data.test_access.max_looks={ta_looks}')
    if data.get('disjoint_declared') is False:
        _dp_problem(out, 'data.disjoint_declared is false: the test split is not disjoint from selection')
    elif data.get('disjoint_declared') is not True:
        _dp_warn(out, 'data.disjoint_declared is not recorded as true; disjointness is unverified by this protocol')

    # ---- operations --------------------------------------------------------
    ops = protocol.get('operations')
    ops = ops if isinstance(ops, dict) else {}
    dispo = ops.get('disposition')
    if not isinstance(dispo, dict):
        need(False, 'operations.disposition is missing (failed-cell policy not pre-registered)',
             missing='operations.disposition')
    else:
        if dispo.get('invalid'):
            _dp_problem(out, f"operations.disposition is invalid: {dispo.get('error') or dispo.get('raw_type')}")
        if strict:
            if dispo.get('retention') != 'first_success':
                _dp_problem(out, "a confirmatory protocol must set disposition.retention='first_success'")
            if dispo.get('replacement') != 'same_cell':
                _dp_problem(out, "a confirmatory protocol must set disposition.replacement='same_cell'")
            if dispo.get('allow_supersede') is not False:
                _dp_problem(out, 'a confirmatory protocol must set disposition.allow_supersede=false')
        if _dp_int(dispo.get('max_attempts_per_cell'), minimum=1) is None:
            _dp_problem(out, 'operations.disposition.max_attempts_per_cell must be an integer >= 1')

    # ---- identity / history ------------------------------------------------
    recorded = _dp_text(protocol.get('content_sha256'))
    if recorded is None:
        need(False, 'content_sha256 is missing; the protocol body is not bound',
             missing='content_sha256')
    else:
        try:
            want = design_protocol_content_sha(protocol)
        except (TypeError, ValueError) as exc:
            _dp_problem(out, f'protocol body is not canonical JSON: {type(exc).__name__}: {exc}')
        else:
            if recorded != want:
                _dp_problem(out, 'content_sha256 does not reproduce from the protocol body; the registered design was edited')
    commitments = protocol.get('commitments')
    if not isinstance(commitments, dict) or not commitments:
        need(False, 'commitments must be an object with one hash per section',
             missing='commitments')
    else:
        for section in DESIGN_PROTOCOL_SECTIONS:
            got = _dp_text(commitments.get(section))
            try:
                want = _dp_sha_obj(protocol.get(section))
            except (TypeError, ValueError, RecursionError) as exc:
                _dp_problem(out, f'commitments.{section} is not canonical JSON: '
                                 f'{type(exc).__name__}: {exc}')
                continue
            if got is None or got != want:
                _dp_problem(out, f'commitments.{section} does not match the section body')
    rev = _dp_int(protocol.get('revision'), minimum=0, allow_none=True)
    if rev is None:
        _dp_problem(out, 'revision must be an integer >= 0')
        rev = 0
    sup = protocol.get('supersedes')
    if not isinstance(sup, list):
        _dp_problem(out, 'supersedes must be a list')
    elif rev == 0 and sup:
        _dp_problem(out, 'a first protocol (revision=0) cannot supersede anything')
    elif rev > 0:
        if not sup:
            _dp_problem(out, 'a revised protocol must list the superseded content hash')
        if _dp_text(protocol.get('reason')) is None:
            _dp_problem(out, 'a revised protocol must carry a non-empty reason')
    created = _dp_timestamp(protocol.get('created_at'))
    if created is None:
        need(False, 'created_at is missing or not a parseable timestamp',
             missing='created_at')

    out['checks']['strict'] = bool(strict)
    if out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out


def design_protocol_conformance(protocol, manifest, source_tree=None):
    """Check that the registration describes the manifest being run.

    ``source_tree`` is injectable so an audit can replay a recorded tree; when
    omitted, the current checkout is hashed.  The analysis-closure digest is a
    hard invalidation.  Whole-tree drift outside that closure is a warning:
    another panel or the dynamic draft may legitimately change while a
    fixed-chunk registration is pending.
    """
    out = {'kind': 'design_protocol_conformance', 'status': 'ok',
           'problems': [], 'warnings': [], 'implementation': None}
    if not isinstance(protocol, dict) or not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['problems'].append('protocol and manifest must be objects')
        return out
    study = protocol.get('study') if isinstance(protocol.get('study'), dict) else {}
    want = _dp_text(study.get('manifest_sha256'))
    got = _dp_manifest_sha(manifest)
    if want != got:
        out['problems'].append('study.manifest_sha256 does not match the supplied manifest body')
    stats = manifest.get('statistics') if isinstance(manifest.get('statistics'), dict) else {}
    comps = manifest.get('primary_comparisons')
    comps = comps if isinstance(comps, list) else []
    manifest_names = []
    for c in comps:
        if isinstance(c, dict):
            n = _dp_text(c.get('name')) or f'{c.get("a")} - {c.get("b")}'
            if n:
                manifest_names.append(n)
    proto_comps = protocol.get('comparisons')
    proto_comps = proto_comps if isinstance(proto_comps, list) else []
    proto_names = [_dp_text(c.get('name')) for c in proto_comps
                   if isinstance(c, dict) and _dp_text(c.get('name'))]
    if set(proto_names) != set(manifest_names):
        out['problems'].append('comparison names do not match manifest.primary_comparisons')
    rq = (protocol.get('research_question')
          if isinstance(protocol.get('research_question'), dict) else {})
    p_hyps = rq.get('hypotheses') if isinstance(rq.get('hypotheses'), list) else []
    p_hids = [str(h.get('id')) for h in p_hyps
              if isinstance(h, dict) and h.get('id') is not None]
    raw_hyps = manifest.get('hypotheses')
    if isinstance(raw_hyps, dict) and raw_hyps:
        if set(p_hids) != set(map(str, raw_hyps)):
            out['problems'].append(
                'hypothesis ids do not match manifest.hypotheses')
    p_arms = protocol.get('arms') if isinstance(protocol.get('arms'), list) else []
    m_arms = manifest.get('arms') if isinstance(manifest.get('arms'), list) else []
    if set(map(str, p_arms)) != set(map(str, m_arms)):
        out['problems'].append('arms do not match manifest.arms')
    ss = protocol.get('sample_size') if isinstance(protocol.get('sample_size'), dict) else {}
    p_n = _dp_int(ss.get('planned_seeds'), minimum=1)
    m_seeds = manifest.get('seeds')
    m_n = len(m_seeds) if isinstance(m_seeds, (list, tuple)) else None
    if p_n is not None and m_n is not None and p_n != m_n:
        out['problems'].append(f'sample_size.planned_seeds={p_n} != manifest seed count {m_n}')
    p_min = _dp_int(ss.get('min_n'), minimum=1)
    m_min = _dp_int(stats.get('min_seeds_for_confirmation'), minimum=1)
    if p_min is not None and m_min is not None and p_min != m_min:
        out['problems'].append(f'sample_size.min_n={p_min} != statistics.min_seeds_for_confirmation={m_min}')
    p_fam = _dp_int(ss.get('family_size'), minimum=1)
    if p_fam is not None and p_fam != len(manifest_names):
        out['problems'].append(f'sample_size.family_size={p_fam} != {len(manifest_names)} primary comparisons')
    study_alpha = _dp_number(study.get('alpha'))
    m_alpha = _dp_number(stats.get('alpha', manifest.get('alpha')))
    if study_alpha is not None and m_alpha is not None and not math.isclose(
            study_alpha, m_alpha, rel_tol=0.0, abs_tol=1e-12):
        out['problems'].append(f'study.alpha={study_alpha} != manifest alpha={m_alpha}')
    an = protocol.get('analysis') if isinstance(protocol.get('analysis'), dict) else {}
    p_corr = _dp_normal_correction(an.get('correction'))
    m_corr = _dp_normal_correction(stats.get('correction'))
    if p_corr is not None and m_corr is not None and p_corr != m_corr:
        out['problems'].append(f'analysis.correction={p_corr!r} != manifest correction={m_corr!r}')
    p_code = _dp_text(study.get('code_semantics'))
    m_code = _dp_text(manifest.get('code_semantics_expected'))
    if p_code is not None and m_code is not None and p_code != m_code:
        out['problems'].append(f'study.code_semantics={p_code!r} != manifest code_semantics_expected={m_code!r}')
    # If the manifest explicitly declares a held-out budget or disposition
    # policy, the protocol must carry the same declaration.
    ta_manifest = None
    if isinstance(stats.get('test_access_policy'), dict):
        ta_manifest = stats.get('test_access_policy')
    elif isinstance(manifest.get('test_access_policy'), dict):
        ta_manifest = manifest.get('test_access_policy')
    data = protocol.get('data') if isinstance(protocol.get('data'), dict) else {}
    p_ta = data.get('test_access') if isinstance(data.get('test_access'), dict) else {}
    if ta_manifest is not None:
        m_looks = _dp_int(ta_manifest.get('max_looks'), minimum=1)
        p_looks = _dp_int(p_ta.get('max_looks'), minimum=1)
        if m_looks is not None and p_looks is not None and m_looks != p_looks:
            out['problems'].append(f'data.test_access.max_looks={p_looks} != manifest budget {m_looks}')
        m_method = _dp_text(ta_manifest.get('method'))
        p_method = _dp_text(p_ta.get('method'))
        if m_method is not None and p_method is not None and m_method != p_method:
            out['problems'].append(f'data.test_access.method={p_method!r} != manifest method {m_method!r}')
    m_disp = None
    if isinstance(stats.get('disposition_policy'), dict):
        m_disp = stats.get('disposition_policy')
    elif isinstance(manifest.get('disposition_policy'), dict):
        m_disp = manifest.get('disposition_policy')
    if isinstance(m_disp, dict) and m_disp.get('kind') == 'run_disposition':
        # A manifest may embed the complete precommitment plan rather than
        # only its policy block; compare the policy, not the wrapper keys.
        m_disp = m_disp.get('policy')
    ops = protocol.get('operations') if isinstance(protocol.get('operations'), dict) else {}
    p_disp = ops.get('disposition') if isinstance(ops.get('disposition'), dict) else {}
    if m_disp is not None and not p_disp.get('invalid'):
        try:
            m_pol = validate_run_disposition_policy(m_disp)  # noqa: F821
        except Exception as exc:
            out['warnings'].append(f'manifest disposition policy is invalid: {type(exc).__name__}: {exc}')
        else:
            for key in ('retention', 'replacement', 'allow_supersede'):
                if p_disp.get(key) != m_pol.get(key):
                    out['problems'].append(
                        f'operations.disposition.{key}={p_disp.get(key)!r} != manifest {m_pol.get(key)!r}')
    # The design is only pre-registered if the implementation that computes
    # the declared endpoint is part of the registration.  A manifest match
    # with a changed analysis closure is exactly the post-registration drift
    # this gate exists to catch.
    impl = _dp_implementation_audit(protocol, source_tree=source_tree)
    out['implementation'] = impl
    level = _dp_text(study.get('level'))
    if impl.get('status') == 'missing':
        if level in DESIGN_PROTOCOL_POLICY['strict_levels']:
            out['problems'].append(
                'analysis.implementation is missing: a confirmatory '
                'registration must bind the analysis closure')
        else:
            out['warnings'].append(
                'analysis.implementation is missing: the design is not bound '
                'to the executable analysis')
    else:
        out['problems'].extend(impl.get('problems') or [])
        out['warnings'].extend(impl.get('warnings') or [])
    if out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    return out


def design_protocol_timing(protocol, events):
    """Check that the protocol timestamp precedes the first recorded event."""
    out = {'kind': 'design_protocol_timing', 'status': 'not_verifiable',
           'protocol_created_at': None, 'first_event_at': None,
           'first_event_kind': None, 'n_events': 0,
           'reconstruction': None, 'warnings': [], 'problems': []}
    if not isinstance(protocol, dict):
        out['status'] = 'fail'
        out['problems'].append('protocol must be an object')
        return out
    created = _dp_timestamp(protocol.get('created_at'))
    out['protocol_created_at'] = protocol.get('created_at')
    if created is None:
        out['warnings'].append('protocol.created_at is missing or unparseable; timing cannot be verified')
        return out
    if events is None:
        events_iter = iter(())
    elif isinstance(events, str):
        # A path is a convenience for callers that already have a ledger.
        events_iter = _dp_iter_events(events=events, panel_dirs=[])
    else:
        try:
            events_iter = iter(events)
        except TypeError:
            events_iter = iter(())
    first = None
    first_kind = None
    n = 0
    # Analysis/report events are post-run reads, not execution starts.  They
    # must not make a protocol look late merely because an exploratory
    # analysis happened before the confirmatory registration was compiled.
    _post_run = {'analysis', 'report', 'audit', 'aggregate', 'look',
                 'synthesis', 'reproduction'}
    for ev in events_iter:
        if not isinstance(ev, dict):
            continue
        if str(ev.get('kind') or '').lower() in _post_run:
            continue
        n += 1
        stamp = ev.get('started_at') or ev.get('created_at')
        when = _dp_timestamp(stamp)
        if when is None or not _dp_comparable_times(created, when):
            continue
        if first is None or when < first:
            first = when
            first_kind = ev.get('kind')
    out['n_events'] = n
    if first is None:
        if n == 0:
            out['warnings'].append('no recorded events; the protocol could not be ordered against execution')
        else:
            out['warnings'].append('no event carried a comparable timestamp; timing cannot be verified')
        return out
    out['first_event_at'] = first.isoformat()
    out['first_event_kind'] = first_kind
    if created > first:
        out['status'] = 'fail'
        out['reconstruction'] = True
        out['problems'].append(
            f'protocol.created_at={protocol.get("created_at")!r} is later than '
            f'the first recorded {first_kind or "event"} at {first.isoformat()}; '
            'this is a reconstruction, not a pre-registration')
        return out
    out['status'] = 'ok'
    out['reconstruction'] = False
    return out


def _dp_reject_json_constant(value):
    """Reject NaN/Infinity: a protocol is canonical JSON, not lenient JSON."""
    raise ValueError(f'non-finite JSON number {value!r} is not allowed')


def _dp_load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh, parse_constant=_dp_reject_json_constant)


def _dp_iter_jsonl(path):
    """Yield object records from one JSONL file, skipping malformed lines.

    Streaming keeps the first-event check usable on a long experiment ledger:
    the audit only needs the earliest timestamp, not every record in memory.
    """
    try:
        fh = open(path, encoding='utf-8')
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(rec, dict):
                yield rec


def _dp_iter_events(events=None, panel_dirs=None):
    """Yield event records from an in-memory stream/path plus panel ledgers.

    An explicit ``panel_dirs=[]`` means "do not discover any panels"; only
    ``None`` (the default) triggers the ``results_*`` scan.  Treating an empty
    list as "unspecified" silently mixed repository history into a caller's
    deterministic event fixture and made the timing check depend on which
    result directories happened to exist at scale.
    """
    if isinstance(events, str):
        yield from _dp_iter_jsonl(events)
    elif events is not None:
        try:
            iterator = iter(events)
        except TypeError:
            iterator = iter(())
        for rec in iterator:
            if isinstance(rec, dict):
                yield rec
    if panel_dirs is None:
        root = globals().get('_REPO') or os.getcwd()
        try:
            import glob as _glob
            dirs = sorted(_glob.glob(os.path.join(root, 'results_*')))
        except Exception:
            dirs = []
    elif isinstance(panel_dirs, str):
        dirs = [panel_dirs]
    elif isinstance(panel_dirs, (list, tuple)):
        dirs = [str(x) for x in panel_dirs]
    else:
        dirs = []
    for d in dirs:
        path = os.path.join(str(d), 'experiments.jsonl')
        if os.path.isfile(path):
            yield from _dp_iter_jsonl(path)


def _dp_read_events(events=None, panel_dirs=None):
    """Materialised compatibility wrapper; audits should stream instead."""
    return list(_dp_iter_events(events=events, panel_dirs=panel_dirs))



def design_protocol_audit(path=None, manifest_path=None, events=None,
                          panel_dirs=None, require_complete=None,
                          verbose=False, source_tree=None):
    """Audit one registered design protocol against the manifest and history.

    The gate fails closed on a tampered body, a manifest mismatch, a
    confirmatory protocol that lacks a design decision, a changed analysis
    closure, or a protocol written after the first event.  An absent protocol
    is ``not_verifiable_locally`` (with the required sections and a compile
    hint), not a green pass.
    """
    root = globals().get('_REPO') or os.getcwd()
    candidates = []
    if path is not None:
        candidates = [str(path)]
    else:
        candidates = [os.path.join(root, 'design_protocol.json'),
                      os.path.join(root, 'analysis_v12', 'design_protocol.json')]
        try:
            import glob as _glob
            candidates.extend(sorted(_glob.glob(
                os.path.join(root, 'results_*', 'design_protocol.json'))))
        except Exception:
            pass
    chosen = next((p for p in candidates if os.path.isfile(p)), None)
    out = {'kind': 'design_protocol_audit', 'status': 'not_verifiable_locally',
           'path': chosen or (candidates[0] if candidates else None),
           'candidates': [str(p) for p in candidates],
           'manifest_path': manifest_path,
           'validation': None, 'conformance': None, 'timing': None,
           'implementation': None,
           'problems': [], 'warnings': []}
    if chosen is None:
        out['required_sections'] = list(DESIGN_PROTOCOL_SECTIONS)
        out['reason'] = ('no design_protocol.json found; the design decisions '
                         'are not registered in one prospective object')
        out['warnings'].append(
            'compile a protocol from the frozen manifest and the existing '
            'run_order/run_disposition artifacts before the first run; an '
            'absent registration is not evidence that the study was not '
            'pre-registered, but this tree cannot verify it')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    try:
        protocol = _dp_load_json(chosen)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(f'protocol is unreadable: {type(exc).__name__}: {exc}')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    validation = validate_design_protocol(protocol,
                                          require_complete=require_complete)
    out['validation'] = validation
    if validation['status'] == 'fail':
        out['problems'].extend(f'validation: {x}' for x in validation['problems'])
    out['warnings'].extend(f'validation: {x}' for x in validation['warnings'])
    # The manifest is the source of truth for the study being run.  When the
    # caller does not name one, use the same defaults as the design preflight.
    mp = manifest_path
    if mp is None:
        mp = os.path.join(root, 'confirmation_frozen.json')
    if os.path.isfile(mp):
        try:
            manifest = _dp_load_json(mp)
        except Exception as exc:
            out['warnings'].append(f'manifest is unreadable ({type(exc).__name__}: {exc}); conformance not checked')
        else:
            conf = design_protocol_conformance(protocol, manifest,
                                               source_tree=source_tree)
            out['conformance'] = conf
            out['implementation'] = conf.get('implementation')
            if conf['status'] == 'fail':
                out['problems'].extend(f'conformance: {x}' for x in conf['problems'])
            out['warnings'].extend(f'conformance: {x}' for x in conf['warnings'])
    else:
        out['warnings'].append(f'no readable manifest at {mp}; conformance not checked')
    evs = _dp_iter_events(events=events, panel_dirs=panel_dirs)
    timing = design_protocol_timing(protocol, evs)
    out['timing'] = timing
    if timing['status'] == 'fail':
        out['problems'].extend(f'timing: {x}' for x in timing['problems'])
    out['warnings'].extend(f'timing: {x}' for x in timing['warnings'])
    if out['problems']:
        out['status'] = 'fail'
    elif out['warnings'] or (validation['status'] == 'warnings'):
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def write_design_protocol(outdir, protocol, *, keep_existing=True):
    """Write a validated protocol without silently rewriting a commitment.

    Same content is a no-op.  A different body is refused unless the caller
    explicitly permits replacement **and** the new object is a valid revision
    that names the old ``content_sha256`` in ``supersedes``.  This mirrors the
    run-disposition writer: history is append-only, not overwritten.
    """
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    val = validate_design_protocol(protocol)
    if val['status'] == 'fail':
        raise ValueError('refusing to write an invalid design protocol: '
                         + '; '.join(val['problems'][:5]))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'design_protocol.json')
    if os.path.isfile(path):
        try:
            old = _dp_load_json(path)
        except Exception as exc:
            raise ValueError(f'existing design_protocol.json is unreadable: '
                             f'{type(exc).__name__}: {exc}') from None
        old_sha = old.get('content_sha256') if isinstance(old, dict) else None
        new_sha = protocol.get('content_sha256')
        if old_sha == new_sha:
            return {'written': False, 'path': path, 'conflict': False,
                    'noop': True, 'content_sha256': new_sha}
        if keep_existing:
            return {'written': False, 'path': path, 'conflict': True,
                    'existing_sha256': old_sha, 'content_sha256': new_sha,
                    'reason': ('a different protocol already exists; pass '
                               'keep_existing=False with a revision that '
                               'supersedes it to replace it')}
        old_rev = old.get('revision', 0) if isinstance(old, dict) else 0
        old_rev = _dp_int(old_rev, minimum=0, allow_none=True)
        if old_rev is None:
            old_rev = 0
        new_rev = protocol.get('revision', 0)
        if _dp_int(new_rev, minimum=0, allow_none=True) is None:
            raise ValueError('replacement revision must be an integer >= 0')
        if int(new_rev) <= old_rev:
            raise ValueError('a replacement protocol must advance the revision')
        if int(new_rev) <= 0:
            raise ValueError('a replacement protocol must have revision >= 1')
        sup = protocol.get('supersedes')
        if not isinstance(sup, list) or old_sha not in sup:
            raise ValueError('a replacement protocol must list the existing '
                             'content_sha256 in supersedes')
        if _dp_text(protocol.get('reason')) is None:
            raise ValueError('a replacement protocol must carry a reason')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(protocol, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write('\n')
    os.replace(tmp, path)
    return {'written': True, 'path': path, 'conflict': False,
            'content_sha256': protocol.get('content_sha256')}


def design_protocol_preflight_summary(path=None, manifest_path=None):
    """Compact block attached to ``audit_design_preflight``.

    The standalone gate keeps the full protocol; the design preflight only
    needs to show whether the study has one and which decisions are missing.
    """
    rep = design_protocol_audit(path=path, manifest_path=manifest_path)
    validation = rep.get('validation') or {}
    return {
        'status': rep.get('status'),
        'path': rep.get('path'),
        'level': validation.get('level'),
        'strict': validation.get('strict'),
        'missing': list(validation.get('missing') or []),
        'n_problems': len(rep.get('problems') or []),
        'n_warnings': len(rep.get('warnings') or []),
        'timing': (rep.get('timing') or {}).get('status'),
        'conformance': (rep.get('conformance') or {}).get('status'),
        'implementation': (rep.get('implementation') or {}).get('status'),
        'problems': list(rep.get('problems') or [])[:12],
        'warnings': list(rep.get('warnings') or [])[:12],
    }
