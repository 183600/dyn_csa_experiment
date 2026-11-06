"""Validators must fail closed on array/huge-integer tampering, never raise.

The earlier array-tamper checks covered the replayed validators that have a
dedicated structural comparator.  These checks extend the same contract to the
remaining artifact readers: a hand-edited (or in-memory, never JSON-encoded)
value in a scalar slot must produce a structured ``fail`` verdict, and a
malformed article-sidecar count must produce a problem string rather than
CPython's integer-to-decimal or NumPy ambiguous-truth exception.

CPU-only, zero-experiment checks.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
HUGE = 10 ** 10000


def _strict(value):
    json.dumps(value, allow_nan=False)
    return value


def _set_path(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _assert_fails(fn, artifact, path, value=None):
    bad = copy.deepcopy(artifact)
    _set_path(bad, path, np.array([1, 2]) if value is None else value)
    verdict = fn(bad)
    assert verdict['status'] == 'fail', (path, verdict)
    assert verdict['problems'], (path, verdict)
    _strict(verdict)


def test_type_sm_and_exact_tost_array_fields_fail_closed():
    tsm = L.build_type_sm_design(12, 1.0, 0.5)
    _assert_fails(L.validate_type_sm_design, tsm, ('kind',))

    tost = L.build_exact_tost([0.1, -0.2, 0.3, 0.05, -0.1, 0.2], 0.5)
    for path in (('kind',), ('result', 'kind'), ('result', 'status'),
                 ('result', 'n'), ('result', 'p_tost')):
        _assert_fails(L.validate_exact_tost, tost, path)


def test_ablation_screening_array_schema_fields_fail_closed():
    import test_artifact_validator_structural_regressions as structural

    cells = structural._ablation_cells(range(4), ('a', 'b'))
    art = L.build_ablation_screening(cells, components=['a', 'b'])
    for path in (('kind',), ('schema',)):
        _assert_fails(L.validate_ablation_screening, art, path)


def test_numerical_canary_array_fields_fail_closed():
    art = L.build_numerical_canary_artifact(seed=1, size=4)
    for path in (('kind',), ('inputs', 'seed'), ('inputs', 'size'),
                 ('inputs', 'dtype'), ('inputs', 'device'),
                 ('result', 'kind'), ('result', 'status')):
        _assert_fails(L.validate_numerical_canary_artifact, art, path)


def test_article_sidecar_array_and_huge_count_metadata_is_a_problem():
    token_articles = np.asarray([10, 10, 11, 11, 10, 10, 11, 11],
                                dtype=np.uint64)
    token_docs = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint64)
    block = L.article_sidecar_block(
        token_articles, split='test', source='unit_test',
        extra={'n_lines': 4})
    block['sidecar_file'] = 'unused.npy'
    block['sidecar_sha256'] = 'a' * 64
    meta = {'article_sidecars': True, 'document_unit': 'line',
            'article_unit': block}

    def problems(edited_meta):
        found = L.article_sidecar_problems(
            edited_meta, token_articles, token_docs, split='test',
            n_tokens=8)
        assert isinstance(found, list)
        _strict(found)
        return found

    for path in (('article_unit', 'schema'), ('article_unit', 'unit'),
                 ('article_unit', 'split')):
        bad = copy.deepcopy(meta)
        _set_path(bad, path, np.array([1, 2]))
        assert problems(bad), path

    for key in ('token_article_count', 'n_articles', 'n_lines'):
        bad = copy.deepcopy(meta)
        bad['article_unit'][key] = HUGE
        found = problems(bad)
        assert any(key in problem for problem in found), (key, found)


def test_design_protocol_and_clearance_array_fields_fail_closed():
    import test_review_part_68 as protocol_fixture
    import test_review_part_72 as clearance_fixture

    protocol = protocol_fixture._protocol()
    _assert_fails(L.validate_design_protocol, protocol, ('kind',))
    _assert_fails(L.validate_design_protocol, protocol, ('schema',))
    _assert_fails(L.validate_design_protocol, protocol,
                  ('operations', 'disposition', 'retention'))
    _assert_fails(L.validate_design_protocol, protocol,
                  ('operations', 'disposition', 'replacement'))

    clearance = clearance_fixture._clearance(protocol)

    def validate(artifact):
        return L.validate_preparation_clearance(artifact, protocol=protocol)

    for path in (('kind',), ('schema',),
                 ('study', 'design_protocol_sha256'),
                 ('checks', 'protocol', 'status'),
                 ('checks', 'power', 'status'),
                 ('status',)):
        _assert_fails(validate, clearance, path)


def test_analysis_rehearsal_array_schema_fields_fail_closed():
    import test_review_part_74 as rehearsal_fixture

    rehearsal = L.build_analysis_rehearsal(
        manifest_path=rehearsal_fixture.MANIFEST)
    for path in (('kind',), ('schema',)):
        _assert_fails(
            lambda artifact: L.validate_analysis_rehearsal(
                artifact, manifest=rehearsal_fixture._manifest()),
            rehearsal, path)

