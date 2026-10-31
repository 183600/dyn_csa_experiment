# Cross-panel synthesis replay: does the rendered pool follow from its inputs?
#
# ``ablation_coverage_across_panels`` now carries a ``synthesis`` block that
# pools every estimable panel's controlled per-seed contrast (not only the
# panels whose p-value cleared alpha).  That block is a quotable result, so it
# gets the same treatment as the other analysis gates: the audit recomputes the
# pooled estimate, heterogeneity, panel-dependence classification and
# selection-bias readout from the block's own recorded points, and fails when a
# derived number, a claim flag or a point input was changed.
#
# The gate is an internal replay, not a substitute for the per-panel admission
# gate: it can prove "this pool follows from these per-seed effects", not that
# those effects came from a particular raw run.  The latter is covered by the
# stamped analysis artifacts and the confirmation freeze lock.
#
# This fragment is executed into the audit_gates namespace, which already
# provides json/math/os, REPO, _load and _finite_number.


def _ps_close(a, b, *, rel_tol=1e-9, abs_tol=1e-12):
    x = _finite_number(a)
    y = _finite_number(b)
    if x is None or y is None:
        return x is None and y is None
    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)


def _ps_compare(stored, actual, *, rel_tol=1e-9, abs_tol=1e-12, path=''):
    """Compare two JSON-like trees, returning a list of mismatch paths."""
    problems = []
    if isinstance(stored, dict) and isinstance(actual, dict):
        for key in sorted(set(stored) | set(actual)):
            if key not in stored:
                problems.append(f'{path}/{key}: recomputed but absent from the artifact')
            elif key not in actual:
                problems.append(f'{path}/{key}: stored but not recomputed')
            else:
                problems.extend(_ps_compare(stored[key], actual[key],
                                            rel_tol=rel_tol, abs_tol=abs_tol,
                                            path=f'{path}/{key}'))
        return problems
    if isinstance(stored, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(stored) != len(actual):
            return [f'{path}: length {len(stored)} != {len(actual)}']
        for i, (a, b) in enumerate(zip(stored, actual)):
            problems.extend(_ps_compare(a, b, rel_tol=rel_tol, abs_tol=abs_tol,
                                        path=f'{path}[{i}]'))
        return problems
    if isinstance(stored, bool) or isinstance(actual, bool):
        if bool(stored) != bool(actual):
            problems.append(f'{path}: {stored!r} != {actual!r}')
        return problems
    if isinstance(stored, (int, float)) or isinstance(actual, (int, float)):
        if not _ps_close(stored, actual, rel_tol=rel_tol, abs_tol=abs_tol):
            problems.append(f'{path}: {stored!r} != {actual!r}')
        return problems
    if stored != actual:
        problems.append(f'{path}: {stored!r} != {actual!r}')
    return problems


def _ps_find_block(rec):
    if not isinstance(rec, dict):
        return None, None
    if rec.get('kind') == 'cross_panel_synthesis' or (
            'components' in rec and 'policy' in rec
            and 'n_contrasts' in rec):
        return rec, 'synthesis'
    meth = rec.get('methodology')
    if isinstance(meth, dict):
        cov = meth.get('ablation_coverage')
        if isinstance(cov, dict) and isinstance(cov.get('synthesis'), dict):
            return cov['synthesis'], 'methodology.ablation_coverage.synthesis'
    top = rec.get('panel_synthesis')
    if isinstance(top, dict):
        return top, 'panel_synthesis'
    cov = rec.get('ablation_coverage')
    if isinstance(cov, dict) and isinstance(cov.get('synthesis'), dict):
        return cov['synthesis'], 'ablation_coverage.synthesis'
    return None, None


def audit_panel_synthesis(path=None):
    """Replay the cross-panel synthesis block from its own recorded points.

    A missing synthesis block is ``not_verifiable_locally`` (no panel evidence
    or an older report), never a pass.  A synthesised block whose pooled
    estimate, heterogeneity, claim flag, panel-dependence tier, exclusion
    reason or input point cannot be rebuilt from the recorded per-seed effects
    is ``fail``.  The gate never changes a panel verdict, p-value or Holm
    family.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'audit_report.json')
    out = {'kind': 'panel_synthesis_audit', 'status': 'ok', 'path': p,
           'n_contrasts': 0, 'n_ok': 0, 'n_skipped': 0, 'n_fail': 0,
           'contrasts': {}, 'problems': [], 'warnings': [],
           'note': ('internal replay of the cross-panel synthesis from its '
                    'recorded per-seed effects; it does not re-admit a panel '
                    'record or change any panel verdict')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no audit report or synthesis artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    rec = _load(p)
    if rec is None:
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    block, where = _ps_find_block(rec)
    if not isinstance(block, dict):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the report carries no cross-panel synthesis block; '
                         'run audit_gates.py report (or methodology) first')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    out['block_path'] = where
    policy = block.get('policy') if isinstance(block.get('policy'), dict) else {}
    alpha = block.get('alpha', policy.get('alpha', 0.05))
    min_n = block.get('min_n', policy.get('min_n', 3))
    try:
        alpha = float(alpha)
        min_n = int(min_n)
    except (TypeError, ValueError, OverflowError):
        out['status'] = 'fail'
        out['problems'].append('the synthesis policy alpha/min_n is malformed')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    forbidden = {'confirmed', 'p_holm', 'verdict', 'decision'}
    bad = sorted(k for k in block if k in forbidden)
    if bad:
        out['problems'].append(
            'the synthesis block must not carry frozen decision fields: '
            f'{bad}')
    comps = block.get('components')
    if not isinstance(comps, dict):
        out['status'] = 'fail'
        out['problems'].append('the synthesis block has no components mapping')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    poolable_declared = set(block.get('poolable_components') or [])
    poolable_actual = set()
    for comp, cb in comps.items():
        if not isinstance(cb, dict):
            continue
        contrasts = cb.get('contrasts')
        if not isinstance(contrasts, dict):
            continue
        for name, b in contrasts.items():
            out['n_contrasts'] += 1
            label = f'{comp}::{name}'
            if not isinstance(b, dict):
                out['problems'].append(f'{label}: contrast block is not an object')
                out['n_fail'] += 1
                continue
            status = b.get('status')
            if status in ('not_measured', 'not_estimable',
                          'refused_direction_mismatch'):
                # Nothing was pooled; only check that no fabricated pooled
                # value is present and that the exclusion reasons exist.
                if b.get('claim_allowed') is True:
                    out['problems'].append(
                        f'{label}: a non-poolable contrast claims a common effect')
                if status == 'refused_direction_mismatch' and not (
                        isinstance(b.get('direction'), dict)
                        and b['direction'].get('consistent') is False):
                    out['problems'].append(
                        f'{label}: direction refusal without an inconsistent '
                        'direction record')
                out['n_skipped'] += 1
                out['contrasts'][label] = {'status': status, 'audit': 'skipped'}
                continue
            points = ((b.get('meta_input') or {}).get('points')
                      if isinstance(b.get('meta_input'), dict) else None)
            panels = b.get('panels')
            if not isinstance(points, list) or not isinstance(panels, dict) or not points:
                out['problems'].append(
                    f'{label}: pooled status without recorded per-seed points')
                out['n_fail'] += 1
                continue
            labels = [str((pt or {}).get('label')) for pt in points
                      if isinstance(pt, dict)]
            if len(labels) != len(points) or len(set(labels)) != len(labels):
                out['problems'].append(
                    f'{label}: meta input points have missing/duplicate labels')
                out['n_fail'] += 1
                continue
            if set(labels) != set(panels):
                out['problems'].append(
                    f'{label}: meta input panels {sorted(labels)} do not match '
                    f'the rendered panels {sorted(panels)}')
                out['n_fail'] += 1
                continue
            # Rebuild the per-seed effects from the recorded deltas?  The
            # synthesis stores the derived percent effects, not the raw deltas;
            # the point input must at least agree with the rendered panel rows
            # bit-for-bit within the JSON tolerance.
            input_bad = []
            for pt in points:
                row = panels.get(str(pt.get('label')))
                if not isinstance(row, dict):
                    input_bad.append(f"missing panel row {pt.get('label')!r}")
                    continue
                input_bad.extend(
                    _ps_compare(pt.get('effects'), row.get('effects'),
                                path=f"{pt.get('label')}:effects"))
                input_bad.extend(
                    _ps_compare(pt.get('seeds'), row.get('seeds'),
                                path=f"{pt.get('label')}:seeds"))
                if row.get('n') != len(pt.get('effects') or []):
                    input_bad.append(f"{pt.get('label')}: n does not match effects")
            if input_bad:
                out['problems'].extend(f'{label}: {x}' for x in input_bad)
                out['n_fail'] += 1
                continue
            try:
                meta = L.meta_analyze_effects(points, alpha=alpha,
                                              min_seeds=min_n)
            except Exception as exc:
                out['problems'].append(
                    f'{label}: meta-analysis replay failed '
                    f'({type(exc).__name__}: {exc})')
                out['n_fail'] += 1
                continue
            stored_meta = b.get('meta') if isinstance(b.get('meta'), dict) else {}
            actual_meta = {
                'status': meta.get('status'),
                'fixed_effect': meta.get('fixed_effect'),
                'random_effects': meta.get('random_effects'),
                'heterogeneity': meta.get('heterogeneity'),
                'prediction_interval': meta.get('prediction_interval'),
                'seed_dependence': meta.get('seed_dependence'),
                'seed_overlap': meta.get('seed_overlap'),
                'paired_common_effect': meta.get('paired_common_effect'),
                'poolable': meta.get('poolable'),
                'corpus_binding': meta.get('corpus_binding'),
                'underpowered_points': meta.get('underpowered_points'),
            }
            for key, actual in actual_meta.items():
                if key in ('note',):
                    continue
                if key in stored_meta:
                    out['problems'].extend(
                        f'{label}/{key}: {x}' for x in
                        _ps_compare(stored_meta.get(key), actual,
                                    path=f'{label}/{key}'))
                else:
                    out['problems'].append(
                        f'{label}: stored meta block omits {key!r}')
            # Claim/dependence replay.  The draw identity is part of the
            # rendered panel row, so independence is recomputed rather than
            # trusted from the stored tier.
            direction = b.get('direction') if isinstance(b.get('direction'), dict) else {}
            corpus = b.get('corpus') if isinstance(b.get('corpus'), dict) else {}
            draws = {str(pt.get('label')): (panels.get(str(pt.get('label'))) or {}).get('draws')
                     for pt in points}
            try:
                ind2 = L.replication_independence_audit(draws)
            except Exception as exc:
                out['problems'].append(
                    f'{label}: independence replay failed '
                    f'({type(exc).__name__}: {exc})')
                ind2 = {'status': None}
            ind = b.get('independence')
            if isinstance(ind, dict):
                out['problems'].extend(
                    f'{label}/independence: {x}' for x in
                    _ps_compare(ind, ind2, path=f'{label}/independence'))
            claim = L._cps_claim(meta,
                                 direction_ok=direction.get('consistent') is True,
                                 independence_status=ind2.get('status'),
                                 corpus_ok=corpus.get('shared') is True)
            if bool(b.get('claim_allowed')) != bool(claim.get('claim_allowed')):
                out['problems'].append(
                    f'{label}: claim_allowed {b.get("claim_allowed")!r} != '
                    f'recomputed {claim.get("claim_allowed")!r}')
            if b.get('evidence_tier') != claim.get('evidence_tier'):
                out['problems'].append(
                    f'{label}: evidence_tier {b.get("evidence_tier")!r} != '
                    f'recomputed {claim.get("evidence_tier")!r}')
            # The status is a function of heterogeneity + claim; do not let a
            # failed claim be relabelled `poolable`.
            if b.get('status') == 'poolable':
                poolable_actual.add(str(comp))
                if not claim.get('claim_allowed'):
                    out['problems'].append(
                        f'{label}: status=poolable while the claim gate refuses')
            if b.get('status') == 'poolable' and not meta.get('poolable'):
                out['problems'].append(
                    f'{label}: status=poolable while meta poolable is false')
            # Leave-one-panel-out is a stated robustness readout; replay it so
            # a deleted/edited panel cannot be hidden behind an unchanged
            # headline pool.
            loo = b.get('leave_one_panel_out')
            if isinstance(loo, dict):
                _full_mean = ((meta.get('random_effects') or {}).get('mean')
                              if meta.get('random_effects') else None)
                if _full_mean is None:
                    _full_mean = (meta.get('fixed_effect') or {}).get('mean')
                try:
                    loo2 = L._cps_leave_one_out(points, alpha=alpha,
                                                min_n=min_n,
                                                full_mean=_full_mean)
                except Exception as exc:
                    out['problems'].append(
                        f'{label}: leave-one-out replay failed '
                        f'({type(exc).__name__}: {exc})')
                else:
                    out['problems'].extend(
                        f'{label}/leave_one_panel_out: {x}' for x in
                        _ps_compare(loo, loo2,
                                    path=f'{label}/leave_one_panel_out'))
                    _same = [v.get('same_sign_as_full')
                             for v in loo2.values()]
                    _stable = (None if not _same or any(x is None for x in _same)
                               else bool(all(_same)))
                    if b.get('sign_stable_leave_one_out') != _stable:
                        out['problems'].append(
                            f'{label}: sign_stable_leave_one_out '
                            f'{b.get("sign_stable_leave_one_out")!r} != '
                            f'recomputed {_stable!r}')
            if b.get('status') == 'heterogeneous':
                _het = meta.get('heterogeneity') or {}
                _i2 = _het.get('I2_percent')
                _qp = _het.get('p')
                try:
                    _cap = float(policy.get('i2_ceiling_percent', 50.0))
                    _q_alpha = float(policy.get('q_alpha', 0.10))
                except (TypeError, ValueError, OverflowError):
                    out['problems'].append(
                        f'{label}: synthesis policy I2/Q thresholds are malformed')
                    _cap, _q_alpha = 50.0, 0.10
                _het_ok = ((_i2 is not None and float(_i2) > _cap)
                           or (_qp is not None and math.isfinite(float(_qp))
                               and float(_qp) < _q_alpha))
                if not _het_ok:
                    out['problems'].append(
                        f'{label}: status=heterogeneous but the heterogeneity '
                        'gate does not fire')
            # Selection-bias replay, when present.
            sel = b.get('verified_only_sensitivity')
            if isinstance(sel, dict):
                rows = [panels.get(str(pt.get('label'))) or {} for pt in points]
                predicted = (b.get('direction') or {}).get('predicted_signs') or []
                try:
                    sel2 = L._cps_selection_readout(
                        points, rows, alpha=alpha, min_n=min_n,
                        predicted_sign=(predicted[0] if len(predicted) == 1 else None))
                except Exception as exc:
                    out['problems'].append(
                        f'{label}: selection replay failed '
                        f'({type(exc).__name__}: {exc})')
                else:
                    for key in ('n_all', 'n_verified', 'status', 'poolable',
                                'random_effects', 'fixed_effect',
                                'benefit_all_panels',
                                'benefit_verified_only',
                                'verified_minus_all_benefit',
                                'verified_only_more_favourable'):
                        if key in sel or key in sel2:
                            out['problems'].extend(
                                f'{label}/selection: {x}' for x in
                                _ps_compare(sel.get(key), sel2.get(key),
                                            path=f'{label}/selection/{key}'))
            out['contrasts'][label] = {'status': b.get('status'),
                                       'audit': 'replayed'}
            out['n_ok'] += 1
    if poolable_declared != poolable_actual:
        out['problems'].append(
            'poolable_components does not match the contrast statuses: '
            f'declared={sorted(poolable_declared)}, '
            f'replayed={sorted(poolable_actual)}')
    if out['n_contrasts'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the synthesis block declares no contrasts'
    elif out['problems'] or out['n_fail']:
        out['status'] = 'fail'
    else:
        out['status'] = 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
