def audit_endpoint_sensitivity(manifest_path=None, pilot_path=None, *,
                               verbose=True):
    """Zero-GPU design gate: does each endpoint have measurable dynamic range?

    The gate reads the frozen/draft manifest and, when available, a validation
    pilot that records the observed control level, repeated identical-evaluation
    values, the paired SD and the unit count per endpoint.  It reports, for
    every endpoint and every primary comparison:

      * the metric bounds/direction and the beneficial headroom from the
        declared control level (or, when only both arm levels are supplied,
        the better arm);
      * whether the headroom covers the declared minimum meaningful effect
        (MEE/SESOI);
      * the measurement resolution from repeated evaluations of the identical
        checkpoint, and whether it is smaller than the MEE;
      * the planned MDE from the pilot paired SD and the frozen unit count.

    A bounded endpoint already at its beneficial bound, or with headroom below
    the declared MEE, cannot support an improvement claim regardless of the
    seed count.  The gate fails closed only on requirements the manifest itself
    declares (``require_headroom``, ``require_measurement_resolution``,
    ``require_mde_within_headroom``, ``require_declared_mee``); a missing pilot
    without such a requirement is a warning, never a pass.  It never reads
    ``wiki.test.raw`` and never changes a p-value, Holm family or frozen
    verdict.
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    if isinstance(mp, (str, os.PathLike)) and not os.path.isabs(os.fspath(mp)):
        mp = os.path.join(REPO, os.fspath(mp))
    pp = pilot_path
    if pp is None:
        _cand = os.path.join(REPO, 'analysis_v12',
                             'endpoint_sensitivity_pilot.json')
        pp = _cand if os.path.exists(_cand) else None
    elif isinstance(pp, (str, os.PathLike)) and not os.path.isabs(os.fspath(pp)):
        pp = os.path.join(REPO, os.fspath(pp))
    out = L.endpoint_sensitivity_audit(mp, pp, verbose=False)
    out['path'] = os.fspath(mp) if isinstance(mp, (str, os.PathLike)) else mp
    out['pilot'] = (os.fspath(pp) if isinstance(pp, (str, os.PathLike))
                    else pp)
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
