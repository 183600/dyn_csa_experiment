# Audit a panel's pre-registered run disposition from its attempt ledger.
# This fragment deliberately carries no module docstring: the audit CLI help
# text lives in part_00 and must remain the package __doc__.
#
# The training drivers build a ``run_disposition.json`` plan before the first
# cell and log every attempt to ``experiments.jsonl``.  This gate joins the two:
# it replays which ``(arm, seed)`` cells were attempted, which attempts failed
# and why, which successful attempt was retained, and whether the retained
# measurement is the one the plan declared (by default the *first* success).
#
# It answers a question the existing gates do not: a failed cell can be retried,
# and "retry until a number appears" is a selection mechanism.  The gate does
# not recompute a PPL or a p-value; it checks the paper trail:
#
# * a retained measurement for a cell that is not in the plan (seed/arm shopping);
# * a later successful attempt retained over the first one (post-hoc selection);
# * a successful attempt under a different ``run_cfg`` (a code/config change after
#   a measurement);
# * a failure class the policy calls non-replaceable that was retried anyway;
# * an unknown failure that was retried (the policy cannot classify it);
# * more attempts than the plan allows, or more than the declared total;
# * retries concentrated in a subset of arms (descriptive, not a test).
#
# Missing plan / missing events is ``not_verifiable_locally``, never a pass.  A
# hash-chain failure in the ledger is a hard ``fail``: the attempt history itself
# is not trustworthy.
import json
import os


def _dispo_panel_paths(paths):
    if paths is None:
        found = []
        for name in sorted(os.listdir(REPO)):
            p = os.path.join(REPO, name)
            if not os.path.isdir(p):
                continue
            if name.startswith('results_'):
                found.append(p)
            elif name.startswith('analysis_') and (
                    os.path.isfile(os.path.join(p, 'experiments.jsonl'))
                    or os.path.isfile(os.path.join(p, 'run_disposition.json'))):
                found.append(p)
        return found
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    out, seen = [], set()
    for raw in paths:
        p = os.path.abspath(_audit_artifact(raw))
        if os.path.isdir(p):
            d = p
        else:
            d = os.path.dirname(p)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _dispo_read_events(path):
    events, malformed = [], []
    if not os.path.isfile(path):
        return events, malformed
    try:
        with open(path, encoding='utf-8') as fh:
            for lineno, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except Exception as e:
                    malformed.append(f'line {lineno}: {type(e).__name__}: {e}')
                    continue
                if isinstance(rec, dict):
                    events.append(rec)
                else:
                    malformed.append(f'line {lineno}: not a JSON object')
    except OSError as e:
        malformed.append(f'cannot read ledger: {type(e).__name__}: {e}')
    return events, malformed


def _dispo_retained_records(outdir, L):
    """Measured (arm, seed) records that actually entered the analysis."""
    retained, source = {}, None
    summary = _load(os.path.join(outdir, 'summary.json'))
    if isinstance(summary, dict):
        source = 'summary.json'
        for key, rec in summary.items():
            if not isinstance(rec, dict):
                continue
            if rec.get('synthesized'):
                continue
            try:
                usable = bool(L.ppl_is_usable(rec.get('ppl')))
            except Exception:
                usable = False
            if not usable:
                continue
            retained[str(key)] = {'run_id': rec.get('run_id'),
                                  'run_cfg': rec.get('run_cfg'),
                                  'ppl': rec.get('ppl')}
    if retained:
        return retained, source
    manifest = _load(os.path.join(outdir, 'run_manifest.json'))
    if isinstance(manifest, dict) and isinstance(manifest.get('panels'), dict):
        source = 'run_manifest.json'
        for _cfg, panel in manifest['panels'].items():
            if not isinstance(panel, dict):
                continue
            cells = panel.get('cells')
            if not isinstance(cells, dict):
                continue
            for key, cell in cells.items():
                if not isinstance(cell, dict):
                    continue
                if not L.ppl_is_usable(cell.get('ppl')):
                    continue
                retained[str(key)] = {'run_id': cell.get('run_id'),
                                      'run_cfg': cell.get('run_cfg'),
                                      'ppl': cell.get('ppl')}
    return retained, source


def audit_run_disposition(paths=None, *more):
    """Zero-GPU audit of the attempt/disposition paper trail.

    ``paths`` may be panel directories, ``experiments.jsonl`` files or
    ``run_disposition.json`` files; with no argument every ``results_*`` (and
    analysis panel with an attempt ledger) is scanned.  The gate reads local
    files only: it starts no training, reads no test split and changes no
    measurement, p-value, Holm family or frozen verdict.
    """
    L = _import_exp_lib()
    if more:
        paths = [paths] + list(more) if paths is not None else list(more)
    out = {
        'kind': 'run_disposition_report',
        'status': 'ok',
        'panels': {},
        'n_panels': 0,
        'n_ok': 0,
        'n_warn': 0,
        'n_fail': 0,
        'n_not_verifiable': 0,
        'problems': [],
        'warnings': [],
        'note': ('pre-registered disposition: the retained measurement must be '
                 'the declared attempt, retries must be policy-allowed and not '
                 'concentrated in one arm; this gate changes no measurement or '
                 'frozen decision'),
    }
    for outdir in _dispo_panel_paths(paths):
        ledger = os.path.join(outdir, 'experiments.jsonl')
        plan_path = os.path.join(outdir, 'run_disposition.json')
        if not os.path.exists(ledger) and not os.path.exists(plan_path):
            continue
        rel = (os.path.relpath(outdir, REPO)
               if os.path.abspath(outdir).startswith(REPO + os.sep) else outdir)
        events, malformed = _dispo_read_events(ledger)
        plan = _load(plan_path)
        plan_unreadable = os.path.exists(plan_path) and not isinstance(plan, dict)
        if plan is None and not any(isinstance(e, dict) and e.get('kind') == 'cell'
                                    for e in events):
            # An analysis/audit ledger holds no training cells and this panel
            # registered no disposition plan; there is nothing to audit.
            continue
        retained, retained_source = _dispo_retained_records(outdir, L)
        ledger_rep = {}
        if os.path.exists(ledger):
            try:
                ledger_rep = L.verify_experiment_ledger(ledger)
            except Exception as e:
                ledger_rep = {'status': 'fail',
                              'problems': [f'ledger verifier raised '
                                           f'{type(e).__name__}: {e}']}
        ledger_status = ledger_rep.get('status')
        panel = {
            'outdir': rel,
            'ledger': ledger_status or ('missing' if not os.path.exists(ledger) else None),
            'ledger_problems': list(ledger_rep.get('problems') or [])[:20],
            'ledger_warnings': list(ledger_rep.get('warnings') or [])[:20],
            'plan_present': bool(isinstance(plan, dict)),
            'plan_unreadable': bool(plan_unreadable),
            'retained_source': retained_source,
            'n_retained': len(retained),
            'n_malformed_ledger_lines': len(malformed),
        }
        if malformed:
            panel['malformed_ledger_lines'] = malformed[:20]
        if plan_unreadable:
            panel['status'] = 'fail'
            panel['problems'] = ['run_disposition.json exists but is not a '
                                 'readable JSON object; a precommitment that '
                                 'cannot be read is not a missing optional '
                                 'artifact']
            out['problems'].extend(f'{rel}: {x}' for x in panel['problems'])
            out['n_fail'] += 1
            out['panels'][rel] = panel
            out['n_panels'] += 1
            continue
        if ledger_status == 'fail':
            panel['status'] = 'fail'
            panel['problems'] = ['the attempt ledger failed its hash-chain '
                                 'check: the attempt history is not trustworthy '
                                 'enough to certify a disposition']
            out['problems'].extend(f'{rel}: {x}' for x in panel['problems'])
            out['n_fail'] += 1
            out['panels'][rel] = panel
            out['n_panels'] += 1
            continue
        if ledger_status in ('warnings', 'not_verifiable_locally'):
            panel.setdefault('warnings', []).append(
                f'ledger status is {ledger_status!r}; the attempt order is not '
                'independently anchored (legacy or missing tail anchor)')
        if malformed:
            panel['status'] = 'fail'
            panel['problems'] = [f'{len(malformed)} ledger line(s) are not '
                                 'valid JSON objects; attempts may be hidden']
            out['problems'].extend(f'{rel}: {x}' for x in panel['problems'])
            out['n_fail'] += 1
            out['panels'][rel] = panel
            out['n_panels'] += 1
            continue
        try:
            rep = L.run_disposition_audit(
                events, plan=plan if isinstance(plan, dict) else None,
                retained=retained if retained else None)
        except Exception as e:
            panel['status'] = 'fail'
            panel['problems'] = [f'disposition audit could not run: '
                                 f'{type(e).__name__}: {e}']
            out['problems'].extend(f'{rel}: {x}' for x in panel['problems'])
            out['n_fail'] += 1
            out['panels'][rel] = panel
            out['n_panels'] += 1
            continue
        panel['audit'] = rep
        panel['status'] = rep.get('status')
        if rep.get('problems'):
            out['problems'].extend(f'{rel}: {x}' for x in rep['problems'])
        if rep.get('warnings'):
            out['warnings'].extend(f'{rel}: {x}' for x in rep['warnings'])
        st = panel['status']
        if st == 'fail':
            out['n_fail'] += 1
        elif st in ('warn', 'warnings'):
            out['n_warn'] += 1
        elif st == 'ok':
            out['n_ok'] += 1
        else:
            out['n_not_verifiable'] += 1
        out['panels'][rel] = panel
        out['n_panels'] += 1
    if out['n_fail']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0 and out['n_warn'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no panel has both a run_disposition.json plan and a '
                         'usable attempt ledger; run a panel with the current '
                         'code (or pass a panel directory explicitly)')
    elif out['n_warn'] or out['n_not_verifiable']:
        out['status'] = 'warn'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
