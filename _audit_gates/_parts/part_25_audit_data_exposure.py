# Audit the training-data exposure block carried by run records.
#
# A run records ``tokens_seen`` (draws) and, since the data-exposure change, a
# ``data_exposure`` block with the exact seeded window interval union.  This
# gate replays that block from its own input echo and refuses a hand-edited or
# inconsistent coverage claim.  It never reads a corpus or the test split,
# never trains, and never changes a PPL/p/Holm/verdict.
#
# Status semantics: ``ok`` -- at least one exposure block replayed and every
# replayed field agrees; ``fail`` -- malformed, inconsistent, or non-replaying
# block (including a ``tokens_seen`` that disagrees with its draw geometry);
# ``not_verifiable_locally`` -- no artifact with an exposure block is present;
# legacy records with only ``tokens_seen`` are warnings, not a pass.
#
# NOTE: this fragment deliberately has no module docstring.  The split loader
# executes fragments in the package globals, so a top-level string would
# overwrite audit_gates.__doc__ (the CLI help) for every later reader.

import glob as _txa_glob
import os as _txa_os


def _txa_int(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return int(value) if int(value) == value else None
    except (TypeError, ValueError, OverflowError):
        return None


def _txa_close(a, b):
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return False
    # Counts such as draws/processed_tokens can legitimately exceed float64.
    # Comparing two exact integers through float() would raise OverflowError
    # (and make a self-consistent record fail); compare those exactly.
    if (isinstance(a, (int, np.integer))
            and isinstance(b, (int, np.integer))):
        return int(a) == int(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    return (math.isfinite(fa) and math.isfinite(fb)
            and abs(fa - fb) <= 1e-9 + 1e-9 * max(abs(fa), abs(fb)))


def _txa_same(stored, replay):
    if isinstance(replay, bool) or replay is None:
        return stored is replay
    if isinstance(replay, (int, float)):
        return _txa_close(stored, replay)
    return stored == replay


def _txa_collect(obj, out, legacy, *, depth=0):
    """Collect run-like dicts without mistaking nested blocks for records."""
    if depth > 10:
        return
    if isinstance(obj, dict):
        if obj.get('kind') == 'training_data_exposure':
            out.append({'data_exposure': obj})
            return
        if isinstance(obj.get('data_exposure'), dict):
            out.append(obj)
            return
        if 'data_exposure' in obj:
            out.append(obj)
            return
        if 'tokens_seen' in obj and (
                'steps' in obj or 'steps_completed' in obj
                or 'train_knobs' in obj or 'variant' in obj):
            legacy.append(obj)
            return
        for value in obj.values():
            _txa_collect(value, out, legacy, depth=depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            _txa_collect(value, out, legacy, depth=depth + 1)


def _txa_paths(path):
    if path is None:
        cands = [_txa_os.path.join(REPO, 'analysis_v12', 'test_stats.json')]
        for name in _iter_result_dirs():
            cands.append(_txa_os.path.join(REPO, name, 'summary.json'))
        return sorted({p for p in cands if _txa_os.path.exists(p)})
    p = _audit_artifact(path)
    if _txa_os.path.isdir(p):
        cands = [_txa_os.path.join(p, 'summary.json'),
                 _txa_os.path.join(p, 'test_stats.json')]
        cands.extend(sorted(_txa_glob.glob(_txa_os.path.join(p, '*.json'))))
        return sorted({x for x in cands if _txa_os.path.exists(x)})
    return [p] if _txa_os.path.exists(p) else []


def audit_data_exposure(path=None):
    """Replay stored training-data exposure blocks (zero GPU, no data read)."""
    L = _import_exp_lib()
    out = {'kind': 'data_exposure_audit', 'status': 'ok', 'path': None,
           'paths': [], 'n_records': 0, 'n_replayed': 0, 'n_legacy': 0,
           'n_not_estimable': 0, 'n_ok': 0, 'records': {},
           'problems': [], 'warnings': [],
           'note': ('replays the same seeded window-start stream as batch_iter '
                    'and recomputes the union of prediction-target intervals; '
                    'a missing block is not a pass and a disagreement is a '
                    'fail; no frozen p/Holm/verdict is changed')}
    paths = _txa_paths(path)
    out['path'] = path if path is not None else None
    out['paths'] = list(paths)
    if not paths:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no run/summary artifact with a data_exposure block '
                         'was found locally')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    records = []
    legacy = []
    for p in paths:
        data = _load(p)
        if data is None:
            out['problems'].append(f'{p}: exists but is not a readable JSON object')
            continue
        _txa_collect(data, records, legacy)
    out['n_records'] = len(records)
    out['n_legacy'] = len(legacy)
    if legacy:
        out['warnings'].append(
            f'{len(legacy)} run-like record(s) carry tokens_seen but no '
            'data_exposure block (legacy artifact); their distinct-data '
            'exposure is not replayable')
    if not records:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the artifact(s) contain no data_exposure block; '
                         'tokens_seen alone is a draw count, not a coverage '
                         'measurement')
        if out['problems']:
            out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    confirm_fields = ('confirmed', 'p_holm', 'verified', 'verdict', 'decision',
                      'p_exact_signflip', 'significant')
    required = (
        'draws', 'processed_tokens', 'window_positions', 'target_positions',
        'unique_starts', 'unique_tokens', 'start_coverage', 'token_coverage',
        'unique_epochs', 'effective_epochs', 'reuse_factor',
        'duplicate_token_fraction', 'start_collision_fraction',
        'max_start_multiplicity', 'exact', 'coverage_quality')
    input_keys = ('n_tokens', 'seq_len', 'batch_size', 'steps', 'seed',
                  'world_size', 'data_seeds', 'max_exact_draws')
    for i, rec in enumerate(records):
        if rec.get('key'):
            name = rec.get('key')
        elif rec.get('seed') is not None:
            name = f"{rec.get('variant') or 'record'}::s{rec.get('seed')}"
        else:
            name = f'record[{i}]'
        row = {'status': 'fail', 'problems': [], 'warnings': []}
        out['records'][str(name)] = row
        block = rec.get('data_exposure')
        if not isinstance(block, dict):
            row['problems'].append(
                f'data_exposure is a {type(block).__name__}, not an object')
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if any(f in block for f in confirm_fields):
            row['problems'].append(
                'data_exposure carries confirmatory field(s) '
                f'{sorted(set(block) & set(confirm_fields))}')
        if block.get('kind') != 'training_data_exposure':
            row['problems'].append(
                f"kind is {block.get('kind')!r}, not 'training_data_exposure'")
        if block.get('schema') != L.TRAINING_EXPOSURE_POLICY['schema']:
            row['problems'].append(
                f"schema is {block.get('schema')!r}, not "
                f"{L.TRAINING_EXPOSURE_POLICY['schema']!r}")
        status = block.get('status')
        if status not in L.TRAINING_EXPOSURE_STATUSES:
            row['problems'].append(f'status {status!r} is not a declared status')
        if status == 'not_estimable':
            if row['problems']:
                out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            else:
                row['status'] = 'not_applicable'
                out['n_not_estimable'] += 1
                out['warnings'].append(
                    f'{name}: data_exposure is not_estimable '
                    f'({block.get("reason") or "no reason recorded"})')
            continue
        if row['problems']:
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        inputs = block.get('inputs')
        if not isinstance(inputs, dict):
            row['problems'].append('inputs is not an object')
        else:
            missing = [k for k in input_keys if k not in inputs]
            if missing:
                row['problems'].append(f'inputs missing {missing}')
            for k in ('n_tokens', 'seq_len', 'batch_size', 'steps', 'seed',
                      'world_size', 'max_exact_draws'):
                if k in inputs and _txa_int(inputs.get(k)) is None:
                    row['problems'].append(f'inputs.{k} is not an exact integer')
            seeds = inputs.get('data_seeds')
            world = _txa_int(inputs.get('world_size'))
            if not isinstance(seeds, list):
                row['problems'].append('inputs.data_seeds is not a list')
            elif world is not None and len(seeds) != world:
                row['problems'].append(
                    f'inputs.data_seeds has {len(seeds)} entries for '
                    f'world_size={world}')
            else:
                for v in seeds:
                    if _txa_int(v) is None or _txa_int(v) < 0:
                        row['problems'].append(
                            f'inputs.data_seeds contains a non-integer/negative '
                            f'value {v!r}')
                        break
            max_exact = _txa_int(inputs.get('max_exact_draws'))
            # Any non-negative threshold is replayable: the core helper
            # clamps a caller-supplied threshold at its own absolute hard cap
            # before allocating the start array.
            if max_exact is not None and max_exact < 0:
                row['problems'].append(
                    'inputs.max_exact_draws must be non-negative')
        if row['problems']:
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        missing = [k for k in required if k not in block]
        if missing:
            row['problems'].append(f'block missing {missing}')
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        try:
            replay = L.training_window_exposure(
                inputs['n_tokens'], inputs['seq_len'], inputs['batch_size'],
                inputs['steps'], seed=inputs['seed'],
                world_size=inputs['world_size'],
                data_seeds=inputs['data_seeds'],
                max_exact_draws=inputs['max_exact_draws'])
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            row['problems'].append(
                f'replay raised {type(exc).__name__}: {exc}')
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        for key in required:
            if not _txa_same(block.get(key), replay.get(key)):
                row['problems'].append(
                    f'{key}: stored {block.get(key)!r} != replayed '
                    f'{replay.get(key)!r}')
        for key in input_keys:
            if key in replay.get('inputs', {}):
                if not _txa_same(inputs.get(key), replay['inputs'].get(key)):
                    row['problems'].append(
                        f'inputs.{key}: stored {inputs.get(key)!r} != '
                        f'replayed {replay["inputs"].get(key)!r}')
        # Cross-checks against the surrounding run record.  These catch the
        # common hand edit where the block is self-consistent but detached
        # from the run's advertised token budget.
        tokens_seen = _txa_int(rec.get('tokens_seen'))
        if tokens_seen is not None and tokens_seen != replay['processed_tokens']:
            row['problems'].append(
                f'record tokens_seen={tokens_seen} != exposure '
                f'processed_tokens={replay["processed_tokens"]}')
        if rec.get('steps_completed') is not None:
            stored_steps = _txa_int(rec.get('steps_completed'))
            if stored_steps is not None and stored_steps != inputs['steps']:
                row['problems'].append(
                    f'record steps_completed={stored_steps} != exposure '
                    f'inputs.steps={inputs["steps"]}')
        if rec.get('distributed_world_size') is not None:
            stored_world = _txa_int(rec.get('distributed_world_size'))
            if stored_world is not None and stored_world != inputs['world_size']:
                row['problems'].append(
                    f'record distributed_world_size={stored_world} != exposure '
                    f'inputs.world_size={inputs["world_size"]}')
        if rec.get('seed') is not None:
            stored_run_seed = _txa_int(rec.get('seed'))
            if (stored_run_seed is not None
                    and stored_run_seed != inputs['seed']):
                row['problems'].append(
                    f'record seed={stored_run_seed} != exposure '
                    f'inputs.seed={inputs["seed"]}')
        if rec.get('data_seed') is not None:
            stored_seed = _txa_int(rec.get('data_seed'))
            if (stored_seed is not None and inputs['data_seeds']
                    and stored_seed != inputs['data_seeds'][0]):
                row['problems'].append(
                    f'record data_seed={stored_seed} != rank-0 exposure seed '
                    f'{inputs["data_seeds"][0]}')
        if row['problems']:
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
        else:
            row['status'] = 'ok'
            out['n_replayed'] += 1
            out['n_ok'] += 1

    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no data_exposure block replayed successfully'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
