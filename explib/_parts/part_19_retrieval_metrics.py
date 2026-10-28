

def retrieval_metrics(y_true, y_pred, labels=None, label_names=None):
    """Top-1 retrieval metrics with explicit trivial floors and macro recall.

    A many-way retrieval task (e.g. NIAH over an 8k vocabulary) is a
    multiclass endpoint.  Top-1 micro accuracy is dominated by frequent answer
    classes and is not interpretable without the score a trivial predictor
    obtains.  This helper therefore reports, on the same observations:

    * ``n_correct``/``acc`` — the raw top-1 count and rate;
    * ``balanced_accuracy`` (macro recall) — every answer class counts equally,
      so a rare-answer failure cannot hide behind frequent classes;
    * named ``f1_macro``/``f1_micro``/``f1_weighted`` — a single-label top-1
      endpoint has per-class TP/FP/FN, so all three averages are computable;
      micro F1 equals accuracy, macro weighs observed answers equally and
      weighted follows support.  PR-AUC/ROC-AUC remain unavailable because a
      top-1 record carries no per-class scores;
    * ``chance_accuracy`` — uniform guessing over the OBSERVED answer classes
      (``1/K``), which is the correct chance floor for this label space (not
      ``1/vocab`` when only K classes actually occur);
    * ``majority_accuracy`` — the always-most-frequent-answer predictor.

    The lift over each floor is recorded so a report cannot present a
    chance-level retrieval score as skill.  Per-class support/recall is kept
    for the audit to recompute the aggregates.  Labels are compared by exact
    identity (mixed Python types stay distinct), and an absent declared class
    is listed with a null recall instead of being scored as zero.
    """
    y = _preserve_label_array(y_true)
    p = _preserve_label_array(y_pred)
    if y.size != p.size:
        raise ValueError(f'retrieval_metrics: y_true has {y.size} entries but y_pred has {p.size}')
    if labels is None:
        labs = _unique_label_values(y) if y.size else np.asarray([], dtype=object)
    else:
        labs = _preserve_label_array(labels)
    if label_names is None:
        label_names = _display_label_names(labs)
    label_names = list(label_names)
    if len(label_names) != int(labs.size) or len(set(label_names)) != len(label_names):
        raise ValueError('retrieval_metrics: label_names must provide one unique name per label')
    _lab_keys = [_hashable_label(v) for v in labs.tolist()]
    if len(set(_lab_keys)) != len(_lab_keys):
        # A repeated declared label would be scored twice: every query would
        # contribute to two per-class rows (inflating n_classes, the uniform
        # chance floor and macro recall).  Labels are class identities, so a
        # duplicate declaration is malformed evidence, not a second class.
        raise ValueError(f'retrieval_metrics: labels contains duplicate class '
                         f'identifiers: {labs.tolist()}')
    true_keys = [_hashable_label(v) for v in y.tolist()]
    pred_keys = [_hashable_label(v) for v in p.tolist()]
    _undeclared = sorted(set(true_keys) - set(_lab_keys), key=repr)
    if _undeclared:
        # Accuracy counts every true answer, but per-class support, the
        # observed-class count and the chance floor only cover the declared
        # labels.  Silently omitting a true class makes macro recall and the
        # floor describe a different label space than the accuracy.
        raise ValueError(
            f'retrieval_metrics: observed answer label(s) '
            f'{[repr(v) for v in _undeclared]} are not in the declared labels '
            f'{labs.tolist()}; include them or pass labels=None')
    support = {}
    correct = {}
    predicted = {}
    for _t, _p in zip(true_keys, pred_keys):
        support[_t] = support.get(_t, 0) + 1
        predicted[_p] = predicted.get(_p, 0) + 1
        if _t == _p:
            correct[_t] = correct.get(_t, 0) + 1
    n = int(y.size)
    n_correct = int(sum(correct.values()))
    acc = (n_correct / n) if n else None
    per_class = {}
    defined = 0
    recall_sum = 0.0
    majority_support = 0
    majority_name = None
    _f1_defined = []
    _f1_supports = []
    for _lab, _name in zip(labs.tolist(), label_names):
        _key = _hashable_label(_lab)
        _sup = int(support.get(_key, 0))
        _pred = int(predicted.get(_key, 0))
        _cor = int(correct.get(_key, 0))
        _rec = (_cor / _sup) if _sup else None
        # Precision is defined only when the class was predicted.  A single-
        # label multiclass endpoint has per-class TP/FP/FN, so macro/micro/
        # weighted F1 are computable from top-1 labels alone; they expose
        # false-positive behaviour that recall cannot.
        _prec = (_cor / _pred) if _pred else 0.0
        _f1 = ((2.0 * _prec * _rec / (_prec + _rec))
               if (_rec is not None and (_prec + _rec) > 0) else 0.0)
        if not _sup:
            _prec = _prec if _pred else None
            _f1 = _f1 if _pred else None
        per_class[_name] = {'support': _sup, 'predicted': _pred,
                            'recall': _rec, 'correct': _cor,
                            'precision': _prec, 'f1': _f1}
        if _sup:
            defined += 1
            recall_sum += float(_rec)
            _f1_defined.append(float(_f1))
            _f1_supports.append(_sup)
        if _sup > majority_support:
            majority_support = _sup
            majority_name = _name
    chance = (1.0 / defined) if defined else None
    majority = (majority_support / n) if n else None
    balanced = (recall_sum / defined) if defined else None
    # F1 aggregation is stated explicitly because the three averages answer
    # different questions under answer-class imbalance.  For a single-label
    # top-1 endpoint micro F1 equals accuracy by construction; macro weights
    # every observed answer class equally; weighted follows its support.
    _f1_macro = (float(np.mean(_f1_defined)) if _f1_defined else None)
    _f1_weighted = (float(np.average(_f1_defined, weights=_f1_supports))
                    if _f1_defined else None)
    _tp_micro = int(n_correct)
    _fp_micro = int(n - n_correct)
    _fn_micro = int(n - n_correct)
    _f1_micro = (2.0 * _tp_micro / (2 * _tp_micro + _fp_micro + _fn_micro)
                 if n else None)
    _n_undeclared_pred = int(sum(v for k, v in predicted.items()
                                 if k not in set(_lab_keys)))
    # Extreme answer-class imbalance: top-1 micro accuracy can be high while a
    # rare answer class is never retrieved.  Macro recall is the recall-aligned
    # conditional endpoint, not an invented PR-AUC from top-1 labels.
    _observed_prev = [(int(row['support']) / n) for row in per_class.values()
                      if int(row['support']) > 0 and n]
    _minority_prev = min(_observed_prev) if _observed_prev else None
    _imbalance_ratio = ((1.0 - _minority_prev) / _minority_prev
                        if _minority_prev is not None and _minority_prev > 0 else None)
    _rare = bool(_extreme_imbalance(_minority_prev))
    _recommended = 'macro_recall' if _rare else 'top1_accuracy'
    return {
        'n': n, 'n_correct': n_correct,
        'minority_prevalence': _minority_prev,
        'imbalance_ratio': _imbalance_ratio,
        'imbalance_warning': (
            'an answer class is at or beyond 1:1000 — report macro recall '
            '(balanced accuracy) as the conditional primary; top-1 accuracy '
            'is dominated by frequent answers' if _rare else None),
        'recommended_primary_metric': _recommended,
        'recommended_primary_reason': (
            ('minority answer prevalence %.3e is at or below 1e-3 (1:1000); '
             'macro recall counts every answer class equally' % _minority_prev) if _rare else
            ('minority answer prevalence %s is not extreme; top-1 accuracy is '
             'the policy primary' % ('n/a' if _minority_prev is None
                                     else '%.3e' % _minority_prev))),
        'recommended_primary_available': bool(
            balanced is not None if _rare else acc is not None),
        'acc': acc, 'top1_accuracy': acc,
        'balanced_accuracy': balanced, 'macro_recall': balanced,
        'n_classes': defined,
        # Three named multiclass F1 averages.  Micro F1 equals accuracy for a
        # single-label endpoint; macro treats each observed answer equally;
        # weighted follows answer support.  They can rank models differently.
        'f1_macro': _f1_macro,
        'f1_micro': _f1_micro,
        'f1_weighted': _f1_weighted,
        'f1_averaging': ('macro = observed-answer-unweighted; micro = pooled '
                         'TP/FP/FN (equals top-1 accuracy on this single-label '
                         'endpoint); weighted = support-weighted'),
        'n_undeclared_predictions': _n_undeclared_pred,
        'answer_support': {name: row['support'] for name, row in per_class.items()},
        'per_class_recall': {name: row['recall'] for name, row in per_class.items()},
        'per_class': per_class,
        'chance_accuracy': chance,
        'majority_accuracy': majority,
        'majority_label': majority_name,
        'acc_lift_over_chance': (acc - chance if acc is not None and chance is not None else None),
        'acc_lift_over_majority': (acc - majority if acc is not None and majority is not None else None),
        'averaging': ('macro_recall_over_observed_answer_classes; top1 is micro over queries; '
                      'F1 is reported separately as macro/micro/weighted'),
        'note': ('chance_accuracy is uniform guessing over the observed answer classes; '
                 'majority_accuracy is the always-most-frequent-answer floor. A top-1 '
                 'score without these floors is not evidence of retrieval skill. '
                 'PR-AUC/ROC-AUC are not reported because a top-1 record carries no '
                 'per-class scores; inventing them from hard labels would be invalid.'),
    }


def _multiclass_ranking_metrics(y_true, y_score, labels, label_names=None):
    """One-vs-rest PR-AUC/ROC-AUC for a 3+ class endpoint.

    Each class is scored as a binary problem against its own score column.
    ``macro`` is the unweighted mean over classes whose one-vs-rest endpoint is
    defined (the class occurs and is not the entire sample); ``weighted`` uses
    the class support; ``micro`` pools every one-vs-rest decision (the standard
    flattened-label definition, which is dominated by frequent classes).  The
    three can rank models differently under imbalance, so the caller must name
    which one it reports.  Undefined classes are recorded explicitly instead of
    being silently assigned 0 or 1.
    """
    y_true = _preserve_label_array(y_true)
    score = np.asarray(y_score, dtype=float)
    labels = _preserve_label_array(labels)
    if label_names is None:
        label_names = _display_label_names(labels)
    label_names = list(label_names)
    if len(label_names) != int(labels.size) or len(set(label_names)) != len(label_names):
        raise ValueError('label_names must provide one unique display name per label')
    if score.ndim != 2 or score.shape[1] != labels.size:
        raise ValueError(
            f'multiclass ranking needs a column-per-label score matrix, got {score.shape} '
            f'for {int(labels.size)} labels')
    if score.shape[0] != y_true.size:
        raise ValueError(f'y_score {score.shape} does not match y_true {y_true.shape}')
    per_class, defined, supports, weights = {}, [], [], []
    _y_onehot = np.zeros((y_true.size, labels.size), dtype=np.int64)
    for i, lab in enumerate(labels):
        y_bin = (y_true == lab).astype(np.int64)
        _y_onehot[:, i] = y_bin
        n_pos = int(y_bin.sum())
        n_neg = int(y_bin.size - n_pos)
        auc = _binary_roc_auc(y_bin, score[:, i]) if n_pos and n_neg else None
        ap = _binary_average_precision(y_bin, score[:, i]) if n_pos and n_neg else None
        _name = label_names[i]
        _prev = float(n_pos) / max(int(y_bin.size), 1)
        # A raw PR-AUC is not interpretable without its one-vs-rest chance
        # floor (the class prevalence).  A frequent class can carry a high
        # raw PR-AUC while a rare class is ranked at chance; per-class lift
        # keeps that difference visible.
        per_class[_name] = {
            'roc_auc': auc, 'pr_auc': ap, 'support': n_pos,
            'prevalence': _prev,
            'pr_auc_chance': _prev,
            'pr_auc_lift_over_chance': (float(ap - _prev)
                                        if ap is not None else None),
            'roc_auc_chance': 0.5,
            'roc_auc_lift_over_chance': (float(auc - 0.5)
                                         if auc is not None else None),
        }
        if auc is not None and ap is not None:
            defined.append(_name)
            supports.append(n_pos)
            weights.append(float(n_pos))
    out = {'per_class': per_class, 'defined_classes': defined,
           'undefined_classes': [n for n in label_names if n not in set(defined)],
           'roc_auc_macro': None, 'roc_auc_weighted': None,
           'pr_auc_macro': None, 'pr_auc_weighted': None,
           'roc_auc_micro': None, 'pr_auc_micro': None,
           'roc_auc_chance': 0.5,
           'pr_auc_macro_chance': None, 'pr_auc_macro_lift_over_chance': None,
           'pr_auc_micro_chance': None, 'pr_auc_micro_lift_over_chance': None,
           'pr_auc_weighted_chance': None, 'pr_auc_weighted_lift_over_chance': None,
           'averaging': 'one-vs-rest; macro/weighted over defined classes, micro over pooled decisions',
           'note': ('macro treats every class equally (sensitive to a rare class); micro is '
                    'dominated by frequent classes; weighted follows the support. '
                    'Undefined one-vs-rest classes are listed, not scored. PR-AUC chance '
                    'is the one-vs-rest prevalence floor, not 0.')}
    if defined:
        out['roc_auc_macro'] = float(np.mean([per_class[c]['roc_auc'] for c in defined]))
        out['pr_auc_macro'] = float(np.mean([per_class[c]['pr_auc'] for c in defined]))
        _prev_defined = [per_class[c]['support'] / max(int(y_true.size), 1) for c in defined]
        out['pr_auc_macro_chance'] = float(np.mean(_prev_defined))
        out['pr_auc_macro_lift_over_chance'] = float(
            out['pr_auc_macro'] - out['pr_auc_macro_chance'])
        if sum(weights) > 0:
            out['roc_auc_weighted'] = float(np.average(
                [per_class[c]['roc_auc'] for c in defined], weights=weights))
            out['pr_auc_weighted'] = float(np.average(
                [per_class[c]['pr_auc'] for c in defined], weights=weights))
            # Support-weighted mean of the per-class chance prevalences.
            out['pr_auc_weighted_chance'] = float(
                sum(w * p for w, p in zip(weights, _prev_defined)) / sum(weights))
            out['pr_auc_weighted_lift_over_chance'] = float(
                out['pr_auc_weighted'] - out['pr_auc_weighted_chance'])
    _flat_y = _y_onehot.reshape(-1)
    _flat_s = score.reshape(-1)
    if _flat_y.size and _flat_y.min() == 0 and _flat_y.max() == 1:
        out['roc_auc_micro'] = _binary_roc_auc(_flat_y, _flat_s)
        out['pr_auc_micro'] = _binary_average_precision(_flat_y, _flat_s)
        # The flattened one-vs-rest pool has exactly one positive per sample,
        # over K*N decisions, so its PR-AUC chance floor is 1/K — not the
        # per-class prevalence and not 0.  Without this floor a raw micro
        # PR-AUC looks skilful merely because K is small.
        _micro_chance = 1.0 / float(labels.size)
        out['pr_auc_micro_chance'] = _micro_chance
        if out['pr_auc_micro'] is not None:
            out['pr_auc_micro_lift_over_chance'] = float(
                out['pr_auc_micro'] - _micro_chance)
    return out


def _majority_class_floor(labels, supports, label_names=None):
    """Metrics of the always-majority predictor (a trivial classification floor).

    A classifier's F1/accuracy is not interpretable without the score its
    labels alone can achieve.  Under imbalance that floor is often deceptively
    high, so it is recorded next to every classification summary rather than
    left implicit.  Ties are resolved by the caller's label order (the first
    label with maximal support), which keeps the record deterministic.
    """
    supports = [int(s) for s in supports]
    n = int(sum(supports))
    labels = list(labels)
    if label_names is None:
        label_names = [str(v) for v in labels]
    label_names = list(label_names)
    if len(label_names) != len(labels):
        raise ValueError('label_names must provide one display name per label')
    if n <= 0 or not labels:
        return {'strategy': 'most_frequent', 'label': None, 'accuracy': None,
                'balanced_accuracy': None, 'f1_macro': None, 'f1_micro': None,
                'f1_weighted': None,
                'note': 'no observations; the trivial floor is undefined'}
    best_i = int(np.argmax(np.asarray(supports, dtype=np.int64)))
    s = supports[best_i]
    p = s / n
    f1_major = 2.0 * p / (1.0 + p) if p > 0 else 0.0
    return {
        'strategy': 'most_frequent',
        'label': str(label_names[best_i]),
        'support': s,
        'accuracy': p,
        'balanced_accuracy': 1.0 / len(labels),
        'f1_macro': f1_major / len(labels),
        # Micro F1 pools TP/FP/FN over all classes and therefore equals
        # accuracy for a single-label multiclass problem; the majority
        # class's own binary F1 is f1_major, not the micro average.
        'f1_micro': p,
        'f1_weighted': f1_major * p,
        'note': ('always predicts the most frequent class (label order breaks ties); '
                 'a real model must beat this floor, and under imbalance this floor '
                 'can have high accuracy with zero minority recall'),
    }

_BINARY_THRESHOLD_METRICS = ('f1', 'balanced_accuracy', 'youden_j')
_BINARY_THRESHOLD_SOURCES = ('default_0.5', 'caller_threshold', 'caller_predictions',
                             'argmax', 'validation_selection', 'unrecorded')


def _binary_positive_vector(y_true, positive_label=None):
    y = _preserve_label_array(y_true)
    uniq = _unique_label_values(y)
    if uniq.size != 2:
        raise ValueError(f'binary threshold analysis needs exactly two observed classes, got {uniq.tolist()}')
    if positive_label is None:
        pos = uniq[-1]
    else:
        hits = np.where(uniq == positive_label)[0]
        if hits.size == 0:
            raise ValueError(f'positive_label={positive_label!r} is not in the observed labels {uniq.tolist()}')
        pos = uniq[int(hits[0])]
    return (y == pos).astype(np.int64), pos, uniq


def _threshold_row_from_counts(tp, fp, fn, tn, threshold):
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {'threshold': float(threshold), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'specificity': specificity,
            'balanced_accuracy': 0.5 * (recall + specificity),
            'youden_j': recall + specificity - 1.0,
            'predicted_positive': int(tp + fp),
            'predicted_positive_rate': float((tp + fp) / max(tp + fp + fn + tn, 1))}


def _binary_threshold_metrics(y_bin, score, threshold):
    pred = np.asarray(score) >= float(threshold)
    return _threshold_row_from_counts(
        np.sum(pred & (y_bin == 1)), np.sum(pred & (y_bin == 0)),
        np.sum((~pred) & (y_bin == 1)), np.sum((~pred) & (y_bin == 0)),
        threshold)


def _binary_calibration(y_bin, score, n_bins=10):
    score = np.asarray(score, dtype=float).ravel()
    _bins = _as_seed_int(n_bins)
    if _bins is None or _bins < 1:
        raise ValueError(f'n_bins must be a positive integer, got {n_bins!r}')
    if score.size and (not np.all(np.isfinite(score))
                       or np.any(score < 0.0) or np.any(score > 1.0)):
        return {'status': 'unavailable',
                'reason': 'scores are not finite probabilities in [0, 1]; Brier/ECE require probabilities'}
    if score.size == 0:
        return {'status': 'unavailable', 'reason': 'no scores'}
    idx = np.clip((score * _bins).astype(np.int64), 0, _bins - 1)
    rows = []
    ece = 0.0
    for b in range(_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        confidence = float(score[mask].mean())
        accuracy = float(y_bin[mask].mean())
        weight = n_b / score.size
        ece += weight * abs(accuracy - confidence)
        rows.append({'bin': b, 'n': n_b, 'confidence': confidence,
                     'accuracy': accuracy, 'gap': accuracy - confidence})
    return {'status': 'ok', 'brier_score': float(np.mean((score - y_bin) ** 2)),
            'ece': float(ece), 'n_bins': _bins, 'reliability': rows,
            'note': 'ECE uses equal-width probability bins; fewer than n_bins non-empty bins are possible'}


def _multiclass_calibration(y_true, y_score, labels, n_bins=10):
    """Brier score and top-label ECE for a multiclass probability matrix.

    Calibration asks whether a predicted probability is usable, not merely
    whether the ranking is good.  The input must be a row-normalized
    probability matrix; logits return ``status="unavailable"`` instead of an
    invented probability.  Brier is the multiclass sum-of-squares, while ECE
    uses the top-1 confidence (the standard top-label reliability diagnostic).
    """
    y_true = _preserve_label_array(y_true)
    p = np.asarray(y_score, dtype=float)
    labels = _preserve_label_array(labels)
    if p.ndim != 2 or p.shape[1] != labels.size or p.shape[0] != y_true.size:
        return {'status': 'unavailable', 'reason': 'expected a rows x classes score matrix'}
    if p.size and (np.any(p < 0.0) or np.any(p > 1.0)):
        return {'status': 'unavailable', 'reason': 'scores are outside [0, 1]; Brier/ECE require probabilities'}
    if p.size and not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        return {'status': 'unavailable', 'reason': 'rows do not sum to 1; Brier/ECE require a probability matrix'}
    onehot = (y_true[:, None] == labels[None, :]).astype(float)
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1))) if p.size else None
    if p.size == 0:
        return {'status': 'unavailable', 'reason': 'no observations'}
    top = np.argmax(p, axis=1)
    confidence = p[np.arange(p.size // p.shape[1]), top]
    correct = (labels[top] == y_true).astype(np.int64)
    _e = _binary_calibration(correct, confidence, n_bins=n_bins)
    return {'status': 'ok', 'brier_score': brier,
            'ece': _e.get('ece'), 'n_bins': _e.get('n_bins'),
            'reliability': _e.get('reliability'),
            'top1_accuracy': float(np.mean(correct)) if correct.size else None,
            'note': ('Brier is multiclass sum-of-squares; ECE uses top-1 confidence '
                     '(top-label calibration) on a row-normalized probability matrix')}


def binary_threshold_curve(y_true, y_score, *, positive_label=None, thresholds=None,
                           metric='f1', max_thresholds=4096):
    """Precision/recall/F1 at every candidate cut for a binary endpoint.

    A binary F1 is only reproducible together with its decision threshold.  The
    default 0.5 cut is a reporting convention, not a tuned optimum; under class
    imbalance it can hide a good ranking behind a badly placed cut.  This helper
    makes the cut explicit and returns the validation-friendly operating point
    for a declared metric (`f1`, `balanced_accuracy` or `youden_j`).

    Threshold selection is post-hoc fitting: callers must tune it on validation
    rows that are disjoint from the evaluation rows and record the split.  The
    returned `best_threshold` is a candidate, never a licence to select on test.
    """
    if metric not in _BINARY_THRESHOLD_METRICS:
        raise ValueError(f'metric must be one of {_BINARY_THRESHOLD_METRICS}, got {metric!r}')
    _mt = _as_seed_int(max_thresholds)
    if _mt is None or _mt < 1:
        raise ValueError(f'max_thresholds must be a positive integer, got {max_thresholds!r}')
    max_thresholds = _mt
    y_bin, pos, uniq = _binary_positive_vector(y_true, positive_label)
    score = np.asarray(y_score, dtype=float).ravel()
    if score.size != y_bin.size:
        raise ValueError(f'y_score {score.shape} and y_true {y_bin.shape} differ')
    if not np.all(np.isfinite(score)):
        raise ValueError('y_score contains NaN/Inf; a threshold cannot be selected from non-finite scores')
    n_pos = int(y_bin.sum())
    n_neg = int(y_bin.size - n_pos)
    order = np.argsort(-score, kind='mergesort')
    sorted_score = score[order]
    tp_cum = np.cumsum(y_bin[order])
    fp_cum = np.cumsum(1 - y_bin[order])
    asc = sorted_score[::-1]
    if thresholds is None:
        last = np.r_[np.nonzero(np.diff(sorted_score))[0], sorted_score.size - 1]
        cand = np.unique(sorted_score[last])
        truncated = False
        if cand.size > int(max_thresholds):
            keep = np.linspace(0, cand.size - 1, int(max_thresholds)).astype(np.int64)
            cand = cand[keep]
            truncated = True
    else:
        cand = np.asarray(list(thresholds), dtype=float).ravel()
        if cand.size == 0 or not np.all(np.isfinite(cand)):
            raise ValueError('thresholds must be a non-empty finite sequence')
        cand = np.unique(cand)
        truncated = False
    rows = []
    for t in cand:
        k = int(score.size - np.searchsorted(asc, float(t), side='left'))
        if k <= 0:
            tp = fp = 0
        else:
            tp, fp = int(tp_cum[k - 1]), int(fp_cum[k - 1])
        rows.append(_threshold_row_from_counts(tp, fp, n_pos - tp, n_neg - fp, t))
    rows.sort(key=lambda r: r['threshold'])
    default = _binary_threshold_metrics(y_bin, score, 0.5)
    if not rows:
        raise ValueError('no candidate threshold could be scored')

    def _key(row):
        value = row[metric]
        return (-value, abs(row['threshold'] - 0.5), row['threshold'])

    best = min(rows, key=_key)
    return {'positive_label': str(pos), 'labels': [str(x) for x in uniq],
            'n': int(y_bin.size), 'n_positive': int(y_bin.sum()),
            'prevalence': float(y_bin.mean()), 'metric': metric,
            'best_threshold': best['threshold'], 'best_value': best[metric],
            'best_metrics': best, 'at_default_0.5': default,
            'threshold_sensitivity': float(best[metric] - default[metric]),
            'n_candidates': int(len(rows)), 'threshold_grid_truncated': truncated,
            'thresholds': rows,
            'note': 'select on validation rows disjoint from evaluation; record the split and metric'}


def select_binary_threshold(y_true, y_score, *, positive_label=None, metric='f1',
                            thresholds=None, max_thresholds=4096):
    """Return the best operating point and its validation diagnostics.

    This is `binary_threshold_curve` reduced to the selected row plus the
    chance floor (positive prevalence) so a caller can persist a compact,
    auditable threshold-selection record.
    """
    curve = binary_threshold_curve(y_true, y_score, positive_label=positive_label,
                                   thresholds=thresholds, metric=metric,
                                   max_thresholds=max_thresholds)
    return {'metric': metric, 'threshold': curve['best_threshold'],
            'value': curve['best_value'], 'metrics': curve['best_metrics'],
            'default_0.5': curve['at_default_0.5'],
            'threshold_sensitivity': curve['threshold_sensitivity'],
            'prevalence': curve['prevalence'],
            'n': curve['n'], 'n_positive': curve['n_positive'],
            'threshold_grid_truncated': curve['threshold_grid_truncated']}


def classification_metrics(y_true, y_score=None, y_pred=None, labels=None, positive_label=None,
                           threshold=None, threshold_sweep=False,
                           include_trivial_baseline=True):
    """Imbalance-aware classification summary without a sklearn dependency.

    Always returns accuracy, balanced accuracy and per-class precision/recall/
    F1; reports **macro, micro and weighted** F1 separately (they can disagree
    sharply under class imbalance; macro includes every requested label, with
    F1=0 for a label the predictions never recover).  For a binary endpoint
    with scores it also returns PR-AUC (average precision), ROC-AUC, the
    positive-class prevalence and the PR-AUC lift over that prevalence, so a
    rare-positive PR-AUC can be read against its chance floor.  For 3+ classes
    it returns one-vs-rest macro/micro/weighted PR-AUC and ROC-AUC.  The caller
    must still choose the endpoint that matches the research question; this
    helper only makes the choice explicit and computable.

    `include_trivial_baseline=True` records the always-most-frequent-class
    predictor's accuracy and macro/micro/weighted F1 under `trivial_baseline`.
    A reported score without that floor is not interpretable: under 99:1
    imbalance a constant majority classifier already has accuracy 0.99.

    `positive_label` names the class whose score/probability column is the
    positive endpoint.  If omitted it defaults to ``labels[-1]`` (the sklearn
    convention for sorted binary labels), but callers with reversed or
    non-numeric labels should pass it explicitly rather than relying on order.

    `threshold` is the cut used when predictions are derived from a 1-D score
    column.  It defaults to 0.5, but the result records that convention in
    `decision_threshold` / `decision_threshold_source`.  A threshold tuned on
    validation must be passed explicitly with its selection metadata; use
    `select_binary_threshold` (or `threshold_sweep=True`) to choose it without
    touching the evaluation split.  `threshold_sweep=True` adds the full
    operating-point curve.  Scores in [0, 1] also get Brier/ECE calibration,
    which asks whether the probabilities are usable, not merely well ranked.
    """
    if threshold is not None:
        if isinstance(threshold, (bool, np.bool_)):
            raise ValueError('threshold must be a finite number, not a boolean')
        try:
            threshold = float(threshold)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'threshold must be a finite number, got {threshold!r}') from None
        if not math.isfinite(threshold):
            raise ValueError(f'threshold must be finite, got {threshold!r}')
    y_true = _preserve_label_array(y_true)
    if labels is None:
        if y_score is not None and np.asarray(y_score).ndim == 2:
            labels = np.arange(np.asarray(y_score).shape[1])
        else:
            labels = _unique_label_values(y_true)
            if y_pred is not None:
                labels = _unique_label_values(np.concatenate([
                    np.asarray(labels, dtype=object),
                    _preserve_label_array(y_pred)]))
    labels = _preserve_label_array(labels)
    if labels.ndim != 1:
        raise ValueError(f'labels must be a flat 1-D sequence, got shape {labels.shape}')
    if labels.size == 0:
        raise ValueError('classification_metrics needs at least one label')
    def _label_identity(value):
        # Canonical identity so equal temporal spellings (Python date vs
        # np.datetime64, aware datetimes of the same instant) are one class.
        return _hashable_label(value)
    if len({_label_identity(v) for v in labels.tolist()}) != int(labels.size):
        raise ValueError(f'labels contains duplicate class identifiers: {labels.tolist()}')
    _label_names = _display_label_names(labels)
    if y_score is not None:
        try:
            y_score = np.asarray(y_score, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('y_score must be numeric (scores/probabilities)') from None
        if not np.all(np.isfinite(y_score)):
            # Ranking with a NaN/Inf score is not defined; treating it as an
            # ordinary rank silently manufactures an AUC/AP from a failed
            # prediction.  Refuse instead of reporting a corrupted metric.
            raise ValueError('y_score contains NaN/Inf; ranking metrics require finite scores')
        if y_score.ndim == 2:
            _ncols = int(y_score.shape[1])
            if _ncols != int(labels.size):
                raise ValueError(
                    f'y_score has {_ncols} score column(s) but labels has '
                    f'{int(labels.size)} identit(ies); a column-per-label score '
                    'matrix must name exactly one label per column')
    pos = None
    if labels.size == 2:
        if positive_label is None:
            pos = labels[-1]
        else:
            hits = np.where(labels == positive_label)[0]
            if hits.size == 0:
                raise ValueError(f'positive_label={positive_label!r} is not in labels {labels.tolist()}')
            pos = labels[int(hits[0])]
    elif positive_label is not None:
        raise ValueError('positive_label is only meaningful for a binary endpoint')
    _threshold_source = None
    _threshold_used = threshold
    if y_pred is None:
        if y_score is None:
            raise ValueError('classification_metrics needs y_pred or y_score')
        score = np.asarray(y_score)
        if score.ndim == 2:
            y_pred = labels[np.argmax(score, axis=1)]
            _threshold_source = 'argmax'
            # Predictions come from argmax, not from the caller's scalar cut;
            # recording that cut as the decision threshold would misdescribe
            # the operating point the metrics were computed at.
            _threshold_used = None
        elif labels.size == 2:
            if score.size != y_true.size:
                raise ValueError(f'y_score {score.shape} and y_true {y_true.shape} differ')
            if threshold is None:
                threshold = 0.5
                _threshold_source = 'default_0.5'
            else:
                _threshold_source = 'caller_threshold'
            _threshold_used = threshold
            neg = labels[0] if pos == labels[-1] else labels[-1]
            _mask = np.asarray(score).ravel() >= threshold
            if labels.dtype == object or y_true.dtype == object:
                # np.where tries to find a common dtype for the two labels;
                # mixed identities (the integer 1 and the string "1") have
                # none in NumPy 2, so build the object array explicitly
                # instead of silently promoting or raising DTypePromotionError.
                y_pred = np.empty(_mask.shape, dtype=object)
                y_pred[_mask] = pos
                y_pred[~_mask] = neg
            else:
                y_pred = np.where(_mask, pos, neg)
        else:
            raise ValueError('a 1-D y_score is only interpretable for a binary endpoint')
    else:
        if threshold is not None:
            _threshold_source = 'caller_threshold'
        elif labels.size == 2:
            _threshold_source = 'caller_predictions'
    y_pred = _preserve_label_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f'y_true {y_true.shape} and y_pred {y_pred.shape} differ')
    # A declared label set must cover every OBSERVED label.  Silently ignoring
    # an undeclared true/predicted class makes micro/macro F1 and balanced
    # accuracy describe a different label space than the one the caller
    # declared.  Requested labels with zero support are still allowed (they
    # count as F1=0), so an explicit full class list remains valid.
    _declared = {_label_identity(v) for v in labels.tolist()}
    _observed = {_label_identity(v) for v in y_true.tolist()} | {
        _label_identity(v) for v in y_pred.tolist()}
    _unknown = [v for v in (_observed - _declared)]
    if _unknown:
        raise ValueError(
            f'observed label(s) {sorted(map(str, _unknown))} are not in the declared '
            f'labels {labels.tolist()}; include them or pass labels=None')
    per_class = {}
    supports = []
    support_all = []
    f1s = []
    rs = []
    all_f1s = []
    total_tp = total_fp = total_fn = 0
    for _li, lab in enumerate(labels):
        _name = _label_names[_li]
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        support = int(np.sum(y_true == lab))
        support_all.append(support)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[_name] = {'precision': precision, 'recall': recall, 'f1': f1,
                            'support': support, 'tp': tp, 'fp': fp, 'fn': fn}
        all_f1s.append(f1)
        # Balanced accuracy / macro recall are also averages over the full
        # REQUESTED label set.  A label absent from y_true (but requested, or
        # present only in y_pred) has recall 0, not "not applicable": dropping
        # it made a constant predictor on a two-label endpoint score balanced
        # accuracy 1.0 instead of 0.5.
        rs.append(recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if support:
            supports.append(support)
            f1s.append(f1)
    micro_f1 = 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
    # Macro averages over the requested label set (zero-support labels count
    # as 0, as in sklearn), so a rare class cannot be dropped from the macro
    # score merely because it happens to be absent from this sample.
    macro_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=supports)) if supports else 0.0
    _pos_name = _label_names[int(np.where(labels == pos)[0][0])] if pos is not None else None
    f1_positive = per_class[_pos_name]['f1'] if _pos_name is not None else None
    out = {
        'n': int(y_true.size),
        'labels': list(_label_names),
        'positive_label': (_pos_name if pos is not None else None),
        'prevalence': {_label_names[i]: float(np.mean(y_true == lab))
                       for i, lab in enumerate(labels)},
        'accuracy': float(np.mean(y_true == y_pred)) if y_true.size else None,
        'balanced_accuracy': float(np.mean(rs)) if rs else None,
        'f1_macro': macro_f1,
        'f1_micro': float(micro_f1),
        'f1_weighted': weighted_f1,
        'precision_micro': (float(total_tp / (total_tp + total_fp))
                            if (total_tp + total_fp) else 0.0),
        'recall_micro': (float(total_tp / (total_tp + total_fn))
                         if (total_tp + total_fn) else 0.0),
        'f1_positive': f1_positive,
        'f1_primary': (f1_positive if pos is not None else macro_f1),
        'f1_primary_average': ('binary_positive_class' if pos is not None else 'macro'),
        'precision_macro': float(np.mean([v['precision'] for v in per_class.values()])) if per_class else 0.0,
        'recall_macro': float(np.mean(rs)) if rs else 0.0,
        'per_class': per_class,
        'f1_averaging': 'macro, micro and weighted are all reported; the caller must name the primary one',
    }
    if pos is not None:
        # A threshold can collapse every prediction to one class.  That is a
        # degenerate operating point, not evidence about ranking: report it
        # explicitly so an F1/accuracy headline is not read as a learned
        # decision rule.
        _pos_entry = per_class.get(_pos_name) or {}
        _pred_pos = int(_pos_entry.get('tp', 0)) + int(_pos_entry.get('fp', 0))
        out['predicted_positive_count'] = _pred_pos
        out['predicted_positive_rate'] = (float(_pred_pos) / float(y_true.size)
                                          if y_true.size else None)
        _true_pos = int(np.sum(y_true == pos))
        if y_true.size and _true_pos and _true_pos < y_true.size and _pred_pos in (0, int(y_true.size)):
            out['degenerate_predictions'] = True
            out['prediction_warning'] = (
                'all predictions collapse to one class at the recorded decision '
                'threshold; F1/precision/recall describe that constant rule, not '
                'the model ranking. Report a threshold-free metric (PR-AUC/ROC-AUC) '
                'or a validation-selected threshold')
        else:
            out['degenerate_predictions'] = False
    if include_trivial_baseline:
        out['trivial_baseline'] = _majority_class_floor(labels, support_all,
                                                       label_names=_label_names)
    if labels.size > 2:
        # Macro and micro F1 answer different questions when classes are
        # imbalanced.  Make the gap visible instead of letting one number stand
        # in for the other.
        out['macro_micro_f1_gap'] = float(macro_f1 - micro_f1)
        _prev = [float(v) for v in out['prevalence'].values()]
        out['minority_prevalence'] = min(_prev) if _prev else None
        out['imbalance_ratio'] = (float((1.0 - min(_prev)) / min(_prev))
                                  if _prev and min(_prev) > 0 else None)
        if _prev and _extreme_imbalance(min(_prev)):
            out['imbalance_warning'] = (
                'a class is at or beyond 1:1000 — report one-vs-rest macro PR-AUC '
                '(pr_auc_macro) as the ranking endpoint; macro/micro/weighted F1 '
                'answer different questions and ROC-AUC can look optimistic')
    if labels.size == 2:
        out['decision_threshold'] = _threshold_used
        out['decision_threshold_source'] = _threshold_source
        out['threshold_dependent_metrics'] = ['accuracy', 'balanced_accuracy', 'precision_macro',
                                              'recall_macro', 'f1_positive', 'f1_primary']
        # Calibration is only threshold-free when the score really is a
        # probability.  Logits/scores outside [0, 1] leave it unavailable, so
        # do not advertise Brier/ECE in the metric list as if they had been
        # measured.
        out['threshold_free_metrics'] = []
    if y_score is not None and labels.size == 2:
        y_bin = (y_true == pos).astype(np.int64)
        # A 2-D score matrix is column-per-label: the binary endpoint uses the
        # named positive-class column, not an assumed last column.
        _score = np.asarray(y_score, dtype=float)
        if _score.ndim == 2:
            _pos_idx = int(np.where(labels == pos)[0][0])
            _bin_score = _score[:, _pos_idx]
        else:
            _bin_score = _score.ravel()
        if _bin_score.size != y_bin.size:
            raise ValueError(f'y_score {_score.shape} does not match y_true {y_true.shape}')
        out['score_min'] = float(np.min(_bin_score)) if _bin_score.size else None
        out['score_max'] = float(np.max(_bin_score)) if _bin_score.size else None
        out['roc_auc'] = _binary_roc_auc(y_bin, _bin_score)
        out['pr_auc'] = _binary_average_precision(y_bin, _bin_score)
        # Only advertise the ranking metrics that were actually computed from
        # a score column; a predictions-only record has neither key.
        out['threshold_free_metrics'].extend(['roc_auc', 'pr_auc'])
        if out['decision_threshold'] is not None and _bin_score.size:
            _t = float(out['decision_threshold'])
            out['threshold_within_score_range'] = bool(
                out['score_min'] <= _t <= out['score_max'])
            if not out['threshold_within_score_range'] and _true_pos and _true_pos < y_bin.size:
                out['threshold_range_warning'] = (
                    f'decision threshold {_t:g} lies outside the observed score range '
                    f'[{out["score_min"]:.4g}, {out["score_max"]:.4g}]; the thresholded '
                    'metrics are a constant predictor. Scores are not calibrated '
                    'probabilities on the 0.5 scale — select a cut on validation or '
                    'report a threshold-free endpoint')
        prevalence = float(np.mean(y_bin))
        out['pr_auc_prevalence'] = prevalence
        out['roc_auc_chance'] = 0.5
        if out['roc_auc'] is not None:
            out['roc_auc_lift_over_chance'] = float(out['roc_auc'] - 0.5)
            out['roc_auc_normalized_skill'] = float((out['roc_auc'] - 0.5) / 0.5)
        if out['pr_auc'] is not None:
            out['pr_auc_lift_over_prevalence'] = float(out['pr_auc'] - prevalence)
            if prevalence < 1.0:
                out['pr_auc_normalized_skill'] = float(
                    (out['pr_auc'] - prevalence) / (1.0 - prevalence))
        out['calibration'] = _binary_calibration(y_bin, _bin_score)
        if out['calibration'].get('status') == 'ok':
            out['brier_score'] = out['calibration']['brier_score']
            out['ece'] = out['calibration']['ece']
            out['threshold_free_metrics'].extend(['brier_score', 'ece'])
        out['score_orientation_warning'] = None
        if (out['roc_auc'] is not None and out['roc_auc'] < 0.5
                and int(y_bin.sum()) >= 2 and int((1 - y_bin).sum()) >= 2):
            out['score_orientation_warning'] = (
                'ROC-AUC is below chance; check that the score column is the '
                'positive-class probability and that positive_label is correct')
        if out['pr_auc'] is not None and out['roc_auc'] is not None:
            # The policy switch is about the POSITIVE endpoint's prevalence.
            # When the positive class is common and the negative class is
            # rare, positive-class PR-AUC has a ~0.999 chance floor and is
            # not the rare-event endpoint; min(prev, 1-prev) would demand it.
            out['imbalance_warning'] = ('positives at or beyond 1:1000 — PR-AUC is the primary ranking metric; ROC-AUC can look optimistic'
                                        if _extreme_imbalance(prevalence) else None)
        if threshold_sweep:
            out['threshold_selection'] = select_binary_threshold(
                y_true, _bin_score, positive_label=pos)
    elif labels.size == 2 and out.get('positive_label') is not None:
        prev = out['prevalence'].get(_pos_name, 0.0)
        if _extreme_imbalance(prev):
            out['imbalance_warning'] = ('positives at or beyond 1:1000 — report PR-AUC with scores; F1 must name its averaging rule and recall should be reported per class')
    if labels.size > 2 and y_score is not None:
        _score = np.asarray(y_score, dtype=float)
        if _score.ndim == 2:
            _rank = _multiclass_ranking_metrics(y_true, _score, labels,
                                                label_names=_label_names)
            # Keep one per_class table: counts/F1 plus the one-vs-rest ranking
            # values, so a class-level read does not need a second artifact.
            for _lab, _vals in _rank.pop('per_class').items():
                out['per_class'][_lab].update(_vals)
            out.update(_rank)
            out['threshold_free_metrics'] = [
                'roc_auc_macro', 'roc_auc_micro', 'roc_auc_weighted',
                'pr_auc_macro', 'pr_auc_micro', 'pr_auc_weighted']
            out['ranking_averaging'] = 'one-vs-rest'
            out['calibration'] = _multiclass_calibration(y_true, _score, labels)
            if out['calibration'].get('status') == 'ok':
                out['brier_score'] = out['calibration']['brier_score']
                out['ece'] = out['calibration']['ece']
                out['threshold_free_metrics'].extend(['brier_score', 'ece'])
        else:
            out['ranking_unavailable_reason'] = (
                'multiclass ranking needs a column-per-label score matrix; '
                'a 1-D score is only interpretable for a binary endpoint')
    # Prevalence-aware endpoint recommendation.  This does not replace the
    # caller's declared hypothesis; it makes the conditional METRIC_POLICY
    # rule machine-readable at the observed base rate, so a rare-positive
    # panel cannot silently keep an F1/ROC-only headline.
    _thr = 1e-3
    if pos is not None:
        _prev = float(np.mean(np.asarray(y_true) == pos))
        # The conditional binary endpoint is keyed to the named positive
        # class (METRIC_POLICY), not to whichever class happens to be rarer.
        # A common positive with a rare negative must keep F1; positive-class
        # PR-AUC would be read against a ~1 - prevalence chance floor.
        _rare = bool(_extreme_imbalance(_prev, _thr))
        if _rare:
            out['recommended_primary_metric'] = 'pr_auc'
            out['recommended_primary_average'] = 'binary_positive_class'
            out['recommended_primary_reason'] = (
                f'positive prevalence {_prev:.3e} is at or beyond 1:1000; '
                'PR-AUC is the policy primary under extreme imbalance')
        else:
            out['recommended_primary_metric'] = 'f1'
            out['recommended_primary_average'] = 'binary_positive_class'
            out['recommended_primary_reason'] = (
                f'positive prevalence {_prev:.3e} is not extreme; '
                'positive-class F1 is the policy primary')
        out['recommended_primary_available'] = bool(
            out.get('pr_auc') is not None if _rare else True)
    elif labels.size > 2:
        _prev_all = [float(v) for v in out.get('prevalence', {}).values()]
        if _prev_all:
            _rare_val = min(_prev_all)
            _rare = bool(_extreme_imbalance(_rare_val, _thr))
            if _rare:
                out['recommended_primary_metric'] = 'pr_auc_macro'
                out['recommended_primary_average'] = 'one_vs_rest_macro'
                out['recommended_primary_reason'] = (
                    f'minority prevalence {_rare_val:.3e} is at or beyond 1:1000; '
                    'one-vs-rest macro PR-AUC is the policy primary')
            else:
                out['recommended_primary_metric'] = 'f1_macro'
                out['recommended_primary_average'] = 'macro'
                out['recommended_primary_reason'] = (
                    f'minority prevalence {_rare_val:.3e} is not extreme; '
                    'macro F1 is the policy primary')
            out['recommended_primary_available'] = bool(
                out.get('pr_auc_macro') is not None if _rare else True)
    return out
