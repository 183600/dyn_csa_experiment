# ---- multi-arm composability: does the ablation table form a coherent graph? --
# Every contrast above was checked PAIR BY PAIR.  A table of independent pairs
# can still splice two different runs of the same arm, or compare components
# that live in disconnected graphs.  This block adds the arm-graph readout; it
# is descriptive, adds no p-value to a declared family, and never changes
# p_holm / verified / a frozen decision.
try:
    _contrast_network = contrast_network_audit(
        all_entries, alpha=alpha, min_n=min_n)
except (TypeError, ValueError) as _cn_exc:
    _contrast_network = {
        'kind': 'contrast_network_audit',
        'status': 'not_estimable',
        'reason': '%s: %s' % (type(_cn_exc).__name__, _cn_exc),
        'role': ('multi-arm composability description only; it adds no '
                 'p-value and never changes p/Holm/verified'),
        'warnings': ['the arm graph could not be built from the audited '
                     'entries; do not quote a cross-component ordering'],
    }
report['contrast_network'] = _contrast_network
for _cn_w in _contrast_network.get('warnings') or []:
    report['warnings'].append('contrast network: ' + str(_cn_w))
