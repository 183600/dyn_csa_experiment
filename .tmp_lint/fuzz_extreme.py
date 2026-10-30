import json, math, sys, traceback
import numpy as np
import exp_lib as L

X = [1.7e308, -1.7e308, 1.7e308]
XP = [1.7e308, 1.6e308, 1.5e308]
TINY = [1e-300, -1e-300, 2e-300]

def check(name, fn, *a, **k):
    try:
        r = fn(*a, **k)
        try:
            json.dumps(r, allow_nan=False)
            print('OK   ', name)
        except Exception as e:
            print('JSON ', name, type(e).__name__, str(e)[:120])
            # find nonfinite
            def walk(v, path=''):
                if isinstance(v, dict):
                    for kk,vv in v.items(): walk(vv, path+'.'+str(kk))
                elif isinstance(v, (list,tuple)):
                    for i,vv in enumerate(v): walk(vv, path+f'[{i}]')
                elif isinstance(v,float) and not math.isfinite(v):
                    print('   nonfinite', path, v)
            walk(r)
    except Exception as e:
        print('ERR  ', name, type(e).__name__, str(e)[:160])

check('paired_bootstrap_ci', L.paired_bootstrap_ci, X, n_boot=20)
check('unpaired_bootstrap_ci', L.unpaired_bootstrap_ci, XP, [1.0,2.0,3.0], n_boot=20)
check('wilcoxon', L.wilcoxon_signed_rank_p, X)
check('new_seed_replication', L.new_seed_replication, X)
check('paired_fragility', L.paired_fragility, X)
check('paired_effect_stats', L.paired_effect_stats, X)
check('tost_paired', L.tost_paired, X, 1e308)
check('ablation_practical_decision', L.ablation_practical_decision, X, margin=1e308)
check('interaction_practical_decision', L.interaction_practical_decision, X, margin=1e308)
check('crossed_variance_components', L.crossed_variance_components, [[1.7e308,1.6e308],[1.5e308,1.4e308]])
check('unit_precision_design', L.unit_precision_design, 10000, pilot_unit_deltas=[x*1e307 for x in [-1,-.8,-1.2,-.9,-1.1,-.95]], target_halfwidth=1e-300, sesoi=1e-300)
check('effect_modification', L.effect_modification, {'a':[1.7e308,-1.7e308,1.7e308],'b':[1.6e308,-1.6e308,1.6e308],'c':[1.5e308,-1.5e308,1.5e308]})
check('scaling_trend', L.scaling_trend, [{'label':'a','scale':1.0,'seeds':[0,1,2],'effects':[1.6e308,-1.6e308,1.6e308]},{'label':'b','scale':2.0,'seeds':[0,1,2],'effects':[1.7e308,-1.7e308,1.7e308]},{'label':'c','scale':4.0,'seeds':[0,1,2],'effects':[1.5e308,-1.5e308,1.5e308]}], min_seeds=2)
check('loglog_regression', L.loglog_regression, [(1.0,1e308),(2.0,1e308),(4.0,1e308)])
check('scale_ladder_audit', L.scale_ladder_audit, [{'name':'a','cfg':{'steps':1,'batch_size':1,'seq_len':1,'n_train_tokens':1,'lr':3e-4,'warmup':1,'seeds':[0,1]},'params':10**300},{'name':'b','cfg':{'steps':1,'batch_size':1,'seq_len':1,'n_train_tokens':1,'lr':3e-4,'warmup':10**300,'seeds':[0,1]},'params':1}])
check('factorial_power', L.factorial_interaction_power, {'11':[1.7e308]*3,'10':[1.0]*3,'01':[1.0]*3,'00':[1.0]*3}, n_seeds=3)
check('trajectory', L.paired_trajectory_stability, {0:[[0,1.0],[10,1.7e308]]}, {0:[[0,1.0],[10,1.5e308]]}, primary_delta=1.0)
check('max_t', L.maxt_signflip_adjust, {'a':[1.7e308,-1.7e308,1.6e308], 'b':[1.5e308,-1.5e308,1.4e308]})
check('missingness', L.missingness_sensitivity, X, seeds=[0,1,2], planned_seeds=[0,1,2], lo=0.0, hi=1e308)
check('analysis_choice', L.analysis_choice_concordance, {'m':1.7e308,'lo':1.6e308,'hi':1.5e308}, [{'name':'a','mean':1.7e308,'ci95_low':-1.7e308,'ci95_high':1.7e308}])
check('directional', L.directional_claim_verdicts, {'a':{'deltas':X}})
check('confirmation_robustness', L.confirmation_robustness, {'a':X})
check('holm_jackknife', L.holm_decision_jackknife, {'a':X})
