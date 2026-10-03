#!/usr/bin/env python3
import os, sys, json
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import pandas as pd
PARQUET_DIR = '/root/autodl-tmp/wt103_parquet'

class FakeSplit:

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def select(self, idx):
        return FakeSplit([self.texts[i] for i in idx])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {'text': self.texts[key]}
        if key == 'text':
            return self.texts
        raise TypeError(key)

_CORPUS = None

def fake_load_dataset(repo, config, cache_dir=None):
    global _CORPUS
    assert config == 'wikitext-103-raw-v1', config
    if _CORPUS is None:
        print(f'[prep] fake load_dataset({repo}, {config}) from {PARQUET_DIR}')
        train = pd.concat([pd.read_parquet(os.path.join(PARQUET_DIR, f'train-0000{i}-of-00002.parquet')) for i in (0, 1)])['text'].tolist()
        val = pd.read_parquet(os.path.join(PARQUET_DIR, 'validation-00000-of-00001.parquet'))['text'].tolist()
        print(f'[prep] train rows={len(train)}  val rows={len(val)}')
        _CORPUS = (train, val)
    return {'train': FakeSplit(_CORPUS[0]), 'validation': FakeSplit(_CORPUS[1])}

def main():
    import numpy as np
    import datasets
    datasets.load_dataset = fake_load_dataset
    import exp_lib as L
    combos = [(512, 1000000), (512, 8000000), (512, 40000000), (1024, 40000000), (512, 110000000), (2048, 8000000), (2048, 1000000), (4097, 4000000)]
    by_seq = {}
    for seq_len, cap in combos:
        by_seq.setdefault(seq_len, set()).add(cap)
    for seq_len in sorted(by_seq):
        caps = sorted(by_seq[seq_len])
        biggest = caps[-1]
        print(f'\n===== prep cache: seq_len={seq_len} cap={biggest} =====')
        train_ids, val_batch, vocab, _, val_bnd = L.load_wikitext(seq_len, biggest)
        if isinstance(train_ids, np.ndarray):
            src_fp = L._ids_fp(train_ids)
        else:
            src_fp = None
        _me_big_path = os.path.join('./wt103_cache', f'meta_v4_sl{seq_len}_cap{biggest}_v8192_vs512.json')
        _big_fp = None
        try:
            with open(_me_big_path, encoding='utf-8') as _f:
                _big_fp = json.load(_f).get('src_fp')
        except Exception:
            _big_fp = None
        if src_fp is not None and _big_fp != src_fp:
            print(f'[prep] seq_len={seq_len}: the cap={biggest} cache carries src_fp={_big_fp!r} while the ids it served hash to {src_fp!r} — the source tokenisation is not verifiable; REFUSING to derive smaller-cap caches from it. Remove the cap={biggest} cache files and re-run to re-tokenise from the corpus.')
            continue
        for cap in caps[:-1]:
            tag = f'v4_sl{seq_len}_cap{cap}_v8192_vs512'
            tr_path = os.path.join('./wt103_cache', f'train_ids_{tag}.npy')
            va_path = os.path.join('./wt103_cache', f'val_batch_{tag}.npy')
            vp_path = os.path.join('./wt103_cache', f'val_bnd_{tag}.npy')
            me_path = os.path.join('./wt103_cache', f'meta_{tag}.json')
            if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
                stale = False
                try:
                    with open(me_path, encoding='utf-8') as _f:
                        _me = json.load(_f)
                    stale = src_fp is not None and _me.get('src_fp') != src_fp
                except Exception:
                    stale = True
                if not stale:
                    print(f'[prep] seq_len={seq_len} cap={cap}: cache already present, skipped')
                    continue
                print(f'[prep] seq_len={seq_len} cap={cap}: derived cache predates the current cap={biggest} tokenisation — regenerating')
            for path, arr in ((tr_path, train_ids[:cap + seq_len]), (va_path, val_batch), (vp_path, val_bnd)):
                _tmp = f'{path}.tmp{os.getpid()}'
                np.save(_tmp, arr)
                os.replace(_tmp + '.npy', path)
            L.atomic_write_json(me_path, {'vocab': vocab, 'src_fp': src_fp}, indent=0)
            print(f'[prep] seq_len={seq_len} cap={cap}: derived from the cap={biggest} tokenisation (same corpus prefix, same vocab)')
    print('\n[prep] ALL CACHES DONE')
if __name__ == '__main__':
    main()
