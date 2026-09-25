#!/usr/bin/env python3
import os
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

def fake_load_dataset(repo, config, cache_dir=None):
    assert config == 'wikitext-103-raw-v1', config
    print(f'[prep] fake load_dataset({repo}, {config}) from {PARQUET_DIR}')
    train = pd.concat([pd.read_parquet(os.path.join(PARQUET_DIR, f'train-0000{i}-of-00002.parquet')) for i in (0, 1)])['text'].tolist()
    val = pd.read_parquet(os.path.join(PARQUET_DIR, 'validation-00000-of-00001.parquet'))['text'].tolist()
    print(f'[prep] train rows={len(train)}  val rows={len(val)}')
    return {'train': FakeSplit(train), 'validation': FakeSplit(val)}

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
        for cap in caps[:-1]:
            tag = f'v3_sl{seq_len}_cap{cap}_v8192_vs512'
            tr_path = os.path.join('./wt103_cache', f'train_ids_{tag}.npy')
            va_path = os.path.join('./wt103_cache', f'val_batch_{tag}.npy')
            vp_path = os.path.join('./wt103_cache', f'val_bnd_{tag}.npy')
            me_path = os.path.join('./wt103_cache', f'meta_{tag}.json')
            if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
                print(f'[prep] seq_len={seq_len} cap={cap}: cache already present, skipped')
                continue
            for path, arr in ((tr_path, train_ids[:cap + seq_len]), (va_path, val_batch), (vp_path, val_bnd)):
                np.save(path + '.tmp', arr)
                os.replace(path + '.tmp.npy', path)
            L.atomic_write_json(me_path, {'vocab': vocab}, indent=0)
            print(f'[prep] seq_len={seq_len} cap={cap}: derived from the cap={biggest} tokenisation (same corpus prefix, same vocab)')
    print('\n[prep] ALL CACHES DONE')
if __name__ == '__main__':
    main()
