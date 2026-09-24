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
    import datasets
    datasets.load_dataset = fake_load_dataset
    import exp_lib as L
    combos = [(512, 1000000), (1024, 40000000), (512, 110000000)]
    for seq_len, cap in combos:
        print(f'\n===== prep cache: seq_len={seq_len} cap={cap} =====')
        L.load_wikitext(seq_len, cap)
    print('\n[prep] ALL CACHES DONE')
if __name__ == '__main__':
    main()
