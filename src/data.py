from __future__ import annotations
import json, random
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import normalize_text
from .tokenizer import SPTokenizer

class TranslationDataset(Dataset):
    def __init__(self, csv_path, src_tok, tgt_tok, max_tokens=126):
        self.df = pd.read_csv(csv_path).fillna('')
        self.src_tok, self.tgt_tok, self.max_tokens = src_tok, tgt_tok, max_tokens
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]
        return self.src_tok.encode(r.src, self.max_tokens), self.tgt_tok.encode(r.tgt, self.max_tokens), r.src, r.tgt

def collate_fn(batch, src_pad, tgt_pad):
    srcs,tgts,raws,refs=zip(*batch)
    ms=max(map(len,srcs)); mt=max(map(len,tgts))
    src=torch.full((len(batch),ms),src_pad,dtype=torch.long); tgt=torch.full((len(batch),mt),tgt_pad,dtype=torch.long)
    for i,x in enumerate(srcs): src[i,:len(x)] = torch.tensor(x)
    for i,x in enumerate(tgts): tgt[i,:len(x)] = torch.tensor(x)
    return {'src':src,'tgt':tgt,'raw_src':raws,'raw_tgt':refs}

def prepare_dataframe(df, max_pairs, min_chars, max_chars, seed=42):
    cols = df[['src','tgt']].copy()
    cols['src']=cols.src.map(normalize_text); cols['tgt']=cols.tgt.map(normalize_text)
    cols=cols[(cols.src.str.len()>=min_chars)&(cols.tgt.str.len()>=min_chars)]
    cols=cols[(cols.src.str.len()<=max_chars)&(cols.tgt.str.len()<=max_chars)]
    cols=cols.drop_duplicates().sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_pairs and len(cols)>max_pairs: cols=cols.iloc[:max_pairs].copy()
    return cols
