from __future__ import annotations
import pandas as pd
import sacrebleu

def corpus_bleu(preds,refs): return float(sacrebleu.corpus_bleu(preds,[refs]).score)

def save_samples(rows,path): pd.DataFrame(rows).to_csv(path,index=False)
