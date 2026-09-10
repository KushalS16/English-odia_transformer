from __future__ import annotations
import json
from pathlib import Path
import sentencepiece as spm

SPECIALS = {'pad':0,'unk':1,'sos':2,'eos':3}

class SPTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id, self.unk_id, self.sos_id, self.eos_id = [self.sp.piece_to_id(x) for x in ['<pad>','<unk>','<s>','</s>']]
    def encode(self, text, max_tokens=None):
        ids = [self.sos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]
        if max_tokens: ids = ids[:max_tokens-1] + [self.eos_id] if len(ids) > max_tokens else ids
        return ids
    def decode(self, ids):
        ids = [int(x) for x in ids if int(x) not in (self.pad_id,self.sos_id)]
        if self.eos_id in ids: ids = ids[:ids.index(self.eos_id)]
        return self.sp.decode(ids)
    def vocab_size(self): return self.sp.get_piece_size()

def train_spm(input_file, model_prefix, vocab_size, character_coverage):
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=str(input_file), model_prefix=str(model_prefix), vocab_size=vocab_size,
        model_type='bpe', character_coverage=character_coverage,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>',
        hard_vocab_limit=False, input_sentence_size=0, shuffle_input_sentence=False
    )
