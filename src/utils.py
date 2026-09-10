from __future__ import annotations
import json, random, re
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_text(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize('NFC', str(text))
    text = ''.join(ch for ch in text if unicodedata.category(ch) not in ('Cc','Cf') or ch in '\n\t')
    return re.sub(r'\s+', ' ', text).strip()

def get_device(cfg_device='auto'):
    if cfg_device == 'auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg_device)
