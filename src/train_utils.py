from __future__ import annotations
import math, torch
from torch.optim import AdamW

def transformer_lr(step,d_model,warmup,base_lr):
    step=max(1,step); warmup=max(1,warmup)
    return base_lr * min(step / warmup, 1.0) * (warmup / step) ** 0.5

def make_optimizer(model,cfg):
    return AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(0.9,0.98),eps=1e-9,weight_decay=cfg['weight_decay'])

def set_warmup_lr(opt,step,cfg,d_model):
    lr=transformer_lr(step,d_model,cfg['warmup_steps'],cfg['learning_rate'])
    for g in opt.param_groups: g['lr']=lr
    return lr

def loss_fn(logits,target,pad_id,label_smoothing=0.0):
    return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),ignore_index=pad_id,label_smoothing=label_smoothing)

def perplexity(loss): return math.exp(min(20,float(loss)))
