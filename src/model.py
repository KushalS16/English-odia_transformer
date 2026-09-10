from __future__ import annotations
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=512):
        super().__init__(); pe=torch.zeros(max_len,d_model); pos=torch.arange(max_len).unsqueeze(1).float(); div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout):
        super().__init__(); assert d_model%n_heads==0; self.h=n_heads; self.d=d_model//n_heads; self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,d_model); self.o=nn.Linear(d_model,d_model); self.drop=nn.Dropout(dropout)
    def forward(self,q,k,v,attn_mask=None,key_padding_mask=None):
        B,Lq,_=q.shape; Lk=k.shape[1]
        q=self.q(q).view(B,Lq,self.h,self.d).transpose(1,2); k=self.k(k).view(B,Lk,self.h,self.d).transpose(1,2); v=self.v(v).view(B,Lk,self.h,self.d).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.d)
        if attn_mask is not None: scores=scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0),float('-inf'))
        if key_padding_mask is not None: scores=scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),float('-inf'))
        a=torch.softmax(scores,dim=-1); a=self.drop(a); out=(a@v).transpose(1,2).contiguous().view(B,Lq,-1); return self.o(out)

class FFN(nn.Module):
    def __init__(self,d_model,d_ff,dropout): super().__init__(); self.net=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_ff,d_model))
    def forward(self,x): return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout):
        super().__init__(); self.a=MultiHeadAttention(d_model,n_heads,dropout); self.f=FFN(d_model,d_ff,dropout); self.n1=nn.LayerNorm(d_model); self.n2=nn.LayerNorm(d_model); self.d=nn.Dropout(dropout)
    def forward(self,x,pad): x=self.n1(x+self.d(self.a(x,x,x,key_padding_mask=pad))); return self.n2(x+self.d(self.f(x)))

class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout):
        super().__init__(); self.sa=MultiHeadAttention(d_model,n_heads,dropout); self.ca=MultiHeadAttention(d_model,n_heads,dropout); self.f=FFN(d_model,d_ff,dropout); self.n1=nn.LayerNorm(d_model); self.n2=nn.LayerNorm(d_model); self.n3=nn.LayerNorm(d_model); self.d=nn.Dropout(dropout)
    def forward(self,x,memory,tgt_pad,src_pad,causal):
        x=self.n1(x+self.d(self.sa(x,x,x,attn_mask=causal,key_padding_mask=tgt_pad)))
        x=self.n2(x+self.d(self.ca(x,memory,memory,key_padding_mask=src_pad)))
        return self.n3(x+self.d(self.f(x)))

class Transformer(nn.Module):
    def __init__(self,src_vocab,tgt_vocab,d_model=128,n_heads=4,n_encoder_layers=2,n_decoder_layers=2,d_ff=512,dropout=.1,max_len=128):
        super().__init__(); self.d_model=d_model; self.src_pad=0; self.tgt_pad=0; self.se=nn.Embedding(src_vocab,d_model,padding_idx=0); self.te=nn.Embedding(tgt_vocab,d_model,padding_idx=0); self.pe=PositionalEncoding(d_model,max_len); self.enc=nn.ModuleList([EncoderBlock(d_model,n_heads,d_ff,dropout) for _ in range(n_encoder_layers)]); self.dec=nn.ModuleList([DecoderBlock(d_model,n_heads,d_ff,dropout) for _ in range(n_decoder_layers)]); self.out=nn.Linear(d_model,tgt_vocab)
    @staticmethod
    def causal_mask(L,device): return torch.triu(torch.ones(L,L,dtype=torch.bool,device=device),diagonal=1)
    def encode(self,src):
        pad=src.eq(self.src_pad); x=self.pe(self.se(src)*math.sqrt(self.d_model))
        for b in self.enc: x=b(x,pad)
        return x,pad
    def decode(self,tgt,memory,src_pad):
        pad=tgt.eq(self.tgt_pad); x=self.pe(self.te(tgt)*math.sqrt(self.d_model)); m=self.causal_mask(tgt.size(1),tgt.device)
        for b in self.dec: x=b(x,memory,pad,src_pad,m)
        return self.out(x)
    def forward(self,src,tgt): memory,sp=self.encode(src); return self.decode(tgt,memory,sp)
