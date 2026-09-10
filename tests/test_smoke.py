import torch
from src.model import Transformer

def test_forward_shape_and_backward():
    model=Transformer(32,40,d_model=128,n_heads=4,n_encoder_layers=2,n_decoder_layers=2,d_ff=256,max_len=32)
    src=torch.randint(1,32,(2,10)); tgt=torch.randint(1,40,(2,9)); logits=model(src,tgt)
    assert logits.shape==(2,9,40)
    logits.mean().backward()
    assert any(p.grad is not None for p in model.parameters())

def test_pad_embedding_is_fixed_zero():
    model=Transformer(32,40,d_model=128,n_heads=4,n_encoder_layers=2,n_decoder_layers=2,d_ff=256,max_len=32)
    assert torch.equal(model.se.weight[0], torch.zeros_like(model.se.weight[0]))
    assert torch.equal(model.te.weight[0], torch.zeros_like(model.te.weight[0]))
