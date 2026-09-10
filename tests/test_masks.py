import torch
from src.model import Transformer

def test_causal_mask_is_strictly_upper_triangular():
    m=Transformer.causal_mask(5,'cpu')
    assert m.dtype==torch.bool
    assert not m[0,0] and not m[4,4]
    assert m[0,1] and m[0,4] and m[2,4]
    assert not m[3,2]

def test_no_future_attention():
    m=Transformer.causal_mask(7,'cpu')
    for i in range(7):
        assert not m[i,:i+1].any()
        assert m[i,i+1:].all()

def test_future_token_cannot_change_earlier_decoder_logits():
    torch.manual_seed(7)
    model=Transformer(32,40,d_model=128,n_heads=4,n_encoder_layers=2,n_decoder_layers=2,d_ff=256,dropout=0.0,max_len=32).eval()
    src=torch.randint(1,32,(1,8))
    tgt_a=torch.tensor([[2,5,6,7,8]])
    tgt_b=tgt_a.clone(); tgt_b[0,-1]=19
    with torch.no_grad():
        la=model(src,tgt_a); lb=model(src,tgt_b)
    assert torch.allclose(la[:,:-1],lb[:,:-1],atol=1e-6,rtol=1e-6)
