from __future__ import annotations
import torch

def greedy_decode(model, src_ids, tokenizer, device, max_new_tokens=126):
    model.eval(); src=torch.tensor([src_ids],dtype=torch.long,device=device); memory,sp=model.encode(src); ys=torch.tensor([[tokenizer.sos_id]],device=device)
    for _ in range(max_new_tokens-1):
        logits=model.decode(ys,memory,sp)[:,-1,:]; nxt=logits.argmax(-1).item(); ys=torch.cat([ys,torch.tensor([[nxt]],device=device)],1)
        if nxt==tokenizer.eos_id: break
    return tokenizer.decode(ys[0].tolist())

def beam_search(model,src_ids,tokenizer,device,beam_size=4,max_new_tokens=126,length_penalty=.7):
    model.eval(); src=torch.tensor([src_ids],dtype=torch.long,device=device); memory,sp=model.encode(src)
    beams=[([tokenizer.sos_id],0.0)]
    for _ in range(max_new_tokens-1):
        cand=[]
        for seq,score in beams:
            if seq[-1]==tokenizer.eos_id: cand.append((seq,score)); continue
            ys=torch.tensor([seq],device=device); logp=torch.log_softmax(model.decode(ys,memory,sp)[:,-1,:],-1); vals,idx=torch.topk(logp,beam_size)
            for v,i in zip(vals[0].tolist(),idx[0].tolist()): cand.append((seq+[i],score+v))
        beams=sorted(cand,key=lambda x:x[1]/(len(x[0])**length_penalty),reverse=True)[:beam_size]
        if all(s[-1]==tokenizer.eos_id for s,_ in beams): break
    return tokenizer.decode(beams[0][0])
