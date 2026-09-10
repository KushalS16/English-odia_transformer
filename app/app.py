from pathlib import Path
import time, yaml, torch, streamlit as st
from src.utils import get_device
from src.tokenizer import SPTokenizer
from src.model import Transformer
from src.inference import greedy_decode,beam_search

ROOT=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load(open(ROOT/'configs/config.yaml',encoding='utf-8')); device=get_device(cfg['device']); a=ROOT/'artifacts'
@st.cache_resource
def load():
    src=SPTokenizer(a/'src.model'); tgt=SPTokenizer(a/'tgt.model'); model=Transformer(src.vocab_size(),tgt.vocab_size(),**cfg['model']).to(device); ck=torch.load(a/'best_model.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); model.eval(); return model,src,tgt,ck
st.set_page_config(page_title='English → Odia Transformer',page_icon='🔤',layout='wide')
st.title('English → Odia Transformer')
st.caption('Transformer trained from scratch • d=128 • 4 heads • 2 encoder + 2 decoder blocks')
if not (a/'best_model.pt').exists() or not (a/'src.model').exists() or not (a/'tgt.model').exists():
    st.error('Model artifacts are missing. Run: python run_project.py setup → python run_project.py test → python run_project.py train')
    st.stop()
model,src_tok,tgt_tok,ck=load()
with st.sidebar:
    st.subheader('System')
    st.write('Device:',str(device))
    st.write('Source vocab:',src_tok.vocab_size())
    st.write('Target vocab:',tgt_tok.vocab_size())
    st.write('Beam search: available')
    st.divider(); st.subheader('Model')
    st.json({'d_model':cfg['model']['d_model'],'heads':cfg['model']['n_heads'],'encoder_blocks':cfg['model']['n_encoder_layers'],'decoder_blocks':cfg['model']['n_decoder_layers'],'d_ff':cfg['model']['d_ff']})
text=st.text_area('English sentence',value='The government is working to improve education and healthcare for all citizens.',height=140)
mode=st.radio('Decoding strategy',['Greedy','Beam search'],horizontal=True)
beam=st.slider('Beam size',2,8,int(cfg['inference']['beam_size'])) if mode=='Beam search' else 1
if text.strip():
    token_count=len(src_tok.sp.encode(text.strip(),out_type=int))+2
    if token_count>cfg['data']['max_tokens']:
        st.warning(f'This input is {token_count} source tokens; the model limit is {cfg["data"]["max_tokens"]}. The input will be safely truncated.')
if st.button('Translate',type='primary',use_container_width=True):
    clean=text.strip()
    if not clean: st.warning('Enter an English sentence.'); st.stop()
    ids=src_tok.encode(clean,cfg['data']['max_tokens'])
    t0=time.perf_counter()
    with st.spinner('Translating…'):
        if mode=='Greedy': out=greedy_decode(model,ids,tgt_tok,device,cfg['inference']['max_new_tokens'])
        else: out=beam_search(model,ids,tgt_tok,device,beam,cfg['inference']['max_new_tokens'],cfg['inference']['length_penalty'])
    ms=(time.perf_counter()-t0)*1000
    st.subheader('Odia translation')
    st.success(out)
    st.caption(f'{mode} decoding • {ms:.1f} ms • input tokens: {len(ids)}')
with st.expander('About this model'):
    st.write('Built from scratch with separate English/Odia SentencePiece BPE tokenizers, sinusoidal positional encoding, explicit multi-head attention, residual + LayerNorm blocks, causal masking, teacher forcing and PAD-ignored cross-entropy.')
