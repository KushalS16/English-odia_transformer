# Architecture and engineering decisions

## Assignment fidelity
The implementation follows the requested architecture rather than replacing it with `torch.nn.Transformer` or a pretrained translation model.

## Attention
Each attention module performs explicit Q/K/V projections, scaled dot-product attention and head concatenation. Encoder attention uses source padding masks. Decoder self-attention uses both target padding masking and a strict causal mask. Decoder cross-attention uses the source padding mask.

## Normalization and residuals
Every encoder/decoder sublayer uses residual addition followed by LayerNorm. FFN uses Linear → GELU → Dropout → Linear.

## Tokenization
Separate SentencePiece BPE vocabularies are used. This avoids forcing English and Odia into a single vocabulary and lets the Odia tokenizer allocate capacity to script-specific subword patterns.

## Data leakage prevention
Splitting occurs before tokenizer training, and tokenizers are trained from the training/validation/test text only in the current utility. For strict experiment hygiene, the production run should train tokenizers from **train.csv only**; the tokenizer script trains from train.csv only, avoiding validation/test text exposure.
