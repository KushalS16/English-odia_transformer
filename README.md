# Test-7 — English → Odia Transformer From Scratch

A complete **English → Odia Neural Machine Translation (NMT)** system implemented from scratch using **PyTorch**, following the Test-7 Transformer architecture requirements.

The project covers the complete machine-translation pipeline:

```text
Dataset
   ↓
Data Cleaning & Filtering
   ↓
Train / Validation / Test Split
   ↓
SentencePiece BPE Tokenization
   ↓
Transformer Training
   ↓
Checkpointing
   ↓
Evaluation
   ↓
Greedy / Beam Search Decoding
   ↓
Streamlit Translation UI
```

---

## 🎯 Project Objective

The objective of this project is to build an **English-to-Odia neural machine translation system without using a pretrained translation model**.

The Transformer architecture is implemented directly in PyTorch.

### Main Features

* English → Odia translation
* Transformer encoder-decoder implemented from scratch
* SentencePiece BPE tokenization
* Separate source and target tokenizers
* Teacher-forced training
* Strict causal decoder masking
* Multi-head self-attention
* Encoder-decoder cross-attention
* Sinusoidal positional encoding
* Greedy decoding
* Beam-search decoding
* Configurable beam size
* SacreBLEU evaluation
* Checkpoint-based training recovery
* CPU-friendly configuration
* Streamlit translation interface
* Automated tests and sanity checks
* Reproducible dataset splitting and experiments

---

# 🧠 Transformer Architecture

The system uses a standard encoder-decoder Transformer architecture.

```text
                         English Sentence
                                │
                                ▼
                       SentencePiece BPE
                                │
                                ▼
                         Token IDs
                                │
                                ▼
                         Token Embeddings
                                │
                                +
                                │
                    Sinusoidal Positional
                         Encoding
                                │
                                ▼
                 ┌──────────────────────────┐
                 │      ENCODER BLOCK 1     │
                 │                          │
                 │ Multi-Head Self-Attention│
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 │          ↓               │
                 │ Feed Forward Network     │
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 └──────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │      ENCODER BLOCK 2     │
                 │                          │
                 │ Multi-Head Self-Attention│
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 │          ↓               │
                 │ Feed Forward Network     │
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 └──────────────────────────┘
                                │
                                ▼
                         Encoder Memory
                                │
                                ▼
                 ┌──────────────────────────┐
                 │      DECODER BLOCK 1     │
                 │                          │
                 │ Masked Self-Attention    │
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 │          ↓               │
                 │ Cross-Attention          │
                 │          ↓               │
                 │ Feed Forward Network     │
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 └──────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │      DECODER BLOCK 2     │
                 │                          │
                 │ Masked Self-Attention    │
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 │          ↓               │
                 │ Cross-Attention          │
                 │          ↓               │
                 │ Feed Forward Network     │
                 │          ↓               │
                 │ Residual + LayerNorm     │
                 └──────────────────────────┘
                                │
                                ▼
                       Linear Projection
                                │
                                ▼
                      Odia Token Probabilities
                                │
                                ▼
                     Greedy / Beam Search
                                │
                                ▼
                         Odia Translation
```

---

# 📐 Model Configuration

| Parameter           | Configuration               |
| ------------------- | --------------------------- |
| Architecture        | Transformer Encoder-Decoder |
| Framework           | PyTorch                     |
| Training            | From Scratch                |
| Source Language     | English                     |
| Target Language     | Odia                        |
| Encoder Blocks      | 2                           |
| Decoder Blocks      | 2                           |
| `d_model`           | 128                         |
| Attention Heads     | 4                           |
| FFN Dimension       | Configurable                |
| Positional Encoding | Sinusoidal                  |
| Decoder Mask        | Strict Causal Mask          |
| Normalization       | LayerNorm                   |
| Optimizer           | AdamW                       |
| Scheduler           | Warmup + Decay              |
| Loss                | Cross-Entropy               |
| PAD Loss            | Ignored                     |
| Decoding            | Greedy + Beam Search        |

---

# 📊 Dataset

The primary parallel corpus used by the project is:

**AI4Bharat Samanantar — English/Odia**

The English → Odia parallel data is processed before being used for model training.

The project does not directly feed the raw dataset into the Transformer. It first performs cleaning, validation, filtering, deduplication and deterministic splitting.

---

# 🧹 Data Preprocessing

The preprocessing pipeline is:

```text
Raw Parallel Corpus
        ↓
Unicode NFC Normalization
        ↓
Control Character Cleanup
        ↓
Whitespace Normalization
        ↓
Empty Sentence Removal
        ↓
Source / Target Validation
        ↓
Duplicate Pair Removal
        ↓
Length Filtering
        ↓
Train / Validation / Test Split
        ↓
Final Dataset
```

### Cleaning Operations

* Unicode normalization
* Invalid control-character removal
* Whitespace normalization
* Empty-pair removal
* Invalid pair removal
* Exact duplicate removal
* Excessively long sentence filtering
* Source/target consistency checks

These steps reduce unnecessary noise and improve training stability.

---

# 🔀 Dataset Splitting

The dataset is divided into:

```text
Training Set
Validation Set
Test Set
```

The split is deterministic so that repeated experiments can reproduce the same dataset partitions.

The tokenizer is trained **only using the training data**.

This prevents validation and test data from influencing tokenizer training.

---

# 🔤 Tokenization

The project uses **SentencePiece BPE tokenization**.

Separate tokenizers are used for the two languages:

```text
English
   ↓
Source SentencePiece Tokenizer
```

```text
Odia
   ↓
Target SentencePiece Tokenizer
```

Special tokens include:

```text
PAD
UNK
BOS
EOS
```

### Tokenization Pipeline

```text
Sentence
   ↓
SentencePiece BPE
   ↓
Subword Tokens
   ↓
Token IDs
   ↓
Padding
   ↓
Transformer
```

BPE helps handle:

* Rare words
* Unknown words
* Morphological variations
* Large vocabularies
* Rare Odia vocabulary

---

# 🚫 Data Leakage Prevention

The project follows this order:

```text
Raw Dataset
      ↓
Cleaning
      ↓
Deduplication
      ↓
Deterministic Split
      ↓
Train / Validation / Test
      ↓
Tokenizer trained only on TRAIN
```

Validation and test examples are not used for tokenizer training.

This keeps evaluation separate from training.

---

# 🏗️ Transformer Implementation

The Transformer is implemented directly in PyTorch rather than using a pretrained translation model.

## Encoder

Each encoder block contains:

```text
Multi-Head Self-Attention
        ↓
Residual Connection + LayerNorm
        ↓
Feed Forward Network
        ↓
Residual Connection + LayerNorm
```

The architecture contains:

```text
2 Encoder Blocks
```

## Decoder

Each decoder block contains:

```text
Masked Multi-Head Self-Attention
        ↓
Residual Connection + LayerNorm
        ↓
Encoder-Decoder Cross Attention
        ↓
Residual Connection + LayerNorm
        ↓
Feed Forward Network
        ↓
Residual Connection + LayerNorm
```

The architecture contains:

```text
2 Decoder Blocks
```

---

# 🔐 Causal Decoder Mask

The decoder uses a **strict causal mask**.

The decoder must not see future target tokens while predicting the next token.

For example:

```text
Target:

<BOS> ମୁଁ ଛାତ୍ର <EOS>
```

The decoder learns:

```text
<BOS>              → ମୁଁ
<BOS> ମୁଁ           → ଛାତ୍ର
<BOS> ମୁଁ ଛାତ୍ର      → <EOS>
```

Conceptually:

```text
          BOS   I   am   a   student
BOS        ✓    ✗   ✗    ✗      ✗
I          ✓    ✓   ✗    ✗      ✗
am         ✓    ✓   ✓    ✗      ✗
a          ✓    ✓   ✓    ✓      ✗
student    ✓    ✓   ✓    ✓      ✓
```

This prevents future-token leakage during training.

Automated tests are included to validate causal-mask behavior.

---

# 🎓 Teacher Forcing

Training uses **teacher forcing**.

For a target sentence:

```text
<BOS> ମୁଁ ଛାତ୍ର <EOS>
```

The decoder input is:

```text
<BOS> ମୁଁ ଛାତ୍ର
```

The expected output is:

```text
ମୁଁ ଛାତ୍ର <EOS>
```

Therefore:

```text
Previous Target Tokens
        ↓
Predict Next Target Token
```

This allows the model to learn next-token prediction efficiently during training.

---

# 📉 Loss Function

The training objective uses **cross-entropy loss**.

PAD tokens are ignored:

```python
CrossEntropyLoss(ignore_index=PAD_ID)
```

Therefore padded positions do not contribute to the training loss.

---

# ⚙️ Optimizer and Learning Rate

The project uses:

```text
AdamW
```

with a learning-rate schedule containing:

```text
Warmup
   ↓
Learning Rate Decay
```

Warmup helps stabilize the early stage of Transformer training.

---

# 💾 Checkpointing

Training is resumable.

Important checkpoints include:

```text
artifacts/
├── best_model.pt
└── last_checkpoint.pt
```

### `best_model.pt`

Stores the best-performing model checkpoint according to the configured validation criterion.

### `last_checkpoint.pt`

Stores the latest training state and can be used to resume interrupted training.

Checkpointing protects training progress if:

* The machine shuts down
* Training is interrupted
* The terminal is closed
* CPU training needs to continue later

---

# 🔄 Resume Training

To continue from an existing checkpoint:

```bash
python run_project.py train
```

To intentionally start a new training run:

```bash
python run_project.py train --fresh
```

---

# 🖥️ CPU-Friendly Design

The project is designed to run on systems without a dedicated GPU.

Important considerations include:

* Configurable batch size
* Dynamic padding
* Token-length filtering
* CPU-safe defaults
* Efficient dataset loading
* Checkpointing
* Resumable training
* Validation during training
* Best-checkpoint preservation

The Transformer is intentionally compact because the project focuses on demonstrating the architecture and complete NMT pipeline.

---

# 🚀 Training Profiles

The project supports configurable training profiles.

## CPU / Demo Profile

Designed for:

```text
CPU
Limited RAM
Fast experimentation
Assignment demonstration
```

Example configuration:

```text
Train:      40,000
Validation: 3,000
Test:       1,000
```

## Full Profile

A larger configuration can be used on a stronger system.

Example:

```text
Train:      100,000
Validation: 5,000
Test:       5,000
```

The actual configuration used for a particular run should be taken from the generated experiment artifacts.

---

# 📏 Token Length Filtering

Very long sequences can significantly increase Transformer memory requirements because self-attention scales with sequence length.

Therefore, examples exceeding the configured maximum token length are filtered.

The goal is to balance:

```text
Translation Coverage
        +
Training Stability
        +
CPU / RAM Usage
```

---

# 🧪 Testing

The project includes automated tests for important components.

Tests cover areas such as:

* Positional encoding
* Attention shapes
* Encoder forward pass
* Decoder forward pass
* Padding masks
* Causal masks
* Teacher forcing
* Model output dimensions
* Tokenizer behavior
* Checkpoint loading
* Translation inference

Run:

```bash
python run_project.py test
```

---

# 🧪 Sanity Check

Before starting a long training run, execute:

```bash
python run_project.py sanity
```

This verifies that the major model and data components can execute correctly.

---

# 📊 Evaluation

The trained model is evaluated on the held-out test set.

The primary automatic evaluation metric is:

```text
SacreBLEU
```

Evaluation uses **generated translations**, rather than teacher-forced outputs.

The project also performs qualitative evaluation using selected translation examples.

---

# 🔎 Decoding

The trained model supports two main decoding strategies.

## 1. Greedy Decoding

At every decoding step, the model selects the token with the highest probability.

```text
Step 1 → Highest-probability token
Step 2 → Highest-probability token
Step 3 → Highest-probability token
...
```

### Advantages

* Fast
* Low memory usage
* Suitable for CPU inference

---

## 2. Beam Search

Beam search maintains multiple candidate translations during decoding.

Example:

```text
Beam Size = 4

Candidate 1
Candidate 2
Candidate 3
Candidate 4
```

At each decoding step, candidates are expanded and scored.

```text
Input
  ↓
Encoder
  ↓
Encoder Memory
  ↓
<BOS>
  ↓
Candidate Expansion
  ↓
Candidate Scoring
  ↓
Top Beam Candidates
  ↓
Candidate Expansion
  ↓
EOS
  ↓
Best Completed Sequence
  ↓
Odia Translation
```

Beam search can produce better translations than greedy decoding for some inputs.

However:

> A larger beam size does not mathematically guarantee higher translation accuracy.

Very large beam sizes can also increase inference time and memory usage.

---

# ⭐ Recommended Demo Settings

For the **final project demonstration**, use **Beam Search** rather than Greedy Decoding.

Recommended demo procedure:

```text
Decode Mode
    ↓
Beam Search
    ↓
Increase Beam Size
    ↓
Use the maximum beam size supported by the current implementation
    ↓
Translate
```

### Why?

Beam Search evaluates multiple possible translations instead of committing to the highest-probability token at every individual step.

For the demo, using the maximum supported beam size gives the model more candidate sequences to consider and can produce a stronger/more representative generated result.

### Important Note

The maximum beam size should be used as a **demo/inference setting**, not as a claim that it always produces the highest BLEU score.

Beam size affects:

```text
Candidate Search
        +
Inference Time
        +
Memory Usage
        +
Potential Translation Quality
```

For formal benchmarking, the beam size used for evaluation should always be recorded in the experiment configuration.

---

# 📈 Evaluation Strategy

The project uses two complementary evaluation approaches.

## Automatic Evaluation

SacreBLEU is calculated on the held-out test set.

## Qualitative Evaluation

Selected examples are translated using the trained model.

Examples include:

* Normal English sentences
* Short sentences
* Longer sentences
* Challenging sentences
* Long test examples

Both source and generated target sentences can be recorded for inspection.

---

# 📁 Evaluation Artifacts

The project generates experiment artifacts such as:

```text
results/
├── run_config.json
├── data_split_stats.json
├── data_filter_stats.json
├── training_history.json
├── metrics.json
├── sample_translations.csv
├── five_required_samples.csv
└── training_curves.png
```

These artifacts make the experiment easier to reproduce, inspect and present.

---

# 🖥️ Streamlit Application

The project includes a Streamlit interface for interactive translation.

Launch it with:

```bash
python run_project.py ui
```

The application provides:

```text
┌──────────────────────────────────────────────┐
│          English → Odia Translator           │
├──────────────────────────────────────────────┤
│                                              │
│ English Input                                │
│ ┌──────────────────────────────────────────┐ │
│ │ Type or paste an English sentence...     │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ Decode Mode: Greedy / Beam Search            │
│ Beam Size: Configurable                      │
│                                              │
│              [ Translate ]                    │
│                                              │
│ Odia Translation                             │
│ ┌──────────────────────────────────────────┐ │
│ │ Generated Odia Translation               │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ Tokens: XX       Latency: X.XX sec           │
│                                              │
└──────────────────────────────────────────────┘
```

---

# ✨ UI Features

The Streamlit application provides:

* English text input
* Odia translation output
* Translate button
* Clear button
* Copy-friendly output
* Greedy decoding
* Beam-search decoding
* Configurable beam size
* Token count
* Inference latency
* Model/device information
* Vocabulary information
* Translation status
* Error handling

The UI performs **actual inference using the trained model**.

It does not use hardcoded translations.

---

# 🎬 Recommended Demo Flow

For an academic/project demonstration, the following sequence is recommended:

### Step 1 — Show the Project

Explain:

```text
English → Odia NMT
Transformer From Scratch
PyTorch
SentencePiece BPE
```

### Step 2 — Show the Architecture

Explain:

```text
Embedding
    ↓
Positional Encoding
    ↓
2 Encoder Blocks
    ↓
2 Decoder Blocks
    ↓
Linear Projection
```

Mention:

* Multi-head self-attention
* Masked self-attention
* Cross-attention
* Feed-forward network
* Residual connections
* LayerNorm

### Step 3 — Show the Training Pipeline

```text
Samanantar Dataset
        ↓
Cleaning
        ↓
Deduplication
        ↓
Train / Validation / Test
        ↓
SentencePiece
        ↓
Transformer Training
        ↓
Checkpoint
```

### Step 4 — Show Evaluation

Demonstrate:

```text
SacreBLEU
Training Curves
Translation Samples
```

### Step 5 — Run the Translation UI

Launch:

```bash
python run_project.py ui
```

### Step 6 — Use Beam Search

For the live demonstration:

```text
Decode Mode → Beam Search
Beam Size   → Maximum Supported Value
```

Then enter several English sentences and show the generated Odia translations.

This provides a stronger demonstration of the model's actual sequence-generation capability than showing only greedy decoding.

---

# 🔄 Complete Pipeline

```text
Samanantar English/Odia Data
              ↓
Unicode NFC Normalization
              ↓
Control & Whitespace Cleanup
              ↓
Exact-Pair Deduplication
              ↓
Deterministic Train / Validation / Test Split
              ↓
SentencePiece BPE
              ↓
Train-Only Tokenizer Training
              ↓
Token-Length Filtering
              ↓
Dynamic Padding
              ↓
Transformer Training
              ↓
Validation
              ↓
Best Checkpoint Selection
              ↓
SacreBLEU Evaluation
              ↓
Greedy / Beam Search
              ↓
Translation Samples
              ↓
Streamlit UI
              ↓
English → Odia Translation
```

---

# 📂 Project Structure

```text
Test7_English_Odia/
│
├── README.md
├── requirements.txt
├── run_project.py
│
├── configs/
│   ├── config.yaml
│   └── config_full.yaml
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── prepare.py
│   │   └── dataset.py
│   │
│   ├── tokenization/
│   │   └── sentencepiece.py
│   │
│   ├── model/
│   │   ├── embeddings.py
│   │   ├── positional_encoding.py
│   │   ├── attention.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── transformer.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── decoding.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── logging.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_model.py
│   ├── test_masks.py
│   ├── test_tokenizer.py
│   └── test_pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── artifacts/
│   ├── best_model.pt
│   ├── last_checkpoint.pt
│   ├── src.model
│   └── tgt.model
│
└── results/
    ├── metrics.json
    ├── training_history.json
    ├── sample_translations.csv
    └── training_curves.png
```

---

# 📦 Installation

## 1. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

### Linux / Colab

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔍 Preflight Check

Before downloading data or starting training:

```bash
python run_project.py preflight
```

This checks the environment and identifies potential configuration problems.

---

# ⚙️ Setup

Run:

```bash
python run_project.py setup
```

This prepares the required project directories and configuration.

---

# 📥 Download Dataset

If the dataset is not already available:

```bash
python run_project.py download-data
```

---

# 🧹 Prepare Dataset

Run:

```bash
python run_project.py prepare-data
```

This performs the configured:

* Cleaning
* Validation
* Deduplication
* Filtering
* Dataset splitting

---

# 🔤 Train Tokenizers

Run:

```bash
python run_project.py train-tokenizers
```

The SentencePiece tokenizers are trained using the training data.

---

# 🔍 Refilter Data

If the filtering configuration needs to be reapplied:

```bash
python run_project.py refilter
```

---

# 🧪 Run Tests

Run:

```bash
python run_project.py test
```

Optionally run:

```bash
python run_project.py sanity
```

---

# 🏋️ Train the Model

For a fresh training run:

```bash
python run_project.py train --fresh
```

The `--fresh` option explicitly starts a new training run rather than continuing from an existing checkpoint.

---

# ▶️ Resume Training

If training was interrupted:

```bash
python run_project.py train
```

The project can continue from the available checkpoint.

---

# 📊 Evaluate the Model

After training:

```bash
python run_project.py evaluate
```

Evaluation generates:

* SacreBLEU score
* Translation samples
* Required qualitative examples
* Evaluation metrics
* Evaluation reports

---

# 🖥️ Launch the UI

After a trained checkpoint is available:

```bash
python run_project.py ui
```

The Streamlit application opens in the browser.

---

# 🧭 Complete Windows Workflow

For a new setup:

```bash
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_project.py preflight
python run_project.py setup
python run_project.py download-data
python run_project.py prepare-data
python run_project.py train-tokenizers

python run_project.py test
python run_project.py sanity

python run_project.py train --fresh
python run_project.py evaluate
python run_project.py ui
```

---

# 🔁 Existing Trained Project

If the project already contains:

```text
artifacts/best_model.pt
```

training is not required just to launch the translation application.

Run:

```bash
python run_project.py ui
```

The application loads the existing trained checkpoint.

---

# 🧹 Reset Project

To reset generated data, tokenizers, checkpoints and results according to the project's reset policy:

```bash
python run_project.py reset
```

After resetting, repeat the required setup and training pipeline.

---

# 🖥️ GPU / Full Training Profile

For a stronger machine with sufficient GPU memory, the full configuration can be selected.

On Windows:

```bash
copy /Y configs\config_full.yaml configs\config.yaml
```

Then:

```bash
python run_project.py reset
python run_project.py setup
python run_project.py download-data
python run_project.py prepare-data
python run_project.py train-tokenizers
python run_project.py test
python run_project.py train --fresh
python run_project.py evaluate
```

Always verify the configuration before starting a large training run.

---

# 🐧 Linux / Colab Workflow

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_project.py preflight
python run_project.py setup
python run_project.py download-data
python run_project.py prepare-data
python run_project.py train-tokenizers
python run_project.py test
python run_project.py train --fresh
python run_project.py evaluate
python run_project.py ui
```

---

# 🧪 Reproducibility

The project records important experiment information.

Reproducibility is supported through:

* Deterministic dataset splitting
* Configurable random seeds
* Recorded configuration
* Dataset statistics
* Filtering statistics
* Training history
* Saved checkpoints
* Evaluation metrics
* Recorded decoding configuration

Important experiment information is stored under:

```text
results/
```

---

# 📋 Generated Reports

The project records information such as:

```text
Dataset statistics
Filtering statistics
Train / validation / test sizes
Tokenizer vocabulary sizes
Training loss
Validation loss
BLEU score
Decoding configuration
Model configuration
Translation examples
```

This makes the final experiment easier to inspect and present.

---

# 📦 Main Artifacts

Important model artifacts:

```text
artifacts/
├── best_model.pt
├── last_checkpoint.pt
├── src.model
└── tgt.model
```

Important evaluation artifacts:

```text
results/
├── run_config.json
├── data_split_stats.json
├── data_filter_stats.json
├── training_history.json
├── metrics.json
├── sample_translations.csv
├── five_required_samples.csv
└── training_curves.png
```

---

# 🎯 Assignment / Rubric Mapping

| Requirement                    | Implementation |
| ------------------------------ | -------------- |
| Transformer from scratch       | ✅              |
| PyTorch                        | ✅              |
| English → Odia translation     | ✅              |
| Token embeddings               | ✅              |
| Sinusoidal positional encoding | ✅              |
| Encoder                        | ✅              |
| 2 encoder blocks               | ✅              |
| Multi-head self-attention      | ✅              |
| Decoder                        | ✅              |
| 2 decoder blocks               | ✅              |
| Masked self-attention          | ✅              |
| Cross-attention                | ✅              |
| Feed-forward network           | ✅              |
| Residual connections           | ✅              |
| LayerNorm                      | ✅              |
| `d_model = 128`                | ✅              |
| 4 attention heads              | ✅              |
| Teacher forcing                | ✅              |
| Cross-entropy loss             | ✅              |
| PAD tokens ignored             | ✅              |
| AdamW                          | ✅              |
| Warmup + decay                 | ✅              |
| Strict causal mask             | ✅              |
| Causal-mask testing            | ✅              |
| Greedy decoding                | ✅              |
| Beam-search decoding           | ✅              |
| Configurable beam size         | ✅              |
| SacreBLEU                      | ✅              |
| Qualitative evaluation         | ✅              |
| Checkpointing                  | ✅              |
| Resumable training             | ✅              |
| Streamlit UI                   | ✅              |

---

# 🛡️ Engineering Safeguards

## Memory Safety

* Configurable batch sizes
* Dynamic padding
* Maximum sequence length
* CPU-friendly defaults
* Avoidance of unnecessarily large tensors

## Training Safety

* Checkpointing
* Resume support
* Best-model preservation
* Validation monitoring
* Explicit fresh-training mode

## Reproducibility

* Fixed/configurable seeds
* Deterministic dataset splitting
* Saved configuration
* Saved metrics
* Saved training history

## Data Safety

* Deduplication
* Unicode normalization
* Length filtering
* Train-only tokenizer training
* Held-out evaluation set

---

# 📌 Important Design Choice

This project intentionally uses a **compact Transformer**.

It is not intended to compete with large pretrained multilingual translation systems.

The compact architecture is designed to demonstrate:

```text
Correct Transformer Architecture
             +
Correct Training Procedure
             +
Correct Evaluation
             +
Reproducible Pipeline
             +
Working Translation Application
```

The focus is therefore on understanding and implementing the complete NMT system rather than simply using an existing pretrained translation model.

---

# ⚠️ Limitations

Because the model is trained from scratch using a relatively compact architecture, translation quality may be lower than that of large pretrained multilingual translation systems.

Potential difficult cases include:

* Very long sentences
* Rare words
* Names
* Numbers
* Complex grammar
* Ambiguous English sentences
* Rare Odia vocabulary
* Domain-specific terminology

These limitations are expected for a compact Transformer trained from scratch.

---

# 🚀 Inference Example

Example input:

```text
I am learning machine translation.
```

The inference pipeline is:

```text
English Sentence
      ↓
Source SentencePiece Tokenizer
      ↓
Token IDs
      ↓
Transformer Encoder
      ↓
Encoder Memory
      ↓
Transformer Decoder
      ↓
Greedy / Beam Search
      ↓
Target Token IDs
      ↓
Odia SentencePiece Decoder
      ↓
Odia Translation
```

The final translation is generated by the trained Transformer model.

---

# 🔬 Beam Search Decoding Flow

When Beam Search is selected:

```text
English Input
      ↓
Source Tokenization
      ↓
Transformer Encoder
      ↓
Encoder Memory
      ↓
<BOS>
      ↓
Candidate Expansion
      ↓
Candidate Scoring
      ↓
Top Beam Candidates
      ↓
Further Expansion
      ↓
EOS Detection
      ↓
Best Completed Sequence
      ↓
Odia Detokenization
      ↓
Final Translation
```

The beam size determines how many candidate sequences are maintained during decoding.

---

# 📊 What Matters Most for Translation Quality

Translation quality depends on multiple factors:

```text
Data Quality
      ↓
Data Coverage
      ↓
Tokenizer Quality
      ↓
Model Architecture
      ↓
Training Quality
      ↓
Checkpoint Selection
      ↓
Decoding Strategy
      ↓
Final Translation
```

Beam size is only one part of the overall system.

A larger beam does **not** automatically guarantee better translation accuracy.

---

# 📝 Final Project Deliverables

A complete submission should contain:

```text
Test7_English_Odia/
│
├── Source Code
├── Configuration
├── Requirements
├── README
├── Tests
├── Dataset Preparation Pipeline
├── Tokenizers
├── Trained Checkpoint
├── Evaluation Results
├── Translation Samples
└── Streamlit Application
```

If the trained checkpoint is distributed separately, place it at:

```text
artifacts/best_model.pt
```

---

# ▶️ Quick Start

If the dataset, tokenizers and trained model artifacts are already available:

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python run_project.py preflight
python run_project.py test
python run_project.py ui
```

For the final live demonstration:

```text
Open Streamlit UI
      ↓
Select Beam Search
      ↓
Set Beam Size to Maximum Supported Value
      ↓
Enter English Sentence
      ↓
Translate
      ↓
Show Generated Odia Translation
```

---

# 🏁 Final Result

The completed project provides a complete English → Odia NMT system built from scratch:

```text
                 TEST-7
                    │
                    ▼
           English → Odia NMT
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   Transformer              Evaluation
   From Scratch              Pipeline
        │                       │
        ▼                       ▼
   Trained Model          SacreBLEU +
        │                  Qualitative
        │                  Evaluation
        ▼
 Greedy / Beam Search
        │
        ▼
   Streamlit UI
        │
        ▼
 English → Odia
  Translation
```

The project therefore provides an end-to-end reproducible machine-translation pipeline covering:

**data preparation → tokenization → Transformer implementation → training → checkpointing → evaluation → decoding → inference → interactive deployment**

---

# 👨‍💻 Author

**Kushal S**

**Test-7 — English → Odia Transformer From Scratch**
