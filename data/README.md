# Data

Do not commit the downloaded corpus into this repository.

Run:
```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

The downloader uses only the Odia (`or`) Parquet files from `ai4bharat/samanantar` on Hugging Face, rather than downloading all Samanantar language data. The source dataset contains English→Odia (`en-or`) parallel sentence pairs.

# 📊 Data Directory

This directory contains the data used by the **English → Odia Neural Machine Translation** system.

The project uses the **AI4Bharat Samanantar English/Odia parallel corpus** as the primary dataset.

---

## 📂 Directory Structure

```text
data/
│
├── raw/
│   └── .gitkeep
│
├── processed/
│   └── .gitkeep
│
└── splits/
    └── .gitkeep