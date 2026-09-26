# JAAL-10K: Codebase & Experimentation Suite

Welcome to the **JAAL-10K** codebase repository. This repository contains the complete source code, preprocessing pipelines, model architectures, evaluation benchmarks, and analytical tools for generating, processing, auditing, and classifying code-mixed (Hinglish) fraud and scam call conversations.

---

## 📁 Repository Structure

```
JAAL-10k-code/
├── README.md                              # Comprehensive codebase documentation
├── requirements.txt                       # Global Python dependencies
├── .gitignore                             # Git ignore rules for data, cache, and envs
│
├── generation/                            # Multi-turn synthetic dialogue generation
│   ├── caller.py                          # Caller LLM simulation agent (Hinglish scammer persona)
│   ├── receiver.py                        # Receiver/Victim LLM simulation agent
│   ├── generate.py                        # Multi-turn conversational orchestrator
│   ├── extract.py                         # Seed fraud context extractor
│   ├── requirements.txt                   # Generation-specific dependencies (google-genai, etc.)
│   ├── fraud_context.txt                  # Extracted scam seed contexts
│   └── fraud_call.file                    # Source raw seed templates
│
├── human_validation/                      # Human annotation validation study
│   └── annotation_analysis.ipynb          # Jupyter notebook for IAA, Kappa, realism, and confidence
|   |--- results                           # contains the annotations
│
├── models/                                # Model training, baselines, and evaluation
│   ├── baselines.py                       # TF-IDF + Logistic Regression / Linear SVM baselines
│   ├── baseline_weight.py                 # Class-weighted TF-IDF baselines
│   ├── bilstm.py                          # BiLSTM neural network with word embeddings
│   ├── encoder.py                         # Transformer encoder fine-tuning pipeline
│   ├── encoder_weight.py                  # Class-weighted transformer encoder fine-tuning
│   ├── roberta.py                         # RoBERTa / XLM-RoBERTa sequence classifier
│   ├── roberta_weight.py                  # Class-weighted RoBERTa sequence classifier
│   ├── modernbert.py                      # ModernBERT classifier (long-context architecture)
│   ├── modernbert_weight.py               # Class-weighted ModernBERT classifier
│   ├── zero_shot.py                       # LLM zero-shot classification evaluation
│   ├── jev_eval.py                        # Joint Evaluation & Verification (LLM-as-a-judge)
|   |--- hingbert.py
│   └── requirements.txt                   # Modeling dependencies (PyTorch, Transformers, etc.)
│
├── scripts/                               # Data engineering, auditing, and analysis
│   ├── compute_annotation_metrics.py      # Statistical evaluation of annotators vs ground truth
│   ├── generate_and_run_notebook.py       # Programmatic generator for annotation notebook
│   │
│   ├── pipeline/                          # Data splitting, hygiene, and scaling pipeline
│   │   ├── split.py                       # Stratified train / validation / test splitter
│   │   ├── human_split.py                 # Sampling and preparation for human validation
│   │   ├── leak_check.py                  # Train-test contamination and near-duplicate audit
│   │   ├── reclassify.py                  # Scam category remapping and taxonomy alignment
│   │   ├── relabel.py                     # Dialogue relabeling and taxonomy standardization
│   │   └── upscale.py                     # Class balancing and dataset upscaling
│   │
│   └── analysis/                          # Linguistic, structural, and benchmark analysis
│       ├── count_analysis.py              # Turn lengths, token/word metrics, duplicate clusters
│       ├── generate_analysis_reports.py   # Automated report and figure generation
│       ├── hinglish_analysis.py           # LLM token-level language identification & CMI
│       ├── oov_bpe.py                     # BPE tokenizer fertility and Out-Of-Vocabulary audit
│       └── yt_analysis.py                 # Comparative analysis against real-world YouTube calls
│
└── yt_org/                                # Real-world YouTube benchmark ingestion
    ├── parse.py                           # YouTube transcript parser and speaker segmenter
    └── filter.py                          # Transcript cleaning, quality filtering, and curation
```

---

## 🔍 Module & File Descriptions

### 1. Generation Pipeline (`generation/`)

Tools to generate synthetic, multi-turn scam phone calls using paired LLM agents in Hinglish.

* **`caller.py`**: Implements the caller persona agent using `google-genai` (Gemma / Gemini models). Enforces scam context, Hinglish code-switching, tone modulation, and urgency ramping across conversational turns.
* **`receiver.py`**: Implements the receiver/victim persona agent. Simulates natural human reactions (hesitation, confusion, compliance, skepticism) across turns.
* **`generate.py`**: Main driver script orchestrating multi-turn dialogues between `caller` and `receiver`, applying rate limiting, and outputting formatted conversation transcripts into text files.
* **`extract.py`**: Preprocesses raw scam message seeds (`fraud_call.file`) into standardized prompt contexts (`fraud_context.txt`).
* **`fraud_context.txt` & `fraud_call.file`**: Seed inputs and prompt context lines used by the generation pipeline.
* **`requirements.txt`**: Minimal requirements file for running LLM generation via Google GenAI APIs.

---

### 2. Machine Learning & Transformer Models (`models/`)

Supervised classifiers, deep learning baselines, transformer architectures, and zero-shot LLM evaluation scripts.

* **`baselines.py`**: Classical machine learning baselines using word- and char-ngram TF-IDF vectorization with Logistic Regression and Linear Support Vector Machines (LinearSVC). Evaluates on test and real-world YouTube sets.
* **`baseline_weight.py`**: Extends classical baselines with inverse-frequency class weighting (`balanced`) to handle class imbalance across the 8 fraud categories.
* **`bilstm.py`**: PyTorch implementation of a Bidirectional Long Short-Term Memory (BiLSTM) network with trainable embeddings, dropout, and multi-class classification head.
* **`encoder.py`**: Transformer encoder fine-tuning script supporting HuggingFace encoder backbones with AdamW optimizer, warmup schedules, and evaluation checkpoints.
* **`encoder_weight.py`**: Weighted cross-entropy variant of the transformer encoder pipeline to address class distribution skew.
* **`roberta.py`**: Fine-tuning pipeline specifically tailored for RoBERTa and multilingual XLM-RoBERTa backbones on code-mixed conversations.
* **`roberta_weight.py`**: Class-weighted loss implementation for RoBERTa/XLM-RoBERTa models.
* **`modernbert.py`**: Fine-tuning pipeline leveraging ModernBERT architectures with support for extended sequence lengths (Flash Attention, rotary embeddings).
* **`modernbert_weight.py`**: Class-weighted implementation of ModernBERT fine-tuning.
* **`zero_shot.py`**: Zero-shot prompting benchmark querying foundation LLMs to classify multi-turn conversations into the 8 scam categories without parameter updates.
* **`jev_eval.py`**: Joint Evaluation & Verification (JEV) framework using an LLM-as-a-judge paradigm to evaluate prediction credibility and resolve edge cases.
* **`requirements.txt`**: PyTorch, Hugging Face Transformers, Scikit-learn, and CUDA support dependencies.

---

### 3. Pipeline & Data Hygiene (`scripts/pipeline/`)

Scripts for dataset splitting, integrity verification, relabeling, and leakage prevention.

* **`split.py`**: Generates reproducible, stratified train (70%), validation (15%), and test (15%) splits across the 8 scam categories with fixed random seeds.
* **`human_split.py`**: Samples balanced subsets for human evaluation and blind annotator packages.
* **`leak_check.py`**: Rigorous data leakage detection. Tests for:
  1. Exact transcript string matches between train and test/val.
  2. Normalized string matches (case-folded, punctuation-stripped, whitespace-collapsed).
  3. Near-duplicate detection using TF-IDF feature vectors and Cosine Similarity thresholds ($\ge 0.85$).
* **`reclassify.py`**: Audits and re-assigns samples whose content aligns with updated scam taxonomy definitions.
* **`relabel.py`**: Normalizes folder structures and renames files to conform to canonical class identifiers.
* **`upscale.py`**: Balances under-represented scam categories through systematic data synthesis and augmentation.

---

### 4. Corpus Analysis & Linguistics (`scripts/analysis/`)

Quantitative analysis of conversational structure, vocabulary, tokenization efficiency, and code-mixing.

* **`count_analysis.py`**: Deterministic statistical analysis of the corpus. Calculates:
  * Conversation length (character, word, and token counts).
  * Turn counts and vocabulary diversity (Type-Token Ratio / TTR).
  * Duplicate and near-duplicate cluster detection via connected components.
  * Scam category confusion analysis.
* **`generate_analysis_reports.py`**: Automated compilation of analytical outputs into publication-ready figures, markdown summaries, and reports.
* **`hinglish_analysis.py`**: Token-level language identification using LLM classification to assign tokens into `EN` (English), `HI` (Romanized Hindi), `AMB` (Ambiguous), and `OTHER`. Computes Code-Mixing Index (CMI) and language distribution per category.
* **`oov_bpe.py`**: Byte-Pair Encoding (BPE) fertility and Out-Of-Vocabulary (OOV) analysis comparing tokenizer subword segmentation across standard English models vs multilingual models.
* **`yt_analysis.py`**: Comparative distribution analysis contrasting synthetic conversations against real-world YouTube fraud calls.

---

### 5. Human Validation & Benchmarking (`human_validation/` & `scripts/`)

Tools for measuring annotator reliability, agreement, and realism.

* **`scripts/compute_annotation_metrics.py`**: Computes Inter-Annotator Agreement (Cohen's Kappa $\kappa$, raw agreement percentage), per-class agreement, annotator accuracy against ground truth, confidence breakdown (High, Medium, Low), and Mann–Whitney U / Chi-Square realism significance tests.
* **`scripts/generate_and_run_notebook.py`**: Programmatically generates and executes `human_validation/annotation_analysis.ipynb`.
* **`human_validation/annotation_analysis.ipynb`**: Complete executable notebook containing the analysis, confusion matrix plots, and realism evaluations.

---

### 6. YouTube Real-World Benchmark Scraper (`yt_org/`)

Processing and curation pipeline for real-world scam call recordings scraped from YouTube.

* **`parse.py`**: Extracts dialogue turns and speaker timestamps from raw YouTube transcript dumps and converts them into standardized multi-turn conversation format.
* **`filter.py`**: Filters out non-conversational noise, sponsor segments, and non-fraudulent content to preserve high-fidelity scam calls.

---

## 🏷️ The 8 Canonical Scam Classes

All models and scripts map to the following standardized 8-class taxonomy:

| ID | Class Label | Description |
|:--:|:------------|:------------|
| `1` | `1_banking_kyc_otp_fraud` | Bank impersonation, expired card alerts, fake KYC updates, OTP theft |
| `2` | `2_upi_wallet_fraud` | Fake QR codes, reverse-charge cashback scams, payment app fraud |
| `3` | `3_investment_task_scam` | Telegram/WhatsApp part-time jobs, YouTube rating tasks, crypto schemes |
| `4` | `4_digital_arrest_govt_impersonation` | Fake police/CBI/TRAI officers, money laundering allegations, Skype interrogation |
| `5` | `5_loan_credit_app_scams` | Predatory instant loan apps, blackmail, harassment, unapproved disbursement |
| `6` | `6_delivery_customer_care_scams` | Undelivered courier packages, custom duty demands, spoofed customer support |
| `7` | `7_dating_romance_sextortion` | Matrimonial fraud, video call recording blackmail, romance investment trap |
| `8` | `8_legacy_telecom_scams` | SIM deactivation threats, lottery SMS, 5G tower installation offers |

---

## 🚀 Getting Started

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running Data Hygiene & Leakage Checks

Ensure no data contamination exists between train and test splits:

```bash
python scripts/pipeline/leak_check.py
```

### 3. Training Classification Baselines

Train TF-IDF Logistic Regression and Linear SVM models:

```bash
python models/baselines.py
```

Or run class-weighted variants:

```bash
python models/baseline_weight.py
```

### 4. Fine-Tuning Transformer Encoders

Fine-tune RoBERTa or ModernBERT on the dataset splits:

```bash
# Fine-tune RoBERTa
python models/roberta.py

# Fine-tune ModernBERT
python models/modernbert.py
```

### 5. Running Statistical & Linguistic Analysis

Run length, vocabulary, and duplicate template analysis:

```bash
python scripts/analysis/count_analysis.py --dataset /path/to/dataset --output /path/to/output
```

Run token-level Hinglish code-mixing analysis:

```bash
python scripts/analysis/hinglish_analysis.py
```

---

## 🛡️ License & Ethics

This codebase is provided for research and academic purposes studying cyber fraud patterns, code-switched NLP, and defensive AI systems.
