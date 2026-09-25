"""
bilstm.py — Hinglish Scam Call Classifier (BiLSTM + Attention Pooling + Word/Char TF-IDF)
----------------------------------------------------------------------------------------
Architecture:
  - Custom BPE tokenizer trained on train set
  - Hybrid architecture: BiLSTM + Attention Pooling + Word/Char TF-IDF lexical features -> Fusion -> Classifier
  - Class-weighted loss + label smoothing + early stopping on validation set

Workflow:
  - Train on dataset_split/train
  - Validate on dataset_split/val (early stopping & model checkpointing)
  - Test on dataset_split/test
  - Validate against real dataset_split/yt
  - Save all per-class metrics for test and yt in a single JSON file.
"""

import argparse
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LABEL_MAP = {
    "1_banking_kyc_otp_fraud": 0,
    "2_upi_wallet_fraud": 1,
    "3_investment_task_scam": 2,
    "4_digital_arrest_govt_impersonation": 3,
    "5_loan_credit_app_scams": 4,
    "6_delivery_customer_care_scams": 5,
    "7_dating_romance_sextortion": 6,
    "8_legacy_telecom_scams": 7,
}
NUM_CLASSES = len(LABEL_MAP)
ID2NAME = {v: k for k, v in LABEL_MAP.items()}

# Tokenizer (for the sequence/BiLSTM branch)
VOCAB_SIZE = 8000
MAX_SEQ_LEN = 2048

# TF-IDF (for the lexical branch)
TFIDF_WORD_MAX_FEATURES = 3000
TFIDF_WORD_NGRAM_RANGE = (1, 2)
TFIDF_CHAR_MAX_FEATURES = 3000
TFIDF_CHAR_NGRAM_RANGE = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_PROJ_DIM = 192

# Model hyperparameters
EMBED_DIM = 128
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
ATTN_DIM = 64
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05

# Training defaults
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 2e-2
SEED = 42
EARLY_STOPPING_PATIENCE = 6

NUM_ENCODE_WORKERS = max(1, cpu_count() - 1)

# Device selection (CUDA -> MPS (Mac) -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def resolve_data_root(data_dir: str = "dataset_split") -> Path:
    candidates = [
        Path(data_dir),
        Path("..") / data_dir,
        Path(__file__).resolve().parent.parent / data_dir,
        Path(__file__).resolve().parent / data_dir,
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()
    return Path(data_dir).resolve()


def load_split(split_dir: Path):
    texts, labels, filenames = [], [], []
    split_dir = Path(split_dir)
    if not split_dir.exists():
        print(f"  [WARN] Split directory does not exist: {split_dir}")
        return texts, labels, filenames

    for folder, label_id in LABEL_MAP.items():
        folder_path = split_dir / folder
        if not folder_path.exists():
            continue

        for fpath in sorted(folder_path.rglob("*.txt")):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    texts.append(text)
                    labels.append(label_id)
                    filenames.append(fpath.name)
            except Exception as e:
                print(f"    [WARN] could not read {fpath}: {e}")

    return texts, labels, filenames


# ─────────────────────────────────────────────
# 2. TOKENIZATION & BPE
# ─────────────────────────────────────────────

def hinglish_pretokenize(text: str):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.findall(r"[\u0900-\u097F]+|[a-z0-9]+|[^\s\w]", text)


class BPETokenizer:
    SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []
        self._id2tok: dict[int, str] = {}
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self._word_cache: dict[str, list[int]] = {}

    def _rebuild_ranks(self):
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._word_cache.clear()

    def train(self, texts: list[str], min_freq: int = 2):
        print("Training BPE tokenizer on training data...")
        self.vocab = dict(self.SPECIAL)

        word_freq = Counter()
        for text in texts:
            for tok in hinglish_pretokenize(text):
                word_freq[tok] += 1

        word_splits = {}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                word_splits[word] = list(word) + ["</w>"]

        for chars in word_splits.values():
            for ch in chars:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        target = self.vocab_size - len(self.SPECIAL)
        while len(self.vocab) < target:
            pair_freq = Counter()
            for word, chars in word_splits.items():
                freq = word_freq[word]
                for a, b in zip(chars, chars[1:]):
                    pair_freq[(a, b)] += freq
            if not pair_freq:
                break
            best = pair_freq.most_common(1)[0][0]
            merged = best[0] + best[1]
            self.merges.append(best)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

            new_splits = {}
            for word, chars in word_splits.items():
                new_chars, i = [], 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_splits[word] = new_chars
            word_splits = new_splits

        self._id2tok = {v: k for k, v in self.vocab.items()}
        self._rebuild_ranks()
        print(f"  Vocab size: {len(self.vocab):,}")

    def _apply_merges(self, chars: list[str]) -> list[str]:
        while True:
            best_rank, best_pos = None, None
            for i in range(len(chars) - 1):
                rank = self.merge_ranks.get((chars[i], chars[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_rank is None:
                break
            a, b = chars[best_pos], chars[best_pos + 1]
            chars = chars[:best_pos] + [a + b] + chars[best_pos + 2:]
        return chars

    def _encode_word(self, raw_tok: str, unk: int) -> list[int]:
        cached = self._word_cache.get(raw_tok)
        if cached is not None:
            return cached
        chars = list(raw_tok) + ["</w>"]
        subwords = self._apply_merges(chars)
        ids = [self.vocab.get(sw, unk) for sw in subwords]
        self._word_cache[raw_tok] = ids
        return ids

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
        ids = [self.SPECIAL["[CLS]"]]
        unk = self.SPECIAL["[UNK]"]
        for raw_tok in hinglish_pretokenize(text):
            ids.extend(self._encode_word(raw_tok, unk))
        ids.append(self.SPECIAL["[SEP]"])
        ids = ids[:max_len]
        ids += [self.SPECIAL["[PAD]"]] * (max_len - len(ids))
        return ids

    def save(self, path: str):
        data = {"vocab": self.vocab, "merges": self.merges}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, vocab_size: int = VOCAB_SIZE):
        obj = cls(vocab_size)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj.vocab = data["vocab"]
        obj.merges = [tuple(m) for m in data["merges"]]
        obj._id2tok = {v: k for k, v in obj.vocab.items()}
        obj._rebuild_ranks()
        return obj


_worker_tokenizer: Optional[BPETokenizer] = None


def _init_encode_worker(tokenizer: BPETokenizer):
    global _worker_tokenizer
    _worker_tokenizer = tokenizer


def _encode_one(text: str) -> list[int]:
    return _worker_tokenizer.encode(text)


def encode_texts_parallel(
    tokenizer: BPETokenizer,
    texts: list[str],
    n_workers: int = NUM_ENCODE_WORKERS,
    chunksize: int = 64,
    min_texts_for_parallel: int = 200,
) -> list[list[int]]:
    if n_workers <= 1 or len(texts) < min_texts_for_parallel:
        return [tokenizer.encode(t) for t in texts]

    with Pool(n_workers, initializer=_init_encode_worker, initargs=(tokenizer,)) as pool:
        return pool.map(_encode_one, texts, chunksize=chunksize)


# ─────────────────────────────────────────────
# 3. COMBINED TF-IDF VECTORIZER
# ─────────────────────────────────────────────

class CombinedTfidfVectorizer:
    def __init__(
        self,
        word_max_features: int = TFIDF_WORD_MAX_FEATURES,
        word_ngram_range: tuple = TFIDF_WORD_NGRAM_RANGE,
        char_max_features: int = TFIDF_CHAR_MAX_FEATURES,
        char_ngram_range: tuple = TFIDF_CHAR_NGRAM_RANGE,
        min_df: int = TFIDF_MIN_DF,
    ):
        self.word_vec = TfidfVectorizer(
            tokenizer=hinglish_pretokenize,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            min_df=min_df,
            sublinear_tf=True,
            lowercase=True,
        )

    def fit(self, texts: list[str]):
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        return self

    def transform(self, texts: list[str]):
        w = self.word_vec.transform(texts)
        c = self.char_vec.transform(texts)
        return hstack([w, c]).tocsr()

    @property
    def combined_dim(self) -> int:
        return len(self.word_vec.vocabulary_) + len(self.char_vec.vocabulary_)


# ─────────────────────────────────────────────
# 4. DATASET & MODEL
# ─────────────────────────────────────────────

class ScamDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer: BPETokenizer,
        tfidf_vectorizer: CombinedTfidfVectorizer,
        n_workers: int = NUM_ENCODE_WORKERS,
    ):
        self.ids = encode_texts_parallel(tokenizer, texts, n_workers=n_workers)
        self.tfidf = tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ids[idx], dtype=torch.long),
            torch.tensor(self.tfidf[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


class HybridScamClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        tfidf_dim: int = TFIDF_WORD_MAX_FEATURES + TFIDF_CHAR_MAX_FEATURES,
        tfidf_proj_dim: int = TFIDF_PROJ_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_pool = AttentionPooling(lstm_hidden * 2)

        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, tfidf_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_dim = lstm_hidden * 2 + tfidf_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, tfidf_vec: torch.Tensor):
        pad_mask = (token_ids == 0)
        e = self.emb_drop(self.embedding(token_ids))
        lstm_out, _ = self.lstm(e)
        seq_feat, attn_weights = self.attn_pool(lstm_out, pad_mask)
        lex_feat = self.tfidf_proj(tfidf_vec)
        fused = torch.cat([seq_feat, lex_feat], dim=-1)
        logits = self.classifier(fused)
        return logits, attn_weights


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION FUNCTIONS
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, class_weights: Optional[torch.Tensor]):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, tfidf_b, yb in loader:
        xb, tfidf_b, yb = xb.to(DEVICE), tfidf_b.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(xb, tfidf_b)
        loss = F.cross_entropy(
            logits, yb,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for xb, tfidf_b, yb in loader:
        xb, tfidf_b, yb = xb.to(DEVICE), tfidf_b.to(DEVICE), yb.to(DEVICE)
        logits, _ = model(xb, tfidf_b)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(yb.cpu().tolist())

    n = len(all_labels)
    loss = total_loss / n if n else 0.0
    return loss, all_preds, all_labels


def compute_metrics_dict(gold, preds, split_name: str, loss: float = 0.0):
    acc = float(accuracy_score(gold, preds))
    macro_f1 = float(f1_score(gold, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gold, preds, average="weighted", zero_division=0))
    macro_prec = float(precision_score(gold, preds, average="macro", zero_division=0))
    weighted_prec = float(precision_score(gold, preds, average="weighted", zero_division=0))
    macro_rec = float(recall_score(gold, preds, average="macro", zero_division=0))
    weighted_rec = float(recall_score(gold, preds, average="weighted", zero_division=0))

    target_names = [ID2NAME[i] for i in range(NUM_CLASSES)]
    report_dict = classification_report(
        gold,
        preds,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(gold, preds, labels=list(range(NUM_CLASSES)))

    per_class = {}
    for i in range(NUM_CLASSES):
        class_name = ID2NAME[i]
        c_stats = report_dict.get(class_name, {})
        correct_i = int(cm[i, i])
        total_i = int(cm[i].sum())
        acc_i = float(correct_i / total_i) if total_i > 0 else 0.0

        per_class[class_name] = {
            "class_id": i,
            "precision": float(c_stats.get("precision", 0.0)),
            "recall": float(c_stats.get("recall", 0.0)),
            "f1_score": float(c_stats.get("f1-score", 0.0)),
            "support": int(c_stats.get("support", 0)),
            "accuracy": acc_i,
            "correct": correct_i,
            "total": total_i,
        }

    return {
        "split": split_name,
        "loss": float(loss),
        "overall": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "weighted_precision": weighted_prec,
            "macro_recall": macro_rec,
            "weighted_recall": weighted_rec,
            "total_samples": len(gold),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate BiLSTM on dataset_split")
    parser.add_argument("--data-dir", default="dataset_split", help="Path to dataset_split directory")
    parser.add_argument("--output-dir", default="results", help="Directory to save results JSON")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Epochs (default {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default {BATCH_SIZE})")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = resolve_data_root(args.data_dir)
    print("\n" + "═" * 70)
    print("HINGLISH SCAM CALL CLASSIFIER — BILSTM + ATTENTION + TFIDF")
    print(f"Device: {DEVICE} | Data root: {data_root}")
    print("═" * 70)

    # 1. Load splits
    print("\n📂 Loading dataset splits...")
    tr_texts, tr_labels, _ = load_split(data_root / "train")
    va_texts, va_labels, _ = load_split(data_root / "val")
    te_texts, te_labels, _ = load_split(data_root / "test")
    yt_texts, yt_labels, _ = load_split(data_root / "yt")

    print(f"  Train samples: {len(tr_texts):,}")
    print(f"  Val   samples: {len(va_texts):,}")
    print(f"  Test  samples: {len(te_texts):,}")
    print(f"  YT    samples: {len(yt_texts):,}")

    if not tr_texts:
        raise RuntimeError(f"No train data found under {data_root / 'train'}")

    # Class weights for training
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(tr_labels),
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

    # 2. Tokenizer & TF-IDF fitted on Train only
    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok_cache = cache_dir / "bilstm_bpe_tokenizer.json"
    if tok_cache.exists():
        print("\n🔤 Loading cached BPE tokenizer...")
        tokenizer = BPETokenizer.load(str(tok_cache), VOCAB_SIZE)
    else:
        print("\n🔤 Training BPE tokenizer on train split...")
        tokenizer = BPETokenizer(VOCAB_SIZE)
        tokenizer.train(tr_texts)
        tokenizer.save(str(tok_cache))

    actual_vocab = len(tokenizer.vocab)

    tfidf_cache = cache_dir / "bilstm_tfidf_v2.pkl"
    if tfidf_cache.exists():
        print("\n📊 Loading cached word+char TF-IDF vectorizer...")
        with open(tfidf_cache, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
    else:
        print("\n📊 Fitting word+char TF-IDF vectorizer on train split...")
        tfidf_vectorizer = CombinedTfidfVectorizer()
        tfidf_vectorizer.fit(tr_texts)
        with open(tfidf_cache, "wb") as f:
            pickle.dump(tfidf_vectorizer, f)

    actual_tfidf_dim = tfidf_vectorizer.combined_dim

    # 3. Datasets & DataLoaders
    print(f"\n📦 Encoding datasets (using up to {NUM_ENCODE_WORKERS} worker process(es))...")
    t_enc0 = time.time()
    tr_ds = ScamDataset(tr_texts, tr_labels, tokenizer, tfidf_vectorizer)
    va_ds = ScamDataset(va_texts, va_labels, tokenizer, tfidf_vectorizer)
    te_ds = ScamDataset(te_texts, te_labels, tokenizer, tfidf_vectorizer)
    yt_ds = ScamDataset(yt_texts, yt_labels, tokenizer, tfidf_vectorizer)
    print(f"  Encoding took {time.time() - t_enc0:.1f}s")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    yt_loader = DataLoader(yt_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Model instantiation
    model = HybridScamClassifier(
        vocab_size=actual_vocab,
        embed_dim=EMBED_DIM,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        tfidf_dim=actual_tfidf_dim,
        tfidf_proj_dim=TFIDF_PROJ_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = count_params(model)
    print(f"\n🧠 Model parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = args.epochs * len(tr_loader)
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Training loop with early stopping on validation set
    print(f"\n🚀 Training for up to {args.epochs} epochs (Early stopping patience={EARLY_STOPPING_PATIENCE})...\n")
    best_acc, best_state = 0.0, None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, scheduler, class_weights)
        va_loss, va_preds, va_gold = evaluate_model(model, va_loader)
        va_acc = sum(p == g for p, g in zip(va_preds, va_gold)) / len(va_gold) if va_gold else 0.0
        elapsed = time.time() - t0

        marker = ""
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " ✅ (Best)"
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.3f} | "
            f"Val Loss={va_loss:.4f} Acc={va_acc:.3f} | "
            f"({elapsed:.1f}s){marker}"
        )

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch} epochs (Best Val Acc: {best_acc:.4f})")
            break

    # Load best model weights
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    ckpt_path = cache_dir / "bilstm_best.pt"
    torch.save(best_state if best_state else model.state_dict(), ckpt_path)
    print(f"\n💾 Best model checkpoint saved → {ckpt_path}")

    # 6. Evaluation on Test Set
    print("\n" + "═" * 70)
    print("FINAL TEST EVALUATION")
    print("═" * 70)
    te_loss, te_preds, te_gold = evaluate_model(model, te_loader)
    test_metrics = compute_metrics_dict(te_gold, te_preds, "test", te_loss)

    print(classification_report(te_gold, te_preds, target_names=[ID2NAME[i] for i in range(NUM_CLASSES)], digits=4))
    print(f"🎯 Test Accuracy: {test_metrics['overall']['accuracy']:.4f} | Test Macro-F1: {test_metrics['overall']['macro_f1']:.4f}")

    # 7. Evaluation on Real YT Set
    print("\n" + "═" * 70)
    print("REAL YOUTUBE (YT) DATASET EVALUATION")
    print("═" * 70)
    yt_loss, yt_preds, yt_gold = evaluate_model(model, yt_loader)
    yt_metrics = compute_metrics_dict(yt_gold, yt_preds, "yt", yt_loss)

    print(classification_report(yt_gold, yt_preds, target_names=[ID2NAME[i] for i in range(NUM_CLASSES)], digits=4))
    print(f"🎯 YT Accuracy: {yt_metrics['overall']['accuracy']:.4f} | YT Macro-F1: {yt_metrics['overall']['macro_f1']:.4f}")

    # 8. Save single JSON with both Test and YT metrics
    single_model_json = {
        "model_name": "BiLSTM_Attention_TFIDF",
        "train_samples": len(tr_labels),
        "validation_samples": len(va_labels),
        "test_evaluation": test_metrics,
        "yt_evaluation": yt_metrics,
    }

    out_dirs = [Path(args.output_dir), Path("models/results")]
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)
        res_file = od / "bilstm_metrics.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(single_model_json, f, indent=2)
        print(f"💾 Saved BiLSTM metrics → {res_file}")

    print("\nDONE ✅\n")


if __name__ == "__main__":
    main()
