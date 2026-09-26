"""
roberta_weight.py — Class-Weighted RoBERTa Classifier (FacebookAI/roberta-base)
-------------------------------------------------------------------------------
Architecture:
  - Pretrained FacebookAI/roberta-base via Hugging Face transformers
  - Fully fine-tuned sequence classification model (all layers trainable)
  - Class-weighted Cross-Entropy loss (balanced inverse class frequencies)
  - Label smoothing (0.05) + validation early stopping
  - Acceleration on MPS (Apple Silicon Mac), CUDA, or CPU

Workflow:
  - Train on dataset_split/train
  - Validate on dataset_split/val (early stopping & model checkpointing)
  - Test on dataset_split/test
  - Validate against real dataset_split/yt
  - Save all per-class metrics for test and yt in results/roberta_weight_metrics.json and models/weighted_roberta.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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

# Model & Tokenizer
MODEL_NAME = "FacebookAI/roberta-base"
MAX_SEQ_LEN = 512

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 1e-2
DROPOUT = 0.15
LABEL_SMOOTHING = 0.05
SEED = 42
EARLY_STOPPING_PATIENCE = 4

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
# 2. DATASET
# ─────────────────────────────────────────────

class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_SEQ_LEN):
        print(f"  Tokenizing {len(texts):,} texts (max_len={max_len})...")
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
        )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 3. TRAINING & EVALUATION FUNCTIONS
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, class_weights=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

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
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Class-Weighted FacebookAI/roberta-base on dataset_split")
    parser.add_argument("--data-dir", default="dataset_split", help="Path to dataset_split directory")
    parser.add_argument("--output-dir", default="results", help="Directory to save results JSON")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Pretrained model identifier")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN, help=f"Max token length (default {MAX_SEQ_LEN})")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Epochs (default {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default {LR})")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = resolve_data_root(args.data_dir)
    print("\n" + "═" * 70)
    print("HINGLISH SCAM CALL CLASSIFIER — CLASS-WEIGHTED ROBERTA-BASE FINE-TUNING")
    print(f"Model: {args.model_name}")
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

    # 2. Compute balanced class weights
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(tr_labels),
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)
    print("\n⚖️  Class weights (inverse frequency balanced):")
    for i in range(NUM_CLASSES):
        print(f"  [{i}] {ID2NAME[i]:<45s}  {class_weights_np[i]:.3f}")

    # 3. Tokenizer
    print(f"\n🔤 Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 4. Datasets & DataLoaders
    print("\n📦 Encoding datasets...")
    tr_ds = ScamDataset(tr_texts, tr_labels, tokenizer, max_len=args.max_seq_len)
    va_ds = ScamDataset(va_texts, va_labels, tokenizer, max_len=args.max_seq_len)
    te_ds = ScamDataset(te_texts, te_labels, tokenizer, max_len=args.max_seq_len)
    yt_ds = ScamDataset(yt_texts, yt_labels, tokenizer, max_len=args.max_seq_len)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    yt_loader = DataLoader(yt_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 5. Model instantiation & full fine-tuning
    print(f"\n🧠 Loading pretrained {args.model_name} for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2NAME,
        label2id=LABEL_MAP,
        classifier_dropout=DROPOUT,
    ).to(DEVICE)

    # Ensure all parameters are trainable (fully fine-tuned)
    for p in model.parameters():
        p.requires_grad = True

    n_params = count_params(model)
    print(f"🧠 Trainable model parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = args.epochs * len(tr_loader)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Training loop with early stopping & class weights
    print(f"\n🚀 Fine-tuning with Class Weights for up to {args.epochs} epochs (Patience={EARLY_STOPPING_PATIENCE})...\n")
    best_acc, best_state = 0.0, None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, scheduler, class_weights=class_weights)
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

    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / "roberta_weight_best.pt"
    torch.save(best_state if best_state else model.state_dict(), ckpt_path)
    print(f"\n💾 Best model checkpoint saved → {ckpt_path}")

    # Also save HuggingFace format checkpoint
    hf_save_dir = cache_dir / "roberta_weight_base_best"
    model.save_pretrained(hf_save_dir)
    tokenizer.save_pretrained(hf_save_dir)
    print(f"💾 HuggingFace model saved → {hf_save_dir}")

    # 7. Evaluation on Test Set
    print("\n" + "═" * 70)
    print("FINAL TEST EVALUATION")
    print("═" * 70)
    te_loss, te_preds, te_gold = evaluate_model(model, te_loader)
    test_metrics = compute_metrics_dict(te_gold, te_preds, "test", te_loss)

    print(classification_report(te_gold, te_preds, target_names=[ID2NAME[i] for i in range(NUM_CLASSES)], digits=4))
    print(f"🎯 Test Accuracy: {test_metrics['overall']['accuracy']:.4f} | Test Macro-F1: {test_metrics['overall']['macro_f1']:.4f}")

    # 8. Evaluation on Real YT Set
    print("\n" + "═" * 70)
    print("REAL YOUTUBE (YT) DATASET EVALUATION")
    print("═" * 70)
    yt_loss, yt_preds, yt_gold = evaluate_model(model, yt_loader)
    yt_metrics = compute_metrics_dict(yt_gold, yt_preds, "yt", yt_loss)

    print(classification_report(yt_gold, yt_preds, target_names=[ID2NAME[i] for i in range(NUM_CLASSES)], digits=4))
    print(f"🎯 YT Accuracy: {yt_metrics['overall']['accuracy']:.4f} | YT Macro-F1: {yt_metrics['overall']['macro_f1']:.4f}")

    # 9. Save single JSON with both Test and YT metrics
    single_model_json = {
        "model_name": "RoBERTa_Base_Weighted",
        "class_weights_used": {i: float(class_weights_np[i]) for i in range(NUM_CLASSES)},
        "train_samples": len(tr_labels),
        "validation_samples": len(va_labels),
        "test_evaluation": test_metrics,
        "yt_evaluation": yt_metrics,
    }

    out_dirs = [Path(args.output_dir), Path("models/results"), Path("models")]
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)
        res_file = od / "roberta_weight_metrics.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(single_model_json, f, indent=2)
        print(f"💾 Saved Weighted RoBERTa Base metrics → {res_file}")

    # Also save weighted_roberta.json in models/
    with open("models/weighted_roberta.json", "w", encoding="utf-8") as f:
        json.dump(single_model_json, f, indent=2)
    print("💾 Saved Weighted RoBERTa metrics → models/weighted_roberta.json")

    print("\nDONE ✅\n")


if __name__ == "__main__":
    main()
