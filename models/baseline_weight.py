"""
baseline_weight.py — Class-Weighted TF-IDF Baselines for Hinglish Scam Classifier
--------------------------------------------------------------------------------
Models:
  1. TF-IDF + Logistic Regression (class_weight='balanced')
  2. TF-IDF + Linear SVM (class_weight='balanced')

Uses class weights just like bilstm.py to counteract dataset class imbalance.

Workflow:
  - Train on dataset_split/train
  - Validate on dataset_split/val
  - Test on dataset_split/test
  - Evaluate against real dataset_split/yt
  - Save all per-class and overall metrics in results/baseline_weight_metrics.json
"""

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

LABEL_MAP = {
    "1_banking_kyc_otp_fraud":             0,
    "2_upi_wallet_fraud":                  1,
    "3_investment_task_scam":              2,
    "4_digital_arrest_govt_impersonation": 3,
    "5_loan_credit_app_scams":             4,
    "6_delivery_customer_care_scams":      5,
    "7_dating_romance_sextortion":         6,
    "8_legacy_telecom_scams":              7,
}
NUM_CLASSES = len(LABEL_MAP)
ID2NAME = {v: k for k, v in LABEL_MAP.items()}

SEED = 42
TFIDF_MAX_FEATURES = 50_000


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────

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
        print(f"[WARN] Split directory does not exist: {split_dir}")
        return texts, labels, filenames

    for folder, label_id in LABEL_MAP.items():
        folder_path = split_dir / folder
        if not folder_path.exists():
            continue

        for file_path in sorted(folder_path.rglob("*.txt")):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    texts.append(text)
                    labels.append(label_id)
                    filenames.append(file_path.name)
            except Exception as e:
                print(f"[WARN] Could not read {file_path}: {e}")

    return texts, labels, filenames


# ─────────────────────────────────────────────────────────────
# 2. TOKENIZATION FOR TF-IDF
# ─────────────────────────────────────────────────────────────

def hinglish_tokenizer(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.findall(r"[\u0900-\u097F]+|[a-z0-9]+|[^\s\w]", text)


def build_tfidf():
    return TfidfVectorizer(
        analyzer="word",
        tokenizer=hinglish_tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )


# ─────────────────────────────────────────────────────────────
# 3. METRIC COMPUTATION HELPER
# ─────────────────────────────────────────────────────────────

def compute_detailed_metrics(gold, preds, model_name: str, split_name: str):
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
        "model": model_name,
        "split": split_name,
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


def print_evaluation_summary(name: str, split: str, metrics: dict):
    ov = metrics["overall"]
    print(f"\n[{name}] Evaluation on {split.upper()} set (N={ov['total_samples']:,}):")
    print(f"  Accuracy   : {ov['accuracy']:.4f}")
    print(f"  Macro-F1   : {ov['macro_f1']:.4f}")
    print(f"  Weighted-F1: {ov['weighted_f1']:.4f}")
    print(f"  Macro-Prec : {ov['macro_precision']:.4f}")
    print(f"  Macro-Rec  : {ov['macro_recall']:.4f}")


# ─────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Class-Weighted TF-IDF Baselines on dataset_split")
    parser.add_argument("--data-dir", default="dataset_split", help="Path to dataset_split directory")
    parser.add_argument("--output-dir", default="results", help="Directory to save results JSON")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_dir)
    print("\n" + "═" * 70)
    print("HINGLISH SCAM CLASSIFIER — CLASS-WEIGHTED TF-IDF BASELINES")
    print(f"Data root: {data_root}")
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

    # 2. Compute class weights (identical to bilstm.py)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=np.array(tr_labels),
    )
    print("\n⚖️  Class weights (inverse frequency, identical to bilstm.py):")
    for i in range(NUM_CLASSES):
        print(f"  [{i}] {ID2NAME[i]:<45s}  {class_weights_np[i]:.3f}")

    class_weight_dict = {i: float(class_weights_np[i]) for i in range(NUM_CLASSES)}

    # 3. Fit TF-IDF on Train set only
    print("\n🔤 Fitting TF-IDF index on train split...")
    t0 = time.time()
    vectorizer = build_tfidf()
    X_train = vectorizer.fit_transform(tr_texts)
    print(f"  TF-IDF build time: {time.time() - t0:.2f}s | Vocab size: {len(vectorizer.vocabulary_):,}")

    print("\n🔢 Transforming val, test, and yt splits...")
    X_val = vectorizer.transform(va_texts) if va_texts else None
    X_test = vectorizer.transform(te_texts) if te_texts else None
    X_yt = vectorizer.transform(yt_texts) if yt_texts else None

    # Class-weighted models
    models = {
        "Weighted_Logistic_Regression": LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="saga",
            class_weight=class_weight_dict,
            n_jobs=-1,
            random_state=SEED,
        ),
        "Weighted_Linear_SVM": LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight=class_weight_dict,
            random_state=SEED,
        ),
    }

    all_models_metrics = {}

    out_dirs = [Path(args.output_dir), Path("models/results")]
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)

    # 4. Train & Evaluate each model
    for model_name, clf in models.items():
        print("\n" + "─" * 70)
        print(f"Training {model_name} with Class Weights...")
        t_start = time.time()
        clf.fit(X_train, tr_labels)
        train_time = time.time() - t_start
        print(f"  Training finished in {train_time:.2f}s")

        # Validation evaluation
        val_metrics = None
        if X_val is not None and len(va_labels) > 0:
            val_preds = clf.predict(X_val)
            if hasattr(val_preds, "astype"):
                val_preds = val_preds.astype(int).ravel()
            val_metrics = compute_detailed_metrics(va_labels, val_preds, model_name, "validation")
            print_evaluation_summary(model_name, "val", val_metrics)

        # Test evaluation
        test_metrics = None
        if X_test is not None and len(te_labels) > 0:
            test_preds = clf.predict(X_test)
            if hasattr(test_preds, "astype"):
                test_preds = test_preds.astype(int).ravel()
            test_metrics = compute_detailed_metrics(te_labels, test_preds, model_name, "test")
            print_evaluation_summary(model_name, "test", test_metrics)

        # YT evaluation
        yt_metrics = None
        if X_yt is not None and len(yt_labels) > 0:
            yt_preds = clf.predict(X_yt)
            if hasattr(yt_preds, "astype"):
                yt_preds = yt_preds.astype(int).ravel()
            yt_metrics = compute_detailed_metrics(yt_labels, yt_preds, model_name, "yt")
            print_evaluation_summary(model_name, "yt", yt_metrics)

        single_model_json = {
            "model_name": model_name,
            "class_weights_used": class_weight_dict,
            "train_samples": len(tr_labels),
            "train_time_seconds": train_time,
            "validation_evaluation": val_metrics,
            "test_evaluation": test_metrics,
            "yt_evaluation": yt_metrics,
        }

        # Save individual model JSON
        for od in out_dirs:
            p = od / f"{model_name.lower()}_metrics.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(single_model_json, f, indent=2)
            print(f"  Saved metrics → {p}")

        all_models_metrics[model_name] = single_model_json

    # Save combined baseline_weight_metrics.json
    for od in out_dirs:
        combined_path = od / "baseline_weight_metrics.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_models_metrics, f, indent=2)
        print(f"\n💾 Saved all class-weighted baseline metrics → {combined_path}")

    # Summary table
    print("\n" + "═" * 85)
    print("CLASS-WEIGHTED BASELINE SUMMARY: TEST SET vs REAL YT DATASET")
    print("═" * 85)
    print(f"{'Model':<30} | {'Test Acc':>9} {'Test MacF1':>11} | {'YT Acc':>9} {'YT MacF1':>11}")
    print("-" * 85)
    for m_name, m_data in all_models_metrics.items():
        te_ov = m_data.get("test_evaluation", {}).get("overall", {}) if m_data.get("test_evaluation") else {}
        yt_ov = m_data.get("yt_evaluation", {}).get("overall", {}) if m_data.get("yt_evaluation") else {}
        print(
            f"{m_name:<30} | "
            f"{te_ov.get('accuracy', 0):>9.4f} {te_ov.get('macro_f1', 0):>11.4f} | "
            f"{yt_ov.get('accuracy', 0):>9.4f} {yt_ov.get('macro_f1', 0):>11.4f}"
        )
    print("═" * 85)
    print("DONE ✅\n")


if __name__ == "__main__":
    main()
