"""
jev_eval.py — Zero-Shot Scam Call Classification with Jev Model
-----------------------------------------------------------------
Evaluates TypeSafe AI's Jev model as a true zero-shot decision engine on:
  1. Synthetic test set (dataset_split/test/)
  2. External YouTube / YT set (dataset_split/yt/)

Key Features:
  - Uses Jev's native structured decision API (POST /api/v1/decisions).
  - Evaluates all 8 standardized scam categories with calibrated criteria.
  - Zero-shot: Category definitions & guidelines only, no few-shot or training examples.
  - Context handling:
      * Raw conversation text preserved 100% verbatim for all standard files (< 24 KB).
      * Smart Head+Tail context preservation for any oversized files exceeding the 32 KiB payload limit.
  - Resilience: Automatic HTTP 429 rate limit backoff and retry handling.
  - Incremental JSONL checkpointing: Safely resumes without re-calling completed samples.
  - Outputs:
      * CSV predictions (with confidence scores & class probabilities)
      * JSON predictions
      * Comprehensive evaluation summary report & metrics JSON (Accuracy, Macro/Weighted F1,
        Per-Class metrics, Confusion Matrix, and Latencies).
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CATEGORY DEFINITIONS & LABEL MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES: List[str] = [
    "1_banking_kyc_otp_fraud",
    "2_upi_wallet_fraud",
    "3_investment_task_scam",
    "4_digital_arrest_govt_impersonation",
    "5_loan_credit_app_scams",
    "6_delivery_customer_care_scams",
    "7_dating_romance_sextortion",
    "8_legacy_telecom_scams",
]

CATEGORY_DEFINITIONS: Dict[str, str] = {
    "1_banking_kyc_otp_fraud": (
        "Impersonation of bank officials, credit card departments, or banking alerts claiming account "
        "suspension, expired KYC, unauthorized transactions, or debit/credit card blocks. Fraudsters coerce the victim "
        "into sharing One-Time Passwords (OTPs), CVV numbers, net banking credentials, or clicking phishing links."
    ),
    "2_upi_wallet_fraud": (
        "Frauds involving UPI applications (e.g., Google Pay, PhonePe, Paytm) or digital wallets. Scams often use "
        "fake payment screenshots, reverse-charge 'Collect Request' links disguised as payment receipts, QR code "
        "scanning tricks claiming 'scan to receive money', or fake marketplace transactions (e.g., OLX buyer scams)."
    ),
    "3_investment_task_scam": (
        "Part-time job offers, freelance task scams, YouTube video liking/Telegram group tasks, fake stock trading "
        "tips, algorithmic trading platforms, or high-yield cryptocurrency schemes promising exorbitant daily returns. "
        "Victims receive small initial payouts before being pressured to deposit large non-refundable sums."
    ),
    "4_digital_arrest_govt_impersonation": (
        "Severe coercion where scammers impersonate law enforcement, CBI, Police, Customs, Narcotics Control Bureau (NCB), "
        "TRAI, or the Supreme Court. They falsely claim parcels containing contraband/narcotics have been seized in the "
        "victim's name, or allege money laundering, placing the victim under fake 'digital arrest' via continuous video/audio "
        "surveillance and extorting funds for 'verification clearance'."
    ),
    "5_loan_credit_app_scams": (
        "Predatory, instant, or unauthorized digital lending applications offering collateral-free loans. Scammers charge "
        "massive hidden processing fees, disburse partial sums, and engage in aggressive harassment, blackmail, and unauthorized "
        "access to personal phone contacts and gallery for extortion."
    ),
    "6_delivery_customer_care_scams": (
        "Impersonation of courier/logistics services (e.g., India Post, Blue Dart, FedEx, DTDC) or customer support for "
        "e-commerce platforms (e.g., Amazon, Flipkart) or utility providers. Fraudsters claim a parcel address is incomplete, "
        "a package is stuck pending a small fee (e.g., ₹5-₹10), or prompt installation of remote desktop tools (e.g., AnyDesk, TeamViewer)."
    ),
    "7_dating_romance_sextortion": (
        "Romance scams, honey-trapping, video call blackmail, or extortion through dating platforms and social media. "
        "Scammers build intimate relationships, entice the victim into explicit video calls, record compromising footage, "
        "and extort money under threats of uploading videos to social media or sending them to contacts/family."
    ),
    "8_legacy_telecom_scams": (
        "Classic telecommunication and lottery frauds, including fake lottery/KBC (Kaun Banega Crorepati) winnings, "
        "SIM card 4G/5G upgrade verification, SIM swap fraud, fake mobile tower installation offers on personal property, "
        "or urgent utility/electricity bill disconnection threats demanding immediate payment."
    ),
}

LABEL_MAP: Dict[str, int] = {cat: idx for idx, cat in enumerate(CATEGORIES)}
ID2NAME: Dict[int, str] = {idx: cat for cat, idx in LABEL_MAP.items()}

# Max characters in state before applying smart head-tail truncation to stay within 32 KiB JSON payload limit
MAX_STATE_CHARS = 24000


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONTEXT TRUNCATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def prepare_conversation_state(raw_text: str, max_chars: int = MAX_STATE_CHARS) -> Tuple[str, bool]:
    """
    Prepares conversation text for the Jev state payload.
    If the text fits within the limit (covers 100% of synthetic test and >97% of YT files),
    it is preserved verbatim without any modification.

    For rare long transcripts exceeding max_chars, applies Head+Tail preservation:
    - Retains initial context (scammer opening, identity impersonation, pretext).
    - Retains final context (escalation, threats, payment/OTP demand, scam execution).
    """
    if len(raw_text) <= max_chars:
        return raw_text, False

    half = (max_chars - 120) // 2
    head = raw_text[:half]
    tail = raw_text[-half:]
    truncated_text = (
        f"{head}\n\n"
        f"[... {len(raw_text) - 2 * half:,} characters omitted to preserve scam setup and resolution ...]\n\n"
        f"{tail}"
    )
    return truncated_text, True


# ─────────────────────────────────────────────────────────────────────────────
# 3. JEV DECISION CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class JevClient:
    """Client for TypeSafe AI / Jev Community REST API."""

    def __init__(self, api_key: str, base_url: str = "https://www.jevai.org", timeout: float = 60.0):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/v1/decisions"
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) FraudCallDS-JevEval/1.0"
        )

    def decide_category(
        self,
        conversation_text: str,
        max_retries: int = 8,
        backoff_base_sec: float = 20.0,
    ) -> Tuple[Optional[str], float, Dict[str, float], Optional[str], float]:
        """
        Submits conversation to Jev as a typed Choice decision across the 8 categories.
        Returns:
            (predicted_category, confidence, probabilities_dict, error_msg, elapsed_seconds)
        """
        state_text, _ = prepare_conversation_state(conversation_text)

        payload = {
            "state": state_text,
            "questions": {
                "scam_category": {
                    "type": "choice",
                    "instructions": (
                        "Classify this telephone conversation or message exchange into exactly one "
                        "of the pre-defined scam categories based on the primary fraudulent mechanism."
                    ),
                    "criteria": CATEGORY_DEFINITIONS,
                }
            },
        }

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            req = urllib.request.Request(self.endpoint, data=body_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    elapsed = time.time() - t0
                    res_data = json.loads(resp.read().decode("utf-8"))
                    
                    code = res_data.get("code", -1)
                    if code != 0:
                        err_msg = res_data.get("message", f"API error code {code}")
                        return None, 0.0, {}, f"API_ERROR: {err_msg}", elapsed

                    answers = res_data.get("data", {}).get("answers", {})
                    cat_ans = answers.get("scam_category", {})
                    chosen_label = cat_ans.get("choice")
                    conf = float(cat_ans.get("confidence", 0.0))
                    probs = {k: float(v) for k, v in cat_ans.get("probabilities", {}).items()}

                    if chosen_label in LABEL_MAP:
                        return chosen_label, conf, probs, None, elapsed
                    
                    # Case-insensitive match check
                    for cat in CATEGORIES:
                        if str(chosen_label).strip().lower() == cat.lower():
                            return cat, conf, probs, None, elapsed

                    return chosen_label, conf, probs, f"UNRECOGNIZED_LABEL: {chosen_label}", elapsed

            except urllib.error.HTTPError as http_err:
                elapsed = time.time() - t0
                if http_err.code in (429, 500, 502, 503, 504):
                    sleep_time = backoff_base_sec * attempt
                    err_type = "Rate Limit (429)" if http_err.code == 429 else f"Server Error ({http_err.code})"
                    print(
                        f"\n⚠️  [{err_type} Encountered] Waiting {sleep_time:.1f}s before retry "
                        f"(attempt {attempt}/{max_retries})...",
                        flush=True,
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    try:
                        err_body = http_err.read().decode("utf-8")
                    except Exception:
                        err_body = str(http_err)
                    return None, 0.0, {}, f"HTTP_{http_err.code}: {err_body}", elapsed

            except Exception as e:
                elapsed = time.time() - t0
                if attempt < max_retries:
                    sleep_time = 15.0 * attempt
                    print(f"\n⚠️  [Network Error: {e}] Retrying in {sleep_time:.1f}s (attempt {attempt}/{max_retries})...", flush=True)
                    time.sleep(sleep_time)
                    continue
                return None, 0.0, {}, f"CONNECTION_ERROR: {str(e)}", elapsed

        return None, 0.0, {}, "MAX_RETRIES_EXCEEDED", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_evaluation_dataset(split_dir: Path, split_name: str) -> List[Dict[str, Any]]:
    """Loads text files organized by category subdirectory."""
    split_dir = Path(split_dir).resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {split_dir}")

    items: List[Dict[str, Any]] = []
    print(f"\n📂 Scanning {split_name.upper()} directory: {split_dir}")

    for category in CATEGORIES:
        cat_dir = split_dir / category
        if not cat_dir.is_dir():
            logging.warning("Category directory not found: %s", cat_dir)
            continue

        txt_files = sorted(cat_dir.glob("*.txt"))
        for file_path in txt_files:
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
                items.append({
                    "dataset": split_name,
                    "filename": file_path.name,
                    "rel_path": str(file_path.relative_to(split_dir)),
                    "abs_path": str(file_path),
                    "ground_truth": category,
                    "text": raw_text,
                    "char_len": len(raw_text),
                })
            except Exception as e:
                logging.error("Failed to read file %s: %s", file_path, e)

    print(f"   Loaded {len(items):,} items across {len(CATEGORIES)} categories from {split_name}.")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 5. CACHE / RESUME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_records(log_path: Path, legacy_json_path: Optional[Path] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads previously computed records from JSONL log or JSON predictions file."""
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        gt = rec.get("ground_truth")
                        fn = rec.get("filename")
                        if gt and fn:
                            records[(gt, fn)] = rec
                    except Exception:
                        continue
        except Exception as e:
            logging.warning("Error reading log %s: %s", log_path, e)

    if not records and legacy_json_path and legacy_json_path.exists():
        try:
            with open(legacy_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        gt = rec.get("ground_truth")
                        fn = rec.get("filename")
                        if gt and fn:
                            records[(gt, fn)] = rec
        except Exception as e:
            logging.warning("Error reading JSON %s: %s", legacy_json_path, e)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    client: JevClient,
    dataset_records: List[Dict[str, Any]],
    split_name: str,
    output_dir: Path,
    delay_sec: float = 2.0,
    resume: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates split with Jev model.
    Saves incremental progress and computes classification metrics.
    """
    print(f"\n{'=' * 75}")
    print(f"STARTING JEV EVALUATION: {split_name.upper()} (N = {len(dataset_records):,})")
    print(f"Delay between requests: {delay_sec:.1f}s | Resume: {resume}")
    print(f"{'=' * 75}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"jev_{split_name}_log.jsonl"
    legacy_json_path = output_dir / f"jev_{split_name}_predictions.json"

    completed_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if resume:
        completed_cache = load_existing_records(log_path, legacy_json_path)
        if completed_cache and (not log_path.exists() or log_path.stat().st_size == 0):
            with open(log_path, "w", encoding="utf-8") as f_backfill:
                for rec in completed_cache.values():
                    f_backfill.write(json.dumps(rec, ensure_ascii=False) + "\n")

    num_total = len(dataset_records)
    resumed_count = sum(
        1 for item in dataset_records
        if (item["ground_truth"], item["filename"]) in completed_cache
        and completed_cache[(item["ground_truth"], item["filename"])].get("is_valid", False)
    )
    if resumed_count > 0:
        print(f"🔄 Resuming {split_name}: found {resumed_count}/{num_total} already evaluated sample(s).")
        print(f"   Log file: {log_path}")
        print(f"   Skipping {resumed_count} completed sample(s), evaluating remaining {num_total - resumed_count}...\n")
    else:
        print(f"📝 Logging predictions incrementally to: {log_path}\n")

    predictions_records: List[Dict[str, Any]] = []
    gold_labels: List[int] = []
    pred_labels: List[int] = []

    num_correct = 0
    num_incorrect = 0
    num_invalid = 0

    start_time = time.time()
    log_file_handle = open(log_path, "a", encoding="utf-8")

    try:
        for idx, item in enumerate(dataset_records, start=1):
            filename = item["filename"]
            ground_truth = item["ground_truth"]
            gold_id = LABEL_MAP[ground_truth]
            sample_key = (ground_truth, filename)

            # Check cache (only reuse if valid prediction was obtained)
            if resume and sample_key in completed_cache and completed_cache[sample_key].get("is_valid", False):
                record = completed_cache[sample_key]
                pred_label = record.get("predicted_label")
                is_valid = True
                is_correct = record.get("is_correct", False)

                if is_valid and pred_label in LABEL_MAP:
                    pred_id = LABEL_MAP[pred_label]
                    if is_correct:
                        num_correct += 1
                    else:
                        num_incorrect += 1
                else:
                    pred_id = -1
                    num_invalid += 1
                    num_incorrect += 1

                gold_labels.append(gold_id)
                pred_labels.append(pred_id)
                predictions_records.append(record)

                if idx % 25 == 0 or idx == resumed_count or idx == num_total:
                    acc = (num_correct / idx) * 100 if idx > 0 else 0.0
                    print(
                        f"[{idx:4d}/{num_total:4d}] {filename[:28]:<28} | "
                        f"[CACHED] Gold: {ground_truth[:18]:<18} | "
                        f"Pred: {str(pred_label)[:18]:<18} | "
                        f"Acc: {acc:5.2f}%"
                    )
                continue

            # Query Jev
            raw_text = item["text"]
            pred_cat, conf, probs, err, latency = client.decide_category(raw_text)

            if err is not None:
                predicted_label = "ERROR_OR_INVALID"
                is_valid = False
                is_correct = False
                pred_id = -1
                num_invalid += 1
                num_incorrect += 1
            else:
                is_valid = (pred_cat in LABEL_MAP)
                predicted_label = pred_cat if is_valid else "UNRECOGNIZED_LABEL"
                if is_valid:
                    pred_id = LABEL_MAP[pred_cat]
                    is_correct = (pred_cat == ground_truth)
                    if is_correct:
                        num_correct += 1
                    else:
                        num_incorrect += 1
                else:
                    pred_id = -1
                    is_correct = False
                    num_invalid += 1
                    num_incorrect += 1

            gold_labels.append(gold_id)
            pred_labels.append(pred_id)

            record = {
                "dataset": split_name,
                "filename": filename,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label,
                "confidence": round(conf, 4),
                "is_valid": is_valid,
                "is_correct": is_correct,
                "latency_sec": round(latency, 3),
                "error": err or "",
                "probabilities": probs,
            }
            predictions_records.append(record)
            completed_cache[sample_key] = record

            # Persist incrementally
            log_file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())

            # Progress log
            acc = (num_correct / idx) * 100
            print(
                f"[{idx:4d}/{num_total:4d}] {filename[:28]:<28} | "
                f"Gold: {ground_truth[:18]:<18} | "
                f"Pred: {str(predicted_label)[:18]:<18} | "
                f"Conf: {conf:4.2f} | "
                f"Acc: {acc:5.2f}%"
            )

            # Delay to respect API rate limits
            if delay_sec > 0 and idx < num_total:
                time.sleep(delay_sec)

    finally:
        log_file_handle.close()

    total_eval_time = time.time() - start_time

    # Compute metrics
    metrics_summary = compute_metrics(
        gold=gold_labels,
        preds=pred_labels,
        records=predictions_records,
        split_name=split_name,
        num_total=num_total,
        num_correct=num_correct,
        num_incorrect=num_incorrect,
        num_invalid=num_invalid,
        total_time_sec=total_eval_time,
    )

    return metrics_summary, predictions_records


# ─────────────────────────────────────────────────────────────────────────────
# 7. METRIC CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    gold: List[int],
    preds: List[int],
    records: List[Dict[str, Any]],
    split_name: str,
    num_total: int,
    num_correct: int,
    num_incorrect: int,
    num_invalid: int,
    total_time_sec: float,
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics."""
    num_classes = len(CATEGORIES)
    all_class_indices = list(range(num_classes))

    accuracy = float(num_correct / num_total) if num_total > 0 else 0.0

    macro_prec = float(precision_score(gold, preds, labels=all_class_indices, average="macro", zero_division=0))
    macro_rec = float(recall_score(gold, preds, labels=all_class_indices, average="macro", zero_division=0))
    macro_f1 = float(f1_score(gold, preds, labels=all_class_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gold, preds, labels=all_class_indices, average="weighted", zero_division=0))

    report_dict = classification_report(
        gold,
        preds,
        labels=all_class_indices,
        target_names=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(gold, preds, labels=all_class_indices)

    per_class_metrics: Dict[str, Any] = {}
    for idx, cat_name in enumerate(CATEGORIES):
        class_stats = report_dict.get(cat_name, {})
        support_val = int(class_stats.get("support", 0))
        correct_val = int(cm[idx, idx])
        per_class_acc = float(correct_val / support_val) if support_val > 0 else 0.0

        per_class_metrics[cat_name] = {
            "class_id": idx,
            "precision": float(class_stats.get("precision", 0.0)),
            "recall": float(class_stats.get("recall", 0.0)),
            "f1_score": float(class_stats.get("f1-score", 0.0)),
            "support": support_val,
            "correct": correct_val,
            "accuracy": per_class_acc,
        }

    # Latencies
    latencies = [r["latency_sec"] for r in records if "latency_sec" in r]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "dataset": split_name,
        "total_conversations": num_total,
        "correct_predictions": num_correct,
        "incorrect_predictions": num_incorrect,
        "invalid_or_unparseable_predictions": num_invalid,
        "evaluation_time_seconds": round(total_time_sec, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "overall_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": CATEGORIES,
            "matrix": cm.tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. OUTPUT PERSISTENCE & REPORTS
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Saves predictions with probabilities to CSV."""
    if not records:
        return
    fieldnames = [
        "dataset",
        "filename",
        "ground_truth",
        "predicted_label",
        "confidence",
        "is_valid",
        "is_correct",
        "latency_sec",
        "error",
    ]
    # Add probability columns for all 8 categories
    for cat in CATEGORIES:
        fieldnames.append(f"prob_{cat}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "dataset": r.get("dataset"),
                "filename": r.get("filename"),
                "ground_truth": r.get("ground_truth"),
                "predicted_label": r.get("predicted_label"),
                "confidence": r.get("confidence", 0.0),
                "is_valid": r.get("is_valid", False),
                "is_correct": r.get("is_correct", False),
                "latency_sec": r.get("latency_sec", 0.0),
                "error": r.get("error", ""),
            }
            probs = r.get("probabilities", {}) or {}
            for cat in CATEGORIES:
                row[f"prob_{cat}"] = probs.get(cat, 0.0)
            writer.writerow(row)
    print(f"💾 Saved raw predictions CSV → {output_path}")


def save_predictions_json(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Saves raw predictions to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"💾 Saved raw predictions JSON → {output_path}")


def print_and_save_final_report(
    test_metrics: Optional[Dict[str, Any]],
    yt_metrics: Optional[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Prints and writes evaluation summary text report."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("JEV MODEL SCAM CLASSIFICATION EVALUATION REPORT")
    lines.append("Model: Jev (System One Decision Model) | Zero-Shot Evaluation")
    lines.append("=" * 78)

    for metrics in [test_metrics, yt_metrics]:
        if metrics is None:
            continue
        split_name = metrics["dataset"].upper()
        lines.append(f"\n[{split_name} DATASET METRICS]")
        lines.append("-" * 78)
        lines.append(f"Total Conversations: {metrics['total_conversations']}")
        lines.append(f"Correctly Classified: {metrics['correct_predictions']}")
        lines.append(f"Incorrectly Classified: {metrics['incorrect_predictions']}")
        lines.append(f"Invalid / Errors: {metrics['invalid_or_unparseable_predictions']}")
        lines.append(f"Evaluation Time: {metrics['evaluation_time_seconds']}s")
        lines.append(f"Average Request Latency: {metrics.get('avg_latency_seconds', 0.0)}s")
        lines.append("")
        ov = metrics["overall_metrics"]
        lines.append(f"Accuracy:        {ov['accuracy']:.4f}")
        lines.append(f"Macro Precision: {ov['macro_precision']:.4f}")
        lines.append(f"Macro Recall:    {ov['macro_recall']:.4f}")
        lines.append(f"Macro F1:        {ov['macro_f1']:.4f}")
        lines.append(f"Weighted F1:     {ov['weighted_f1']:.4f}")
        lines.append("")
        lines.append(f"{'Category':<38} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7}")
        lines.append("-" * 78)
        for cat, stats in metrics["per_class_metrics"].items():
            lines.append(
                f"{cat:<38} | {stats['precision']:6.4f} | {stats['recall']:6.4f} | "
                f"{stats['f1_score']:6.4f} | {stats['support']:7d}"
            )
        lines.append("-" * 78)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_file = output_dir / "jev_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n💾 Saved summary report text → {report_file}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Shot Scam-Call Classification using Jev Model"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("JEV_API_KEY", "jev_8NrLp29rmYuZdKVK0M77i6NRsc93ZSKp"),
        help="Jev API key (default: from JEV_API_KEY env or provided key)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("JEV_BASE_URL", "https://www.jevai.org"),
        help="Jev API base URL (default: https://www.jevai.org)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset_split/test",
        help="Path to synthetic test split directory (default: dataset_split/test)",
    )
    parser.add_argument(
        "--yt-dir",
        type=str,
        default="dataset_split/yt",
        help="Path to external YouTube split directory (default: dataset_split/yt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/results",
        help="Directory where predictions and summary metrics will be saved (default: models/results)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples to evaluate per split (useful for smoke tests)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Delay in seconds between API calls to avoid 429 rate limits (default: 10.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip evaluation on the synthetic test set",
    )
    parser.add_argument(
        "--skip-yt",
        action="store_true",
        help="Skip evaluation on the external YT set",
    )
    parser.add_argument(
        "--yt-first",
        action="store_true",
        default=True,
        help="Evaluate YT set before Test set (default: True, as YT has 77 samples and completes faster)",
    )
    parser.add_argument(
        "--test-first",
        dest="yt_first",
        action="store_false",
        help="Evaluate Test set before YT set",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing log files and re-run all evaluations from scratch",
    )
    return parser.parse_args()


def resolve_directory(path_str: str) -> Path:
    cand = Path(path_str)
    if cand.exists():
        return cand.resolve()
    repo_cand = Path(__file__).resolve().parent.parent / path_str
    if repo_cand.exists():
        return repo_cand.resolve()
    return cand.resolve()


def main() -> None:
    args = parse_arguments()

    if not args.api_key:
        print("❌ [ERROR] No Jev API key provided. Specify via --api-key or set JEV_API_KEY environment variable.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = JevClient(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)

    test_dir = resolve_directory(args.test_dir)
    yt_dir = resolve_directory(args.yt_dir)

    all_results: Dict[str, Any] = {
        "experiment": "jev_zero_shot_classification",
        "model": "jev-1",
        "base_url": args.base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    test_metrics = None
    yt_metrics = None

    def run_yt():
        nonlocal yt_metrics
        yt_items = load_evaluation_dataset(yt_dir, "yt")
        if args.max_samples is not None:
            print(f"⚠️ Limiting YT set to first {args.max_samples} samples as requested.")
            yt_items = yt_items[: args.max_samples]

        yt_metrics, yt_preds = evaluate_split(
            client=client,
            dataset_records=yt_items,
            split_name="yt",
            output_dir=output_dir,
            delay_sec=args.delay,
            resume=not args.force_rerun,
        )

        save_predictions_csv(yt_preds, output_dir / "jev_yt_predictions.csv")
        save_predictions_json(yt_preds, output_dir / "jev_yt_predictions.json")
        all_results["yt_evaluation"] = yt_metrics

        # Save metrics & report incrementally
        combined_metrics_path = output_dir / "jev_metrics.json"
        with open(combined_metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print_and_save_final_report(test_metrics=test_metrics, yt_metrics=yt_metrics, output_dir=output_dir)

    def run_test():
        nonlocal test_metrics
        test_items = load_evaluation_dataset(test_dir, "test")
        if args.max_samples is not None:
            print(f"⚠️ Limiting test set to first {args.max_samples} samples as requested.")
            test_items = test_items[: args.max_samples]

        test_metrics, test_preds = evaluate_split(
            client=client,
            dataset_records=test_items,
            split_name="test",
            output_dir=output_dir,
            delay_sec=args.delay,
            resume=not args.force_rerun,
        )

        save_predictions_csv(test_preds, output_dir / "jev_test_predictions.csv")
        save_predictions_json(test_preds, output_dir / "jev_test_predictions.json")
        all_results["test_evaluation"] = test_metrics

        # Save metrics & report incrementally
        combined_metrics_path = output_dir / "jev_metrics.json"
        with open(combined_metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print_and_save_final_report(test_metrics=test_metrics, yt_metrics=yt_metrics, output_dir=output_dir)

    # Determine execution order
    splits_to_run = []
    if args.yt_first:
        if not args.skip_yt:
            splits_to_run.append(("yt", run_yt))
        if not args.skip_test:
            splits_to_run.append(("test", run_test))
    else:
        if not args.skip_test:
            splits_to_run.append(("test", run_test))
        if not args.skip_yt:
            splits_to_run.append(("yt", run_yt))

    for split_label, run_func in splits_to_run:
        run_func()

    print("\n✅ Jev evaluation pipeline completed.")


if __name__ == "__main__":
    main()
