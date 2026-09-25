"""
zero_shot.py — Zero-Shot Scam Call Classification with GPT-OSS 120B via Ollama
-------------------------------------------------------------------------------
Evaluates GPT-OSS (default: gpt-oss:120b) as a true zero-shot classifier on:
  1. Synthetic test set (dataset_split/test/)
  2. External YouTube / YT set (dataset_split/yt/)

Constraints:
  - Strictly zero-shot: No training, validation, or few-shot examples in prompt.
  - Category definitions & instructions only + verbatim conversation text.
  - Temperature = 0 for deterministic evaluation.
  - Uses the official `ollama` Python library.
  - Preserves exact raw conversation text without cleaning or manual truncation.
  - Robust parser recording invalid / unparseable outputs explicitly.
  - Comprehensive metrics: Accuracy, Macro/Weighted Precision/Recall/F1, Per-class metrics,
    Confusion Matrix, Support, Correct/Incorrect counts, and Invalid prediction counts.
  - Full output logging: per-split predictions (CSV & JSON) and evaluation summary report.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
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

try:
    import ollama
except ImportError:
    ollama = None


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


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZERO-SHOT CLASSIFICATION PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_zero_shot_prompt(conversation_text: str) -> Tuple[str, str]:
    """
    Constructs the zero-shot system instruction and user prompt.
    Does NOT include any few-shot examples or training samples.
    """
    category_descriptions = "\n".join(
        [f"- \"{cat}\": {CATEGORY_DEFINITIONS[cat]}" for cat in CATEGORIES]
    )

    system_message = (
        "You are an expert conversational AI scam-detection analyst. "
        "Your task is to accurately classify an incoming scam telephone conversation or message exchange "
        "into exactly ONE of the pre-defined scam categories.\n\n"
        "Available Scam Categories:\n"
        f"{category_descriptions}\n\n"
        "Instructions:\n"
        "1. Carefully read the conversation text provided by the user.\n"
        "2. Select exactly one category from the available scam categories listed above that best characterizes the primary scam mechanism.\n"
        "3. Respond ONLY with a valid JSON object in the exact format shown below:\n"
        '{"label": "<category_name>"}\n'
        "4. Do NOT output any analysis, explanation, thoughts, notes, markdown commentary, or additional keys.\n"
        "5. The value for \"label\" MUST be one of the exact 8 category names listed above."
    )

    user_message = (
        "Classify the following scam conversation into exactly one of the 8 categories.\n\n"
        "--- BEGIN CONVERSATION ---\n"
        f"{conversation_text}\n"
        "--- END CONVERSATION ---\n\n"
        "Respond strictly with the JSON object: {\"label\": \"<exact_category_name>\"}"
    )

    return system_message, user_message


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESPONSE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_model_response(raw_response: str) -> Tuple[Optional[str], bool, str]:
    """
    Parses the predicted scam label from the model's raw text response.
    Returns:
        (predicted_label, is_valid, failure_reason)
    """
    if not raw_response or not raw_response.strip():
        return None, False, "Empty response from model"

    cleaned = raw_response.strip()

    # 1. Try stripping markdown code fences (```json ... ``` or ``` ...)
    json_str_candidate = cleaned
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        json_str_candidate = fence_match.group(1).strip()

    # 2. Try direct JSON parsing
    parsed_json = None
    try:
        parsed_json = json.loads(json_str_candidate)
    except Exception:
        # Fallback: search for first { ... } block
        brace_match = re.search(r"\{[\s\S]*?\}", cleaned)
        if brace_match:
            try:
                parsed_json = json.loads(brace_match.group(0))
            except Exception:
                pass

    if isinstance(parsed_json, dict) and "label" in parsed_json:
        label_val = str(parsed_json["label"]).strip()
        if label_val in LABEL_MAP:
            return label_val, True, ""
        # Check case-insensitive exact match
        for valid_cat in CATEGORIES:
            if label_val.lower() == valid_cat.lower():
                return valid_cat, True, ""
        return label_val, False, f"Parsed JSON label '{label_val}' not in allowed 8 categories"

    # 3. Fallback: regex search for "label": "..." pattern
    label_pattern_match = re.search(r'["\']?label["\']?\s*:\s*["\']([^"\']+)["\']', cleaned, re.IGNORECASE)
    if label_pattern_match:
        extracted = label_pattern_match.group(1).strip()
        if extracted in LABEL_MAP:
            return extracted, True, ""
        for valid_cat in CATEGORIES:
            if extracted.lower() == valid_cat.lower():
                return valid_cat, True, ""
        return extracted, False, f"Extracted label '{extracted}' is not an authorized category"

    # 4. Check if the entire output is simply the category string
    for valid_cat in CATEGORIES:
        if cleaned == valid_cat or cleaned.lower() == valid_cat.lower():
            return valid_cat, True, ""

    return None, False, f"Could not parse valid category JSON from response: {cleaned[:120]}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA LOADING (TEST & YT ONLY)
# ─────────────────────────────────────────────────────────────────────────────

def load_evaluation_dataset(split_dir: Path, split_name: str) -> List[Dict[str, Any]]:
    """
    Loads text files from the split directory organized by category subdirectory.
    Strictly verifies that only test or yt datasets are evaluated.
    """
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
                # Preserve raw text exactly without cleaning, translation, or shortening
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
# 5. OLLAMA CLIENT & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def verify_ollama_setup(client: Any, model_name: str) -> None:
    """
    Verifies that the Ollama service is reachable and the specified model exists.
    """
    print(f"🔍 Checking Ollama connection and model '{model_name}'...")
    try:
        models_resp = client.list()
    except Exception as e:
        print(f"\n❌ [ERROR] Unable to connect to Ollama: {e}")
        print("Please verify that Ollama is installed, running locally, and accessible.\n")
        sys.exit(1)

    available_models = []
    # Handle both client.list() dictionary format and Model objects
    models_list = getattr(models_resp, "models", None) or models_resp.get("models", [])
    for m in models_list:
        name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") if isinstance(m, dict) else m.get("name"))
        if name:
            available_models.append(name)

    model_matched = any(
        model_name == m or model_name in m or m.startswith(model_name)
        for m in available_models
    )

    if not model_matched:
        print(f"\n⚠️ [WARNING] Requested model '{model_name}' was not found in local Ollama library.")
        print(f"Available models ({len(available_models)}):")
        for m in available_models:
            print(f"  - {m}")
        print(f"\nAttempting to continue. If '{model_name}' is not pulled, Ollama may return an error.\n")
    else:
        print(f"✅ Ollama is ready. Target model '{model_name}' detected.")


def query_ollama_zero_shot(
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    timeout: Optional[float] = 120.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Executes zero-shot inference via the official ollama Python client.
    Returns: (raw_response_text, error_message)
    """
    options: Dict[str, Any] = {
        "temperature": temperature,
    }
    if seed is not None:
        options["seed"] = seed

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat(
            model=model_name,
            messages=messages,
            options=options,
            format="json",
        )
        # response['message']['content']
        msg = getattr(response, "message", None) or response.get("message", {})
        content = getattr(msg, "content", None) or msg.get("content", "")
        return content, None
    except Exception as exc:
        err_msg = str(exc)
        return None, err_msg


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION ENGINE & METRIC CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_records(
    log_path: Path, legacy_json_path: Optional[Path] = None
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Loads previously computed records from the JSONL log file (or legacy JSON file)
    keyed by (ground_truth, filename).
    """
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # 1. Check incremental JSONL log file first
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
            logging.warning("Error reading existing log file %s: %s", log_path, e)

    # 2. Fallback to existing JSON predictions file if log_path didn't have records
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
            logging.warning("Error reading legacy predictions JSON %s: %s", legacy_json_path, e)

    return records


def evaluate_split(
    client: Any,
    model_name: str,
    dataset_records: List[Dict[str, Any]],
    split_name: str,
    output_dir: Path,
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    timeout: Optional[float] = 120.0,
    resume: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Runs zero-shot classification on all conversations in the split.
    Calculates overall and per-class metrics strictly without mixing datasets.
    Supports incremental saving to a log file and resuming execution.
    """
    print(f"\n{'=' * 75}")
    print(f"STARTING ZERO-SHOT EVALUATION: {split_name.upper()} (N = {len(dataset_records):,})")
    print(f"Model: {model_name} | Temperature: {temperature} | Seed: {seed} | Resume: {resume}")
    print(f"{'=' * 75}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"zero_shot_{split_name}_log.jsonl"
    legacy_json_path = output_dir / f"zero_shot_{split_name}_predictions.json"

    completed_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if resume:
        completed_cache = load_existing_records(log_path, legacy_json_path)
        # If loaded from legacy JSON and log_path is missing, backfill log_path
        if completed_cache and (not log_path.exists() or log_path.stat().st_size == 0):
            with open(log_path, "w", encoding="utf-8") as f_backfill:
                for rec in completed_cache.values():
                    f_backfill.write(json.dumps(rec, ensure_ascii=False) + "\n")

    num_total = len(dataset_records)
    resumed_count = sum(
        1 for item in dataset_records if (item["ground_truth"], item["filename"]) in completed_cache
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

            # If sample was already completed in prior run, reuse it directly
            if resume and sample_key in completed_cache:
                record = completed_cache[sample_key]
                predicted_label = record.get("predicted_label")
                is_valid = record.get("is_valid", False)
                is_correct = record.get("is_correct", False)

                if is_valid and predicted_label in LABEL_MAP:
                    pred_id = LABEL_MAP[predicted_label]
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

                if idx % 50 == 0 or idx == resumed_count or idx == num_total:
                    current_acc = (num_correct / idx) * 100 if idx > 0 else 0.0
                    print(
                        f"[{idx:4d}/{num_total:4d}] {filename[:30]:<30} | "
                        f"[CACHED] Gold: {ground_truth[:18]:<18} | "
                        f"Pred: {str(predicted_label)[:18]:<18} | "
                        f"Acc: {current_acc:5.2f}%"
                    )
                continue

            # Evaluate new sample with Ollama
            raw_text = item["text"]
            system_msg, user_msg = build_zero_shot_prompt(raw_text)

            t0 = time.time()
            raw_output, err = query_ollama_zero_shot(
                client=client,
                model_name=model_name,
                system_prompt=system_msg,
                user_prompt=user_msg,
                temperature=temperature,
                seed=seed,
                timeout=timeout,
            )
            elapsed = time.time() - t0

            if err is not None:
                predicted_label = None
                is_valid = False
                is_correct = False
                raw_response_str = f"[INFERENCE_ERROR: {err}]"
                pred_id = -1
                num_invalid += 1
                num_incorrect += 1
            else:
                raw_response_str = raw_output or ""
                pred_cat, valid, parse_err = parse_model_response(raw_response_str)
                is_valid = valid
                predicted_label = pred_cat

                if is_valid and pred_cat in LABEL_MAP:
                    pred_id = LABEL_MAP[pred_cat]
                    is_correct = (predicted_label == ground_truth)
                    if is_correct:
                        num_correct += 1
                    else:
                        num_incorrect += 1
                else:
                    pred_id = -1
                    num_invalid += 1
                    num_incorrect += 1
                    is_correct = False

            gold_labels.append(gold_id)
            pred_labels.append(pred_id)

            record = {
                "dataset": split_name,
                "filename": filename,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label if predicted_label is not None else "INVALID_OR_ERROR",
                "is_valid": is_valid,
                "is_correct": is_correct,
                "latency_sec": round(elapsed, 3),
                "raw_response": raw_response_str,
            }
            predictions_records.append(record)
            completed_cache[sample_key] = record

            # Immediately write and flush to log file so progress is persistently preserved
            log_file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())

            # Progress log
            if idx % 10 == 0 or idx == num_total or idx <= 5:
                current_acc = (num_correct / idx) * 100
                print(
                    f"[{idx:4d}/{num_total:4d}] {filename[:30]:<30} | "
                    f"Gold: {ground_truth[:18]:<18} | "
                    f"Pred: {str(predicted_label)[:18]:<18} | "
                    f"Valid: {str(is_valid):<5} | "
                    f"Acc: {current_acc:5.2f}%"
                )
    finally:
        log_file_handle.close()

    total_eval_time = time.time() - start_time

    # Compute classification metrics
    metrics_summary = compute_metrics(
        gold=gold_labels,
        preds=pred_labels,
        split_name=split_name,
        num_total=num_total,
        num_correct=num_correct,
        num_incorrect=num_incorrect,
        num_invalid=num_invalid,
        total_time_sec=total_eval_time,
    )

    return metrics_summary, predictions_records



def compute_metrics(
    gold: List[int],
    preds: List[int],
    split_name: str,
    num_total: int,
    num_correct: int,
    num_incorrect: int,
    num_invalid: int,
    total_time_sec: float,
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics. Any invalid or unparseable prediction
    counts as an incorrect prediction.
    """
    num_classes = len(CATEGORIES)
    all_class_indices = list(range(num_classes))

    # Strict accuracy: correctly predicted instances / total instances
    accuracy = float(num_correct / num_total) if num_total > 0 else 0.0

    # Scikit-learn calculation over the 8 classes
    # Any prediction not in 0..7 (i.e., -1) will be treated as false negative for the gold class
    macro_prec = float(precision_score(gold, preds, labels=all_class_indices, average="macro", zero_division=0))
    macro_rec = float(recall_score(gold, preds, labels=all_class_indices, average="macro", zero_division=0))
    macro_f1 = float(f1_score(gold, preds, labels=all_class_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gold, preds, labels=all_class_indices, average="weighted", zero_division=0))

    # Per-class classification report
    report_dict = classification_report(
        gold,
        preds,
        labels=all_class_indices,
        target_names=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    # 8x8 confusion matrix restricted to valid classes
    # Gold on rows, Predictions on columns
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

    return {
        "dataset": split_name,
        "total_conversations": num_total,
        "correct_predictions": num_correct,
        "incorrect_predictions": num_incorrect,
        "invalid_or_unparseable_predictions": num_invalid,
        "evaluation_time_seconds": round(total_time_sec, 2),
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
# 7. OUTPUT PERSISTENCE & REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Saves raw predictions to CSV."""
    if not records:
        return
    fieldnames = [
        "dataset",
        "filename",
        "ground_truth",
        "predicted_label",
        "is_valid",
        "is_correct",
        "latency_sec",
        "raw_response",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
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
    model_name: str,
) -> None:
    """
    Generates and prints a clean, formatted evaluation report.
    """
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"ZERO-SHOT SCAM CLASSIFICATION EVALUATION REPORT")
    lines.append(f"Model: {model_name} (via Ollama) | Zero-Shot (No Few-Shot, No Training)")
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
        lines.append(f"Invalid / Unparseable: {metrics['invalid_or_unparseable_predictions']}")
        lines.append(f"Evaluation Time: {metrics['evaluation_time_seconds']}s")
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

    report_file = output_dir / "zero_shot_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n💾 Saved summary report text → {report_file}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENTRYPOINT & CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Shot Scam-Call Classification with GPT-OSS 120B using Ollama"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma4:31b-cloud",
        help="Ollama model name (default: gpt-oss:120b)",
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
        "--host",
        type=str,
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama server host endpoint (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for deterministic evaluation (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each Ollama request (default: 120.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples to evaluate per split (useful for quick smoke tests)",
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
        "--force-rerun",
        action="store_true",
        help="Ignore existing log files and re-run all evaluations from scratch",
    )
    return parser.parse_args()


def resolve_directory(path_str: str) -> Path:
    """Finds path relative to cwd or repo root."""
    cand = Path(path_str)
    if cand.exists():
        return cand.resolve()
    repo_cand = Path(__file__).resolve().parent.parent / path_str
    if repo_cand.exists():
        return repo_cand.resolve()
    return cand.resolve()


def main() -> None:
    args = parse_arguments()

    if ollama is None:
        print("❌ [ERROR] The 'ollama' library is not installed.")
        print("Please install it via: pip install ollama")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Ollama client
    client = ollama.Client(host=args.host)
    verify_ollama_setup(client, args.model)

    test_dir = resolve_directory(args.test_dir)
    yt_dir = resolve_directory(args.yt_dir)

    all_results: Dict[str, Any] = {
        "experiment": "zero_shot_classification",
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    test_metrics = None
    yt_metrics = None

    # ─────────────────────────────────────────────────────────────
    # 1. EVALUATE SYNTHETIC TEST SET
    # ─────────────────────────────────────────────────────────────
    if not args.skip_test:
        test_items = load_evaluation_dataset(test_dir, "test")
        if args.max_samples is not None:
            print(f"⚠️ Limiting test set to first {args.max_samples} samples as requested.")
            test_items = test_items[: args.max_samples]

        test_metrics, test_preds = evaluate_split(
            client=client,
            model_name=args.model,
            dataset_records=test_items,
            split_name="test",
            output_dir=output_dir,
            temperature=args.temperature,
            seed=args.seed,
            timeout=args.timeout,
            resume=not args.force_rerun,
        )

        save_predictions_csv(test_preds, output_dir / "zero_shot_test_predictions.csv")
        save_predictions_json(test_preds, output_dir / "zero_shot_test_predictions.json")
        all_results["test_evaluation"] = test_metrics

    # ─────────────────────────────────────────────────────────────
    # 2. EVALUATE EXTERNAL YT SET
    # ─────────────────────────────────────────────────────────────
    if not args.skip_yt:
        yt_items = load_evaluation_dataset(yt_dir, "yt")
        if args.max_samples is not None:
            print(f"⚠️ Limiting YT set to first {args.max_samples} samples as requested.")
            yt_items = yt_items[: args.max_samples]

        yt_metrics, yt_preds = evaluate_split(
            client=client,
            model_name=args.model,
            dataset_records=yt_items,
            split_name="yt",
            output_dir=output_dir,
            temperature=args.temperature,
            seed=args.seed,
            timeout=args.timeout,
            resume=not args.force_rerun,
        )

        save_predictions_csv(yt_preds, output_dir / "zero_shot_yt_predictions.csv")
        save_predictions_json(yt_preds, output_dir / "zero_shot_yt_predictions.json")
        all_results["yt_evaluation"] = yt_metrics

    # ─────────────────────────────────────────────────────────────
    # 3. SAVE COMBINED SUMMARY METRICS & GENERATE REPORT
    # ─────────────────────────────────────────────────────────────
    combined_metrics_path = output_dir / "zero_shot_metrics.json"
    with open(combined_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Saved combined evaluation metrics JSON → {combined_metrics_path}")

    # Also save in models/ if running from root or models/ directory
    aux_models_dir = Path("models/results")
    if aux_models_dir.resolve() != output_dir.resolve():
        try:
            aux_models_dir.mkdir(parents=True, exist_ok=True)
            with open(aux_models_dir / "zero_shot_metrics.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
        except Exception:
            pass

    print_and_save_final_report(
        test_metrics=test_metrics,
        yt_metrics=yt_metrics,
        output_dir=output_dir,
        model_name=args.model,
    )

    print("\n✅ Zero-shot evaluation pipeline finished successfully.")


if __name__ == "__main__":
    main()
