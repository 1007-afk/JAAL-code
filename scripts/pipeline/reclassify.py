"""
Reclassify scam transcripts using gemma3:12b-cloud via Ollama context-aware classification.
Reclassifies from 8 classes to 8 classes — fixes misclassified ones.
Replaces keyword-based class assignment with semantic understanding.

Usage:
    python reclassify.py --input <scam_8_folder> --output <new_folder>

Output:
    - New folder with same 8-class structure, files moved to correct class
    - Console report of how many files moved and where
"""

import os
import sys
import re
import json
import shutil
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL = "gemma3:12b-cloud"
CLASSES = [
    "1_bank_impersonation",
    "2_covid_health_scam",
    "3_credit_debit_card_fraud",
    "4_dating_romance_scam",
    "5_delivery_customer_care_scam",
    "6_government_police_scam",
    "7_investment_money_scam",
    "8_kyc_fraud",
]
CLASSIFY_PROMPT = """
You are an expert at classifying Indian scam call transcripts.

Choose EXACTLY ONE class.

1_bank_impersonation
- Fake bank employee
- Account verification
- Banking support staff
- Fraudulent banking assistance

2_covid_health_scam
- Fake covid relief
- Medical aid
- Health schemes
- Pandemic related scams

3_credit_debit_card_fraud
- Credit card fraud
- Debit card fraud
- Card blocked
- Card upgrade scams

4_dating_romance_scam
- Romance scam
- Dating app scam
- Emotional manipulation for money

5_delivery_customer_care_scam
- Fake courier
- Delivery problem
- Customer support scam
- Parcel issue

6_government_police_scam
- Police
- CBI
- Customs
- Court notice
- Government officer
- Legal threats

7_investment_money_scam
- Investment plans
- Trading groups
- Crypto
- Stock tips
- Ponzi schemes

8_kyc_fraud
- KYC update
- Aadhaar linking
- PAN linking
- Account verification requests

Instructions:
- Read the ENTIRE transcript.
- Determine the MAIN scam mechanism.
- Ignore isolated keywords.
- Classify based on overall intent and attack strategy.

Transcript:
{text}

Respond ONLY with one exact class name matching the given class names do not try to respond with class number or partial name. exact class name as it is must be returned
"""


def _ollama_ping(model):
    try:
        payload = json.dumps({"name": model}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_chat(model, prompt):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0}  # deterministic
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"].strip()


def read_transcript(filepath):
    """Read and truncate transcript to first ~1500 chars to keep inference fast."""
    text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
    # Take first 1500 chars — enough context for classification, keeps it fast
    return text[:1500]


def classify_transcript(text, model=MODEL, retries=2):
    """
    Classify a transcript into one of the 8 classes.
    Returns the class string. Retries if output is not a valid class.
    """
    raw = ""
    for attempt in range(retries + 1):
        raw = _ollama_chat(model, CLASSIFY_PROMPT.format(text=text))
        # Clean up — model might add punctuation or extra text
        cleaned = raw.lower().strip().rstrip('.,;:!')
        # Try exact match first
        if cleaned in CLASSES:
            return cleaned
        # Try partial match (model might say "class: 1_bank_impersonation" etc)
        for cls in CLASSES:
            if cls.lower().endswith(cleaned) or cleaned.endswith(cls.lower()):
                return cls
        # If still no match, retry
        if attempt < retries:
            print(f"      [retry {attempt+1}] got '{raw[:60]}' — retrying...", end=' ', flush=True)

    # Final fallback — return most mentioned class keyword
    counts = {cls: raw.lower().count(cls.replace('_', ' ')) + raw.lower().count(cls)
              for cls in CLASSES}
    best = max(counts, key=counts.get)
    print(f"      [fallback] using '{best}'")
    return best


def load_all_files(input_folder):
    """Returns list of (original_class, filepath) for all txt files."""
    files = []
    base = Path(input_folder)
    for cls in CLASSES:
        class_dir = base / cls
        if not class_dir.exists():
            print(f"  [WARN] Class folder not found: {class_dir}")
            continue
        for txt_file in sorted(class_dir.glob("*.txt")):
            files.append((cls, txt_file))
    return files


def main():
    parser = argparse.ArgumentParser(description='Context-aware scam transcript reclassifier (8→8)')
    parser.add_argument('--input',  required=True, help='Input folder (8-class dataset)')
    parser.add_argument('--output', required=True, help='Output folder for reclassified dataset')
    parser.add_argument('--model',  default=MODEL, help=f'Ollama model (default: {MODEL})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip files already in output folder')
    args = parser.parse_args()

    # ── Ping Ollama ──────────────────────────────────────
    print(f"\nPinging Ollama ({args.model}) at {OLLAMA_HOST} ...")
    if not _ollama_ping(args.model):
        print(f"[FATAL] Ollama not reachable or model '{args.model}' not found.")
        print(f"  Run:  ollama serve   (in another terminal)")
        print(f"  Run:  ollama pull {args.model}")
        sys.exit(1)
    print(f"  [OK] {args.model} is ready.\n")

    # ── Setup output folders ─────────────────────────────
    out_base = Path(args.output)
    for cls in CLASSES:
        (out_base / cls).mkdir(parents=True, exist_ok=True)

    # ── Load all files ───────────────────────────────────
    all_files = load_all_files(args.input)
    total = len(all_files)
    print(f"Found {total} transcripts across {len(CLASSES)} classes.\n")

    if total == 0:
        print("No files found. Check --input path and folder structure.")
        sys.exit(1)

    # ── Classify ─────────────────────────────────────────
    # Track moves: {(old_class, new_class): [filenames]}
    moves = defaultdict(list)
    stayed = defaultdict(int)
    results = []  # (filename, old_class, new_class)

    for i, (orig_class, filepath) in enumerate(all_files, 1):
        out_file = out_base / orig_class / filepath.name

        # Resume: skip if already exists in output
        if args.resume:
            # Check if file exists in any output class folder
            existing = None
            for cls in CLASSES:
                candidate = out_base / cls / filepath.name
                if candidate.exists():
                    existing = cls
                    break
            if existing:
                print(f"  [{i:4}/{total}] SKIP (already in {existing}): {filepath.name}")
                results.append((filepath.name, orig_class, existing))
                if orig_class == existing:
                    stayed[orig_class] += 1
                else:
                    moves[(orig_class, existing)].append(filepath.name)
                continue

        print(f"  [{i:4}/{total}] {orig_class}/{filepath.name} ...", end=' ', flush=True)

        text = read_transcript(filepath)
        new_class = classify_transcript(text, model=args.model)

        # Copy to new location
        dest = out_base / new_class / filepath.name
        shutil.copy2(filepath, dest)

        results.append((filepath.name, orig_class, new_class))

        if new_class == orig_class:
            stayed[orig_class] += 1
            print(f"→ {new_class} [same]")
        else:
            moves[(orig_class, new_class)].append(filepath.name)
            print(f"→ {new_class}  *** MOVED from {orig_class} ***")

    # ── Summary ──────────────────────────────────────────
    total_moved = sum(len(v) for v in moves.values())
    total_stayed = sum(stayed.values())

    print("\n" + "="*60)
    print("  RECLASSIFICATION SUMMARY (8 → 8)")
    print("="*60)
    print(f"\n  Total files    : {total}")
    print(f"  Stayed same    : {total_stayed}  ({total_stayed/total*100:.1f}%)")
    print(f"  Moved          : {total_moved}   ({total_moved/total*100:.1f}%)")

    print(f"\n  New class distribution:")
    new_counts = defaultdict(int)
    for _, _, new_cls in results:
        new_counts[new_cls] += 1
    for cls in CLASSES:
        old_count = sum(1 for orig, _ in [(o, f) for f, o, n in results] if orig == cls)
        print(f"    {cls:<30} {old_count:>3} → {new_counts[cls]:>3}")

    if total_moved == 0:
        print("\n  No files were moved — keyword-based classes already match semantic classes.")
    elif total_moved <= 20:
        print(f"\n  Moved files ({total_moved}):")
        for (old_cls, new_cls), filenames in sorted(moves.items()):
            print(f"\n    {old_cls} → {new_cls}  ({len(filenames)} files):")
            for fn in filenames:
                print(f"      {fn}")
    else:
        print(f"\n  Moved files by route ({total_moved} total):")
        for (old_cls, new_cls), filenames in sorted(moves.items(), key=lambda x: -len(x[1])):
            print(f"    {old_cls:<30} → {new_cls:<30} : {len(filenames)} files")

    # Save full results to JSON for reference
    results_path = out_base / "reclassification_log.json"
    log = {
        "total": total,
        "moved": total_moved,
        "stayed": total_stayed,
        "moves_by_route": {
            f"{o} -> {n}": files for (o, n), files in moves.items()
        },
        "all_results": [
            {"file": f, "original_class": o, "new_class": n, "moved": o != n}
            for f, o, n in results
        ]
    }
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Full log saved → {results_path}")
    print(f"  Reclassified dataset → {out_base}/")
    print("="*60)


if __name__ == "__main__":
    main()
