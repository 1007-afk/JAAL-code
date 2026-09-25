"""
youtube_conversation_filter.py

Purpose
-------
Filter a balanced YouTube scam-transcript dataset into a clean
external test set containing conversational scam transcripts.

IMPORTANT:
- No conversation generation.
- No paraphrasing.
- No rewriting.
- The original transcript text is preserved exactly.
- The LLM is used ONLY as a classifier/judge.
- Deterministic quality checks are applied after LLM classification.

Pipeline
--------
400 YouTube transcripts
        |
        v
[1] Deterministic pre-filter
        |
        v
[2] LLM classification via Ollama
        |
        v
[3] Deterministic quality gate
        |
        v
[4] Final candidate dataset
        |
        v
[5] Manual verification CSV

Expected input structure
------------------------
yt_dataset_9_hinglish/
├── banking_kyc_otp_fraud/
│   ├── *.txt
│   └── ...
├── digital_arrest_govt_impersonation/
├── loan_credit_app_scams/
├── dating_romance_sextortion/
├── investment_task_scam/
├── upi_wallet_fraud/
├── delivery_customer_care_scams/
└── legacy_telecom_scams/

Requirements
------------
pip install requests pandas tqdm

Ollama
------
Make sure Ollama is running and the model is available:

    ollama pull gemma4:31b-cloud

Run
---
python youtube_conversation_filter.py \
    --dataset "/Users/user/Downloads/FraudCallDS/yt_dataset_9_hinglish" \
    --output "/Users/user/Downloads/FraudCallDS/yt_filter_output"

Optional:
    --model gemma4:31b-cloud
    --batch-size 10

Outputs
-------
all_transcripts.csv
llm_classification.csv
final_filter_results.csv
yt_conversation_candidates.csv
manual_review.csv
filter_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:4b"

EXPECTED_CLASSES = {
    "banking_kyc_otp_fraud",
    "digital_arrest_govt_impersonation",
    "loan_credit_app_scams",
    "dating_romance_sextortion",
    "investment_task_scam",
    "upi_wallet_fraud",
    "delivery_customer_care_scams",
    "legacy_telecom_scams",
}

# Minimum deterministic requirements.
MIN_CHARS = 500
MIN_WORDS = 80

# A final conversational sample should contain a reasonable
# amount of interaction.
MIN_FINAL_WORDS = 80

# We don't want a transcript that is almost entirely a presenter
# talking about the scam.
MAX_NARRATION_RATIO = 0.65

# Promotional YouTube material.
CTA_PATTERNS = [
    r"\blike and subscribe\b",
    r"\blike\s+(this\s+)?video\b",
    r"\bsubscribe\s+(to\s+the\s+)?channel\b",
    r"\bsubscribe\s+now\b",
    r"\bhit the bell\b",
    r"\bpress the bell\b",
    r"\bcomment below\b",
    r"\bshare this video\b",
    r"\bfollow (our|my|this) channel\b",
]

# Obvious tutorial / awareness signals.
TUTORIAL_PATTERNS = [
    r"\btoday we (will|are going to) discuss\b",
    r"\bin this video\b",
    r"\btoday's video\b",
    r"\blet me explain\b",
    r"\bhow to (report|avoid|identify|protect|recover)\b",
    r"\bhow you can\b",
    r"\bhere is how\b",
    r"\bstep\s*[-:]?\s*[1-9]\b",
    r"\bfirst of all\b",
    r"\bthe first step\b",
    r"\bnext step\b",
    r"\bto report (this|the) fraud\b",
    r"\byou should (always|never)\b",
    r"\bcyber crime helpline\b",
    r"\bcybercrime portal\b",
    r"\bawareness\b",
    r"\bawareness video\b",
]

# Narration / reporting signals.
NARRATION_PATTERNS = [
    r"\bthe victim\b",
    r"\bthe scammer\b",
    r"\bthe fraudster\b",
    r"\bthe caller\b",
    r"\bthe accused\b",
    r"\baccording to\b",
    r"\bpolice said\b",
    r"\bpolice officials\b",
    r"\breportedly\b",
    r"\bthe incident\b",
    r"\bthis incident\b",
    r"\bin the video\b",
    r"\bthe video shows\b",
    r"\bthe story\b",
    r"\bthe case\b",
]

# Strong dialogue indicators.
DIALOGUE_PATTERNS = [
    r'"[^"]{10,}"',
    r"'[^']{10,}'",
    r"\bscammer\s*:",
    r"\bscamster\s*:",
    r"\bcaller\s*:",
    r"\bvictim\s*:",
    r"\bfraudster\s*:",
    r"\bperson\s*1\s*:",
    r"\bperson\s*2\s*:",
    r"\bspeaker\s*1\s*:",
    r"\bspeaker\s*2\s*:",
    r"\bcaller said\b",
    r"\bvictim said\b",
    r"\bscammer said\b",
    r"\bcaller asked\b",
    r"\bvictim asked\b",
    r"\bscammer asked\b",
    r"\bcaller replied\b",
    r"\bvictim replied\b",
    r"\bscammer replied\b",
    r"\bhe replied\b",
    r"\bshe replied\b",
]

# Scam vocabulary. This is NOT used to generate anything.
# It is only used to make sure the retained text still contains
# evidence of the relevant scam topic.
SCAM_TERMS = [
    # Banking / KYC
    "kyc", "otp", "bank account", "bank", "debit card",
    "credit card", "net banking", "account blocked",

    # UPI / wallet
    "upi", "phonepe", "paytm", "gpay", "google pay",
    "wallet", "cashback", "payment request",

    # Investment
    "investment", "invest", "profit", "trading", "crypto",
    "cryptocurrency", "task", "commission", "withdrawal",

    # Digital arrest
    "arrest", "police", "court", "warrant", "narcotics",
    "crime branch", "cyber cell", "illegal parcel",

    # Loan
    "loan", "credit", "emi", "processing fee", "loan app",
    "disbursement", "interest",

    # Delivery
    "delivery", "parcel", "courier", "order", "shipment",
    "customer care", "refund",

    # Dating
    "dating", "romance", "girlfriend", "boyfriend", "nude",
    "sextortion", "video call", "blackmail",

    # Telecom
    "sim", "mobile", "telecom", "jio", "airtel", "vi",
    "bsnl", "number",
]


# ============================================================
# BASIC TEXT FUNCTIONS
# ============================================================

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def sentence_count(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
    return sum(bool(s.strip()) for s in sentences)


def count_patterns(text: str, patterns: List[str]) -> int:
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return count


def contains_any(text: str, terms: List[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def estimate_dialogue_score(text: str) -> float:
    """
    Deterministic dialogue score.

    This is intentionally conservative. It does not claim that
    the transcript is a conversation. It only estimates whether
    there are obvious dialogue signals.
    """
    score = 0.0

    dialogue_hits = count_patterns(text, DIALOGUE_PATTERNS)

    if dialogue_hits >= 1:
        score += 0.35

    if dialogue_hits >= 3:
        score += 0.25

    if dialogue_hits >= 6:
        score += 0.20

    # Explicit speaker labels are strong evidence.
    speaker_labels = re.findall(
        r"\b(?:scammer|caller|victim|fraudster|speaker\s*[12]|person\s*[12])\s*:",
        text,
        flags=re.IGNORECASE,
    )

    unique_speakers = set(x.lower() for x in speaker_labels)

    if len(unique_speakers) >= 2:
        score += 0.40

    # Quoted dialogue.
    quoted_segments = re.findall(r'"[^"]{10,}"', text)
    if len(quoted_segments) >= 2:
        score += 0.30

    return min(score, 1.0)


def estimate_narration_score(text: str) -> float:
    """
    Deterministic narration estimate.

    Again, this is only a pre-filter signal.
    """
    wc = max(word_count(text), 1)

    narration_hits = count_patterns(text, NARRATION_PATTERNS)
    tutorial_hits = count_patterns(text, TUTORIAL_PATTERNS)
    cta_hits = count_patterns(text, CTA_PATTERNS)

    # Normalize approximately by document length.
    narration_density = narration_hits / max(wc / 100.0, 1.0)
    tutorial_density = tutorial_hits / max(wc / 100.0, 1.0)

    score = 0.0

    if narration_hits >= 2:
        score += 0.25

    if narration_hits >= 5:
        score += 0.20

    if tutorial_hits >= 2:
        score += 0.30

    if tutorial_hits >= 5:
        score += 0.15

    if cta_hits >= 1:
        score += 0.20

    if narration_density > 3:
        score += 0.10

    if tutorial_density > 3:
        score += 0.10

    return min(score, 1.0)


def deterministic_precheck(text: str) -> Dict:
    """
    Fast deterministic screening before sending a transcript
    to the LLM.
    """

    wc = word_count(text)
    chars = len(text)

    dialogue_score = estimate_dialogue_score(text)
    narration_score = estimate_narration_score(text)

    cta_hits = count_patterns(text, CTA_PATTERNS)
    tutorial_hits = count_patterns(text, TUTORIAL_PATTERNS)

    has_dialogue_signal = dialogue_score >= 0.35

    # We do NOT reject everything without dialogue markers because
    # many ASR transcripts have no quotation marks or speaker labels.
    #
    # Instead, transcripts are classified into:
    #   LIKELY
    #   REVIEW
    #   WEAK
    #
    if wc < MIN_WORDS or chars < MIN_CHARS:
        precheck = "REJECT_TOO_SHORT"

    elif (
        tutorial_hits >= 4
        and dialogue_score < 0.35
    ):
        precheck = "REJECT_TUTORIAL"

    elif (
        cta_hits >= 2
        and dialogue_score < 0.35
    ):
        precheck = "REJECT_PROMOTIONAL"

    elif dialogue_score >= 0.65 and narration_score < 0.50:
        precheck = "LIKELY_CONVERSATION"

    elif has_dialogue_signal:
        precheck = "REVIEW"

    else:
        precheck = "WEAK_DIALOGUE_SIGNAL"

    return {
        "word_count": wc,
        "char_count": chars,
        "sentence_count": sentence_count(text),
        "det_dialogue_score": round(dialogue_score, 4),
        "det_narration_score": round(narration_score, 4),
        "cta_hits": cta_hits,
        "tutorial_hits": tutorial_hits,
        "det_precheck": precheck,
    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(dataset_dir: Path) -> pd.DataFrame:

    rows = []

    for class_dir in sorted(dataset_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        if class_name not in EXPECTED_CLASSES:
            print(f"[WARNING] Unexpected folder skipped: {class_name}")
            continue

        txt_files = sorted(class_dir.glob("*.txt"))

        print(f"{class_name}: {len(txt_files)} files")

        for txt_file in txt_files:

            try:
                text = txt_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as exc:
                print(f"[ERROR] Could not read {txt_file}: {exc}")
                continue

            text = text.strip()

            precheck = deterministic_precheck(text)

            rows.append({
                "source_file": txt_file.name,
                "source_path": str(txt_file),
                "true_class": class_name,
                "original_text": text,
                **precheck,
            })

    return pd.DataFrame(rows)


# ============================================================
# OLLAMA
# ============================================================

SYSTEM_PROMPT = r"""
You are a strict dataset-quality classifier.

Your task is ONLY to classify whether a supplied YouTube transcript
contains a substantial conversational interaction between participants.

You must NOT:
- rewrite the transcript
- summarize the transcript
- generate dialogue
- invent missing dialogue
- paraphrase anything
- add any text from the transcript

You are only a judge/filter.

Definitions:

CONVERSATION
------------
The transcript contains a substantial back-and-forth interaction
between two or more participants, such as:
- scammer and victim
- caller and victim
- fraudster and target
- customer-care scammer and customer

The interaction should form a meaningful portion of the transcript.

REENACTMENT
-----------
The transcript contains a staged/dramatized conversation that is
clearly presented as an example, role-play, reenactment, or dramatization.

This is still conversational text, but it is NOT a real interaction.

AWARENESS
---------
The transcript is primarily a presenter explaining, discussing,
describing, or warning about a scam. It may contain short quoted
dialogue, but the majority is narration/explanation.

TUTORIAL
--------
The transcript primarily explains how to report, avoid, detect,
recover from, or protect against scams.

NARRATION
---------
The transcript primarily tells a story, reports an incident,
describes what happened, or discusses a case rather than containing
substantial back-and-forth interaction.

UNCLEAR
-------
The transcript is ambiguous, badly transcribed, incomplete, or does
not provide enough evidence.

Important:
A transcript is NOT CONVERSATION merely because it mentions:
"the scammer said..."
A substantial back-and-forth interaction is required.

Also distinguish:
- "A presenter describes a conversation" -> AWARENESS/NARRATION
- Actual transcript of the participants talking -> CONVERSATION
- Staged role-play -> REENACTMENT

Return ONLY valid JSON.

Required format:

{
  "label": "CONVERSATION | REENACTMENT | AWARENESS | TUTORIAL | NARRATION | UNCLEAR",
  "confidence": 0.0,
  "has_two_or_more_participants": true,
  "has_substantial_back_and_forth": true,
  "is_mostly_narration": false,
  "is_mostly_tutorial": false,
  "is_reenactment": false,
  "reason": "short reason"
}

Do not return markdown.
Do not return additional fields.
"""


def make_llm_prompt(text: str) -> str:
    return f"""
Classify the following transcript.

Do not rewrite it.
Do not summarize it.
Do not generate anything based on it.

TRANSCRIPT START
----------------
{text}
----------------
TRANSCRIPT END
"""


def call_ollama(
    text: str,
    model: str,
    timeout: int = 180,
) -> Dict:

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": make_llm_prompt(text),
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    result = response.json()

    content = result["message"]["content"].strip()

    # Remove accidental markdown fences.
    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
    )

    parsed = json.loads(content)

    required = [
        "label",
        "confidence",
        "has_two_or_more_participants",
        "has_substantial_back_and_forth",
        "is_mostly_narration",
        "is_mostly_tutorial",
        "is_reenactment",
        "reason",
    ]

    for key in required:
        if key not in parsed:
            raise ValueError(f"Missing field: {key}")

    valid_labels = {
        "CONVERSATION",
        "REENACTMENT",
        "AWARENESS",
        "TUTORIAL",
        "NARRATION",
        "UNCLEAR",
    }

    if parsed["label"] not in valid_labels:
        raise ValueError(
            f"Invalid label: {parsed['label']}"
        )

    return parsed


# ============================================================
# LLM FILTERING
# ============================================================

def run_llm_filter(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:

    results = []

    # Only reject obvious cases deterministically.
    # Everything uncertain goes to the LLM.
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="LLM filtering",
    ):

        source_file = row["source_file"]

        precheck = row["det_precheck"]

        if precheck in {
            "REJECT_TOO_SHORT",
            "REJECT_TUTORIAL",
            "REJECT_PROMOTIONAL",
        }:

            result = {
                "source_file": source_file,
                "llm_label": "DETERMINISTIC_REJECT",
                "llm_confidence": 1.0,
                "has_two_or_more_participants": False,
                "has_substantial_back_and_forth": False,
                "is_mostly_narration": True,
                "is_mostly_tutorial": (
                    precheck == "REJECT_TUTORIAL"
                ),
                "is_reenactment": False,
                "llm_reason": precheck,
                "llm_status": "SKIPPED",
            }

            results.append(result)
            continue

        try:

            llm_result = call_ollama(
                row["original_text"],
                model=model,
            )

            results.append({
                "source_file": source_file,
                "llm_label": llm_result["label"],
                "llm_confidence": float(
                    llm_result["confidence"]
                ),
                "has_two_or_more_participants": bool(
                    llm_result["has_two_or_more_participants"]
                ),
                "has_substantial_back_and_forth": bool(
                    llm_result["has_substantial_back_and_forth"]
                ),
                "is_mostly_narration": bool(
                    llm_result["is_mostly_narration"]
                ),
                "is_mostly_tutorial": bool(
                    llm_result["is_mostly_tutorial"]
                ),
                "is_reenactment": bool(
                    llm_result["is_reenactment"]
                ),
                "llm_reason": llm_result["reason"],
                "llm_status": "OK",
            })

        except Exception as exc:

            print(
                f"\n[LLM ERROR] {source_file}: {exc}"
            )

            results.append({
                "source_file": source_file,
                "llm_label": "LLM_ERROR",
                "llm_confidence": 0.0,
                "has_two_or_more_participants": False,
                "has_substantial_back_and_forth": False,
                "is_mostly_narration": False,
                "is_mostly_tutorial": False,
                "is_reenactment": False,
                "llm_reason": str(exc),
                "llm_status": "ERROR",
            })

    return pd.DataFrame(results)


# ============================================================
# FINAL DETERMINISTIC QUALITY GATE
# ============================================================

def final_quality_gate(row: pd.Series) -> Dict:
    """
    Apply conservative deterministic rules after LLM classification.

    IMPORTANT:
    The original transcript remains untouched.
    """

    label = row["llm_label"]

    confidence = float(row["llm_confidence"])

    word_count_value = int(row["word_count"])

    participant_ok = bool(
        row["has_two_or_more_participants"]
    )

    interaction_ok = bool(
        row["has_substantial_back_and_forth"]
    )

    mostly_narration = bool(
        row["is_mostly_narration"]
    )

    mostly_tutorial = bool(
        row["is_mostly_tutorial"]
    )

    # --------------------------------------------
    # Hard rejection
    # --------------------------------------------

    if label != "CONVERSATION":
        return {
            "final_status": "REJECT",
            "final_reason": f"LLM_LABEL_{label}",
        }

    if confidence < 0.75:
        return {
            "final_status": "MANUAL_REVIEW",
            "final_reason": "LOW_LLM_CONFIDENCE",
        }

    if word_count_value < MIN_FINAL_WORDS:
        return {
            "final_status": "REJECT",
            "final_reason": "TOO_SHORT",
        }

    if not participant_ok:
        return {
            "final_status": "MANUAL_REVIEW",
            "final_reason": "PARTICIPANTS_NOT_CONFIRMED",
        }

    if not interaction_ok:
        return {
            "final_status": "MANUAL_REVIEW",
            "final_reason": "BACK_AND_FORTH_NOT_CONFIRMED",
        }

    if mostly_narration:
        return {
            "final_status": "REJECT",
            "final_reason": "MOSTLY_NARRATION",
        }

    if mostly_tutorial:
        return {
            "final_status": "REJECT",
            "final_reason": "MOSTLY_TUTORIAL",
        }

    # --------------------------------------------
    # Scam-topic check
    # --------------------------------------------

    if not contains_any(
        row["original_text"],
        SCAM_TERMS,
    ):
        return {
            "final_status": "MANUAL_REVIEW",
            "final_reason": "NO_CLEAR_SCAM_TERMS",
        }

    # --------------------------------------------
    # Final acceptance
    # --------------------------------------------

    return {
        "final_status": "ACCEPT_CANDIDATE",
        "final_reason": "PASSED_ALL_FILTERS",
    }


def apply_quality_gate(
    df: pd.DataFrame,
) -> pd.DataFrame:

    gate_results = []

    for _, row in df.iterrows():
        gate_results.append(
            final_quality_gate(row)
        )

    gate_df = pd.DataFrame(gate_results)

    return pd.concat(
        [
            df.reset_index(drop=True),
            gate_df,
        ],
        axis=1,
    )


# ============================================================
# MANUAL REVIEW FILE
# ============================================================

def create_manual_review(
    df: pd.DataFrame,
) -> pd.DataFrame:

    review = df[
        df["final_status"].isin(
            ["ACCEPT_CANDIDATE", "MANUAL_REVIEW"]
        )
    ].copy()

    review["manual_decision"] = ""
    review["manual_notes"] = ""

    # Put important columns first.
    columns = [
        "source_file",
        "true_class",
        "word_count",
        "sentence_count",
        "det_dialogue_score",
        "det_narration_score",
        "llm_label",
        "llm_confidence",
        "has_two_or_more_participants",
        "has_substantial_back_and_forth",
        "is_reenactment",
        "llm_reason",
        "final_status",
        "final_reason",
        "manual_decision",
        "manual_notes",
        "original_text",
    ]

    columns = [
        c for c in columns
        if c in review.columns
    ]

    return review[columns]


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    all_df: pd.DataFrame,
    final_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    summary = {
        "total_transcripts": int(len(all_df)),
        "total_classes": int(
            all_df["true_class"].nunique()
        ),
        "llm_labels": (
            final_df["llm_label"]
            .value_counts()
            .to_dict()
        ),
        "final_status": (
            final_df["final_status"]
            .value_counts()
            .to_dict()
        ),
        "accepted_candidates": int(
            (
                final_df["final_status"]
                == "ACCEPT_CANDIDATE"
            ).sum()
        ),
        "manual_review": int(
            (
                final_df["final_status"]
                == "MANUAL_REVIEW"
            ).sum()
        ),
        "rejected": int(
            (
                final_df["final_status"]
                == "REJECT"
            ).sum()
        ),
        "accepted_candidates_by_class": (
            final_df[
                final_df["final_status"]
                == "ACCEPT_CANDIDATE"
            ]["true_class"]
            .value_counts()
            .to_dict()
        ),
        "model": final_df.attrs.get(
            "model",
            DEFAULT_MODEL,
        ),
    }

    with open(
        output_dir / "filter_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("FILTER SUMMARY")
    print("=" * 70)

    print(
        f"Total transcripts:       "
        f"{summary['total_transcripts']}"
    )

    print(
        f"Accepted candidates:     "
        f"{summary['accepted_candidates']}"
    )

    print(
        f"Manual review:           "
        f"{summary['manual_review']}"
    )

    print(
        f"Rejected:                "
        f"{summary['rejected']}"
    )

    print("\nAccepted candidates by class:")

    for cls, count in sorted(
        summary["accepted_candidates_by_class"].items()
    ):
        print(f"  {cls}: {count}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Filter YouTube scam transcripts into "
            "conversational external-test candidates."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to yt_dataset_9_hinglish",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_dir}"
        )

    # --------------------------------------------------------
    # Check Ollama
    # --------------------------------------------------------

    print("\nChecking Ollama...")

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        print("Ollama is running.")

    except Exception as exc:

        raise RuntimeError(
            "Could not connect to Ollama at "
            "http://localhost:11434\n"
            "Start Ollama first.\n"
            f"Error: {exc}"
        )

    # --------------------------------------------------------
    # Load transcripts
    # --------------------------------------------------------

    print("\nLoading transcripts...")

    all_df = load_dataset(dataset_dir)

    print(
        f"\nFound {len(all_df)} transcripts."
    )

    if len(all_df) == 0:
        raise RuntimeError(
            "No .txt transcripts found."
        )

    # Save raw inventory.
    all_df.drop(
        columns=["original_text"],
    ).to_csv(
        output_dir / "all_transcripts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Show class balance
    # --------------------------------------------------------

    print("\nInput class distribution:")

    print(
        all_df["true_class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Deterministic pre-filter statistics
    # --------------------------------------------------------

    print("\nDeterministic pre-filter:")

    print(
        all_df["det_precheck"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # LLM classification
    # --------------------------------------------------------

    print(
        f"\nRunning LLM filter with: {args.model}"
    )

    print(
        "IMPORTANT: The LLM will ONLY classify "
        "the transcript. It will not generate or "
        "modify any test-set text."
    )

    llm_df = run_llm_filter(
        all_df,
        model=args.model,
    )

    llm_df.to_csv(
        output_dir / "llm_classification.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = all_df.merge(
        llm_df,
        on="source_file",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Final deterministic quality gate
    # --------------------------------------------------------

    print(
        "\nApplying final deterministic quality gate..."
    )

    final_df = apply_quality_gate(
        merged
    )

    final_df.attrs["model"] = args.model

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    final_df.to_csv(
        output_dir / "final_filter_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Candidate dataset
    # --------------------------------------------------------

    candidates = final_df[
        final_df["final_status"]
        == "ACCEPT_CANDIDATE"
    ].copy()

    candidate_columns = [
        "source_file",
        "source_path",
        "true_class",
        "word_count",
        "sentence_count",
        "llm_label",
        "llm_confidence",
        "is_reenactment",
        "original_text",
    ]

    candidates = candidates[
        candidate_columns
    ]

    candidates.to_csv(
        output_dir / "yt_conversation_candidates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Manual review
    # --------------------------------------------------------

    review_df = create_manual_review(
        final_df
    )

    review_df.to_csv(
        output_dir / "manual_review.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    create_summary(
        all_df,
        final_df,
        output_dir,
    )

    print("\nFiles written to:")
    print(output_dir)

    print("\nRecommended next step:")
    print(
        "Open manual_review.csv and manually verify every "
        "ACCEPT_CANDIDATE before using it as the external test set."
    )

    print(
        "\nDO NOT modify original_text when constructing "
        "the final external test set."
    )


if __name__ == "__main__":
    main()
