#!/usr/bin/env python3

"""
Vocabulary / OOV / Token Fragmentation Analysis

Uses the same simple deterministic tokenization style as the structural
analysis script:

    TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")

This version intentionally does NOT require tokenizer.json or a trained BPE
tokenizer. It compares train/test/YT using the training vocabulary.

Metrics:
- word/token counts
- unique vocabulary
- OOV token count and rate (relative to TRAIN vocabulary)
- unique OOV token count and rate
- average token length
- type-token ratio
- repeated-token rate
- vocabulary overlap with TRAIN
- per-conversation statistics
- top OOV tokens

The eight numbered Scam10K classes are analyzed; yt is treated as a separate
external dataset when supplied as --yt.

Example:
    python yt_analysis.py \
        --train dataset_split/train \
        --test dataset_split/test \
        --yt dataset_split/yt \
        --output analysis/vocabulary_shift
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


CLASSES = [
    "1_banking_kyc_otp_fraud",
    "2_upi_wallet_fraud",
    "3_investment_task_scam",
    "4_digital_arrest_govt_impersonation",
    "5_loan_credit_app_scams",
    "6_delivery_customer_care_scams",
    "7_dating_romance_sextortion",
    "8_legacy_telecom_scams",
]

# Same tokenization style as scam10k_structural_analysis.py
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
WS_RE = re.compile(r"\s+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_dataset(dataset_dir, split_name):
    dataset_dir = Path(dataset_dir)
    records = []

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            print(f"[WARNING] Missing class folder: {class_name}")
            continue

        for path in sorted(class_dir.glob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[WARNING] Could not read {path}: {e}")
                continue

            tokens = tokenize(text)

            records.append(
                {
                    "split": split_name,
                    "filename": path.name,
                    "filepath": str(path.resolve()),
                    "true_class": class_name,
                    "text": text,
                    "tokens": tokens,
                }
            )

    if not records:
        raise RuntimeError(f"No .txt conversations found in {dataset_dir}")

    return records


def safe_rate(num, den):
    return float(num / den) if den else 0.0


def analyze_split(records, train_vocab):
    rows = []

    for row in records:
        tokens = row["tokens"]
        token_counts = Counter(tokens)

        total_tokens = len(tokens)
        unique_tokens = len(token_counts)

        oov_tokens = [t for t in tokens if t not in train_vocab]
        unique_oov = set(oov_tokens)

        repeated_occurrences = sum(
            count - 1 for count in token_counts.values() if count > 1
        )

        alphabetic_tokens = sum(t.isalpha() for t in tokens)
        numeric_tokens = sum(t.isdigit() for t in tokens)

        rows.append(
            {
                "split": row["split"],
                "filename": row["filename"],
                "true_class": row["true_class"],
                "token_count": total_tokens,
                "unique_token_count": unique_tokens,
                "oov_token_count": len(oov_tokens),
                "unique_oov_token_count": len(unique_oov),
                "oov_token_rate_percent": 100 * safe_rate(
                    len(oov_tokens), total_tokens
                ),
                "unique_oov_rate_percent": 100 * safe_rate(
                    len(unique_oov), unique_tokens
                ),
                "train_vocab_overlap_percent": 100 * safe_rate(
                    len(set(tokens) & train_vocab),
                    unique_tokens,
                ),
                "type_token_ratio": safe_rate(unique_tokens, total_tokens),
                "repeated_token_occurrences": repeated_occurrences,
                "repeated_token_rate_percent": 100 * safe_rate(
                    repeated_occurrences, total_tokens
                ),
                "alphabetic_token_percent": 100 * safe_rate(
                    alphabetic_tokens, total_tokens
                ),
                "numeric_token_percent": 100 * safe_rate(
                    numeric_tokens, total_tokens
                ),
                "avg_token_length": (
                    sum(len(t) for t in tokens) / total_tokens
                    if total_tokens
                    else 0.0
                ),
                "character_count": len(row["text"]),
                "whitespace_word_count": len(row["text"].split()),
            }
        )

    return pd.DataFrame(rows)


def top_oov(records, train_vocab, top_n=100):
    counter = Counter()

    for row in records:
        counter.update(t for t in row["tokens"] if t not in train_vocab)

    return pd.DataFrame(
        [
            {
                "token": token,
                "count": count,
                "split": records[0]["split"] if records else "",
            }
            for token, count in counter.most_common(top_n)
        ]
    )


def make_summary(df, train_vocab, split_name):
    total_tokens = int(df["token_count"].sum())
    unique_tokens = int(
        sum(
            1
            for _ in set(
                token
                for row in df.itertuples()
                for token in []
            )
        )
    )

    # Dataset-level vocabulary must be reconstructed from the source records,
    # so this placeholder is replaced by the caller.
    return {
        "split": split_name,
        "conversations": int(len(df)),
        "total_tokens": total_tokens,
        "mean_tokens_per_conversation": float(df["token_count"].mean()),
        "median_tokens_per_conversation": float(df["token_count"].median()),
        "mean_unique_tokens_per_conversation": float(
            df["unique_token_count"].mean()
        ),
        "mean_type_token_ratio": float(df["type_token_ratio"].mean()),
        "mean_oov_token_rate_percent": float(
            df["oov_token_rate_percent"].mean()
        ),
        "mean_unique_oov_rate_percent": float(
            df["unique_oov_rate_percent"].mean()
        ),
        "mean_train_vocab_overlap_percent": float(
            df["train_vocab_overlap_percent"].mean()
        ),
        "mean_repeated_token_rate_percent": float(
            df["repeated_token_rate_percent"].mean()
        ),
        "mean_alphabetic_token_percent": float(
            df["alphabetic_token_percent"].mean()
        ),
        "mean_numeric_token_percent": float(
            df["numeric_token_percent"].mean()
        ),
        "mean_token_length": float(df["avg_token_length"].mean()),
    }


def class_analysis(df):
    return (
        df.groupby(["split", "true_class"])
        .agg(
            conversations=("filename", "count"),
            avg_tokens=("token_count", "mean"),
            median_tokens=("token_count", "median"),
            avg_unique_tokens=("unique_token_count", "mean"),
            avg_oov_rate_percent=("oov_token_rate_percent", "mean"),
            avg_unique_oov_rate_percent=("unique_oov_rate_percent", "mean"),
            avg_vocab_overlap_percent=("train_vocab_overlap_percent", "mean"),
            avg_ttr=("type_token_ratio", "mean"),
            avg_repeated_token_rate_percent=("repeated_token_rate_percent", "mean"),
            avg_token_length=("avg_token_length", "mean"),
        )
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic vocabulary/OOV analysis using regex tokenization."
    )

    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--yt", required=True)

    # Kept only for backward compatibility with the previous command.
    # It is ignored intentionally.
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Deprecated/ignored. This version does not require tokenizer.json.",
    )

    parser.add_argument(
        "--output",
        default="analysis/vocabulary_shift",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VOCABULARY / OOV / TOKENIZATION ANALYSIS")
    print("=" * 70)
    print("Tokenization: TOKEN_RE regex (same as structural analysis)")
    print("Tokenizer file: NOT REQUIRED")
    print()

    train_records = load_dataset(args.train, "train")
    test_records = load_dataset(args.test, "test")
    yt_records = load_dataset(args.yt, "yt")

    print(f"Train conversations: {len(train_records):,}")
    print(f"Test conversations:  {len(test_records):,}")
    print(f"YT conversations:    {len(yt_records):,}")

    train_vocab_counter = Counter(
        token
        for row in train_records
        for token in row["tokens"]
    )
    train_vocab = set(train_vocab_counter)

    print(f"Train vocabulary:    {len(train_vocab):,}")

    all_records = train_records + test_records + yt_records

    frames = []
    summaries = []

    for split_name, records in [
        ("train", train_records),
        ("test", test_records),
        ("yt", yt_records),
    ]:
        df = analyze_split(records, train_vocab)
        frames.append(df)

        # Dataset-level vocabulary for this split
        split_vocab = set(
            token
            for row in records
            for token in row["tokens"]
        )

        summary = make_summary(df, train_vocab, split_name)
        summary["unique_tokens_in_split"] = len(split_vocab)
        summary["unique_oov_tokens_in_split"] = len(
            split_vocab - train_vocab
        )
        summary["dataset_oov_vocabulary_rate_percent"] = 100 * safe_rate(
            len(split_vocab - train_vocab),
            len(split_vocab),
        )
        summary["dataset_train_vocab_overlap_percent"] = 100 * safe_rate(
            len(split_vocab & train_vocab),
            len(split_vocab),
        )

        summaries.append(summary)

        print(f"\n{split_name.upper()}")
        print("-" * 70)
        print(f"Conversations:                 {len(df):,}")
        print(f"Unique tokens:                 {len(split_vocab):,}")
        print(f"Unique OOV tokens:             {len(split_vocab - train_vocab):,}")
        print(
            f"Dataset OOV vocabulary rate:   "
            f"{summary['dataset_oov_vocabulary_rate_percent']:.2f}%"
        )
        print(
            f"Mean message OOV rate:          "
            f"{summary['mean_oov_token_rate_percent']:.2f}%"
        )
        print(
            f"Mean train-vocab overlap:       "
            f"{summary['mean_train_vocab_overlap_percent']:.2f}%"
        )
        print(
            f"Mean repeated-token rate:       "
            f"{summary['mean_repeated_token_rate_percent']:.2f}%"
        )

        top_oov_df = top_oov(records, train_vocab, top_n=100)
        top_oov_df.to_csv(
            output_dir / f"top_oov_words_{split_name}.csv",
            index=False,
        )

    per_message = pd.concat(frames, ignore_index=True)
    per_message.to_csv(
        output_dir / "vocabulary_shift_by_message.csv",
        index=False,
    )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        output_dir / "vocabulary_shift_summary.csv",
        index=False,
    )

    class_df = class_analysis(per_message)
    class_df.to_csv(
        output_dir / "vocabulary_shift_by_class.csv",
        index=False,
    )

    # Save the actual training vocabulary for reproducibility.
    vocab_df = pd.DataFrame(
        [
            {"token": token, "train_count": count}
            for token, count in train_vocab_counter.most_common()
        ]
    )
    vocab_df.to_csv(
        output_dir / "train_vocabulary.csv",
        index=False,
    )

    with open(
        output_dir / "vocabulary_shift_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results written to: {output_dir.resolve()}")
    print()
    print("Files:")
    print("  vocabulary_shift_summary.csv")
    print("  vocabulary_shift_summary.json")
    print("  vocabulary_shift_by_message.csv")
    print("  vocabulary_shift_by_class.csv")
    print("  train_vocabulary.csv")
    print("  top_oov_words_train.csv")
    print("  top_oov_words_test.csv")
    print("  top_oov_words_yt.csv")


if __name__ == "__main__":
    main()
