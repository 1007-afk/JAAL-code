#!/usr/bin/env python3

"""
Scam10K: Conversation Length, Confusion, and Duplicate/Template Analysis

Runs three deterministic analyses:

1. Conversation length analysis
   - token count
   - character count
   - word count
   - vocabulary size
   - type-token ratio
   - short/medium/long bins
   - class-level statistics
   - CSV summaries and plots

2. Scam-category confusion analysis
   - Reads prediction CSVs supplied with --predictions
   - confusion matrix
   - normalized confusion matrix
   - top confusing class pairs
   - per-class precision/recall/F1
   - confusion heatmaps

3. Duplicate/template analysis
   - exact duplicates
   - normalized exact duplicates
   - near-duplicate pairs using TF-IDF cosine similarity
   - duplicate clusters (connected components)
   - repeated template statistics
   - optional split-leakage analysis if a split column/file is available

No LLM calls. Deterministic.

Example:

python scam10k_structural_analysis.py \
    --dataset /path/to/scam_10k_relabeled \
    --output ./scam10k_structural_results

For model predictions:

python scam10k_structural_analysis.py \
    --dataset /path/to/scam_10k_relabeled \
    --output ./scam10k_structural_results \
    --predictions svm_predictions.csv bilstm_v2_predictions.csv

Prediction CSV must contain:
    true_class
    predicted_class

Optional:
    filename
    split
"""

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Expected Scam10K classes
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")

WS_RE = re.compile(r"\s+")

PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def normalize_text(text):
    """
    Conservative normalization for template detection.

    Keeps actual words but removes:
      - case differences
      - URLs
      - phone numbers
      - long numeric IDs
      - punctuation
      - repeated whitespace

    This is intentionally NOT aggressive paraphrase normalization.
    """
    text = text.lower()

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " EMAIL ",
        text,
    )

    # Phone-like numbers
    text = re.sub(
        r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)",
        " PHONE ",
        text,
    )

    # Money / large numeric values
    text = re.sub(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)", " NUM ", text)

    # Punctuation -> spaces
    text = PUNCT_RE.sub(" ", text)

    # Whitespace
    text = WS_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)

    records = []

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            print(f"[WARNING] Missing class folder: {class_name}")
            continue

        files = sorted(class_dir.glob("*.txt"))

        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[WARNING] Could not read {path}: {e}")
                continue

            records.append(
                {
                    "filename": path.name,
                    "filepath": str(path.resolve()),
                    "true_class": class_name,
                    "text": text,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No .txt conversations were found.")

    return df


# ---------------------------------------------------------------------
# 1. Conversation length analysis
# ---------------------------------------------------------------------

def length_analysis(df, output_dir):
    print("\n" + "=" * 70)
    print("1. CONVERSATION LENGTH ANALYSIS")
    print("=" * 70)

    rows = []

    for _, row in df.iterrows():
        text = row["text"]
        tokens = tokenize(text)

        n_tokens = len(tokens)
        n_chars = len(text)
        n_words = len(text.split())
        vocab = len(set(tokens))

        ttr = vocab / n_tokens if n_tokens else 0.0

        rows.append(
            {
                "filename": row["filename"],
                "true_class": row["true_class"],
                "char_count": n_chars,
                "word_count_whitespace": n_words,
                "token_count": n_tokens,
                "vocabulary_size": vocab,
                "type_token_ratio": ttr,
            }
        )

    conv = pd.DataFrame(rows)

    # Dataset-level quartiles are used for consistent bins.
    q1 = conv["token_count"].quantile(0.25)
    q3 = conv["token_count"].quantile(0.75)

    def length_bin(n):
        if n <= q1:
            return "SHORT"
        elif n <= q3:
            return "MEDIUM"
        return "LONG"

    conv["length_bin"] = conv["token_count"].apply(length_bin)

    # Save per-conversation measurements
    conv.to_csv(
        output_dir / "conversation_length_by_conversation.csv",
        index=False,
    )

    # Class-level statistics
    class_stats = (
        conv.groupby("true_class")
        .agg(
            conversations=("filename", "count"),
            avg_tokens=("token_count", "mean"),
            median_tokens=("token_count", "median"),
            std_tokens=("token_count", "std"),
            min_tokens=("token_count", "min"),
            max_tokens=("token_count", "max"),
            avg_characters=("char_count", "mean"),
            median_characters=("char_count", "median"),
            avg_vocabulary=("vocabulary_size", "mean"),
            median_vocabulary=("vocabulary_size", "median"),
            avg_ttr=("type_token_ratio", "mean"),
            median_ttr=("type_token_ratio", "median"),
        )
        .reset_index()
    )

    class_stats.to_csv(
        output_dir / "conversation_length_by_class.csv",
        index=False,
    )

    # Length-bin distribution by class
    bin_counts = pd.crosstab(
        conv["true_class"],
        conv["length_bin"],
    ).reindex(columns=["SHORT", "MEDIUM", "LONG"], fill_value=0)

    bin_pct = (
        bin_counts.div(bin_counts.sum(axis=1), axis=0) * 100
    ).reset_index()

    bin_counts.reset_index().to_csv(
        output_dir / "conversation_length_bins_counts.csv",
        index=False,
    )

    bin_pct.to_csv(
        output_dir / "conversation_length_bins_percent.csv",
        index=False,
    )

    # Overall summary
    overall = {
        "conversations": int(len(conv)),
        "total_tokens": int(conv["token_count"].sum()),
        "mean_tokens": float(conv["token_count"].mean()),
        "median_tokens": float(conv["token_count"].median()),
        "std_tokens": float(conv["token_count"].std()),
        "min_tokens": int(conv["token_count"].min()),
        "max_tokens": int(conv["token_count"].max()),
        "mean_characters": float(conv["char_count"].mean()),
        "median_characters": float(conv["char_count"].median()),
        "mean_vocabulary_size": float(conv["vocabulary_size"].mean()),
        "median_vocabulary_size": float(conv["vocabulary_size"].median()),
        "mean_type_token_ratio": float(conv["type_token_ratio"].mean()),
        "median_type_token_ratio": float(conv["type_token_ratio"].median()),
        "short_threshold_tokens_q1": float(q1),
        "long_threshold_tokens_q3": float(q3),
        "short_count": int((conv["length_bin"] == "SHORT").sum()),
        "medium_count": int((conv["length_bin"] == "MEDIUM").sum()),
        "long_count": int((conv["length_bin"] == "LONG").sum()),
    }

    with open(
        output_dir / "conversation_length_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall, f, indent=2)

    print(f"Conversations: {len(conv):,}")
    print(f"Mean tokens:   {conv['token_count'].mean():,.1f}")
    print(f"Median tokens: {conv['token_count'].median():,.1f}")
    print(f"Min tokens:    {conv['token_count'].min():,}")
    print(f"Max tokens:    {conv['token_count'].max():,}")
    print(f"SHORT <= Q1:   {q1:.1f} tokens")
    print(f"LONG  > Q3:    {q3:.1f} tokens")

    # Optional plots
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(conv["token_count"], bins=50)
        plt.xlabel("Token count")
        plt.ylabel("Number of conversations")
        plt.title("Scam10K Conversation Length Distribution")
        plt.tight_layout()
        plt.savefig(
            output_dir / "conversation_length_distribution.png",
            dpi=200,
        )
        plt.close()

        # Boxplot by class
        data = [
            conv.loc[
                conv["true_class"] == class_name,
                "token_count",
            ].values
            for class_name in CLASSES
            if class_name in set(conv["true_class"])
        ]

        labels = [
            class_name
            for class_name in CLASSES
            if class_name in set(conv["true_class"])
        ]

        plt.figure(figsize=(14, 7))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Token count")
        plt.title("Conversation Length by Scam Category")
        plt.tight_layout()
        plt.savefig(
            output_dir / "conversation_length_by_class.png",
            dpi=200,
        )
        plt.close()

    except ImportError:
        print("[INFO] matplotlib not installed; skipping plots.")

    return conv


# ---------------------------------------------------------------------
# 2. Scam-category confusion analysis
# ---------------------------------------------------------------------

def confusion_analysis(prediction_paths, output_dir):
    print("\n" + "=" * 70)
    print("2. SCAM-CATEGORY CONFUSION ANALYSIS")
    print("=" * 70)

    if not prediction_paths:
        print(
            "[INFO] No prediction CSVs supplied. "
            "Skipping confusion analysis."
        )
        return

    confusion_dir = output_dir / "confusion_analysis"
    confusion_dir.mkdir(exist_ok=True)

    for prediction_path in prediction_paths:
        prediction_path = Path(prediction_path)

        if not prediction_path.exists():
            print(f"[WARNING] Prediction file not found: {prediction_path}")
            continue

        print(f"\nAnalyzing: {prediction_path.name}")

        pred = pd.read_csv(prediction_path)

        true_col_candidates = [
            "true_class",
            "true_label",
            "y_true",
            "actual",
        ]

        pred_col_candidates = [
            "predicted_class",
            "predicted_label",
            "y_pred",
            "prediction",
            "predicted",
        ]

        true_col = next(
            (c for c in true_col_candidates if c in pred.columns),
            None,
        )

        pred_col = next(
            (c for c in pred_col_candidates if c in pred.columns),
            None,
        )

        if true_col is None or pred_col is None:
            print(
                f"[WARNING] Could not identify true/predicted columns "
                f"in {prediction_path.name}"
            )
            print(f"Columns: {list(pred.columns)}")
            continue

        y_true = pred[true_col].astype(str)
        y_pred = pred[pred_col].astype(str)

        labels = [
            c for c in CLASSES
            if c in set(y_true) or c in set(y_pred)
        ]

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels,
        )

        stem = prediction_path.stem

        cm_df.to_csv(
            confusion_dir / f"{stem}_confusion_matrix.csv"
        )

        # Row-normalized confusion matrix
        row_sums = cm.sum(axis=1, keepdims=True)

        cm_norm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

        cm_norm_df = pd.DataFrame(
            cm_norm * 100,
            index=labels,
            columns=labels,
        )

        cm_norm_df.to_csv(
            confusion_dir / f"{stem}_confusion_matrix_normalized.csv"
        )

        # Per-class report
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).T
        report_df.to_csv(
            confusion_dir / f"{stem}_classification_report.csv"
        )

        # Top off-diagonal confusion pairs
        pairs = []

        for i, true_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                if i == j:
                    continue

                count = int(cm[i, j])

                if count > 0:
                    pairs.append(
                        {
                            "true_class": true_label,
                            "predicted_class": predicted_label,
                            "count": count,
                            "rate_within_true_class_percent": (
                                100 * count / cm[i].sum()
                                if cm[i].sum() else 0
                            ),
                        }
                    )

        pairs_df = (
            pd.DataFrame(pairs)
            .sort_values(
                ["count", "rate_within_true_class_percent"],
                ascending=False,
            )
            if pairs
            else pd.DataFrame(
                columns=[
                    "true_class",
                    "predicted_class",
                    "count",
                    "rate_within_true_class_percent",
                ]
            )
        )

        pairs_df.to_csv(
            confusion_dir / f"{stem}_top_confusions.csv",
            index=False,
        )

        # Summary metrics
        summary = {
            "samples": int(len(pred)),
            "accuracy": float((y_true == y_pred).mean()),
            "macro_precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

        with open(
            confusion_dir / f"{stem}_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        print(f"Samples:    {summary['samples']:,}")
        print(f"Accuracy:   {summary['accuracy']:.4f}")
        print(f"Macro-F1:   {summary['macro_f1']:.4f}")
        print(f"Weighted-F1:{summary['weighted_f1']:.4f}")

        if not pairs_df.empty:
            print("\nTop confusion pairs:")
            print(
                pairs_df.head(10).to_string(index=False)
            )

        # Plot heatmaps
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 9))
            im = ax.imshow(cm, interpolation="nearest")

            ax.set_title(f"Confusion Matrix: {stem}")
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")

            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                    )

            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            fig.savefig(
                confusion_dir / f"{stem}_confusion_matrix.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 9))
            im = ax.imshow(cm_norm, interpolation="nearest")

            ax.set_title(
                f"Normalized Confusion Matrix: {stem}"
            )
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")

            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(
                        j,
                        i,
                        f"{cm_norm[i, j] * 100:.1f}",
                        ha="center",
                        va="center",
                    )

            fig.colorbar(im, ax=ax, label="Percent")
            fig.tight_layout()

            fig.savefig(
                confusion_dir / f"{stem}_confusion_matrix_normalized.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        except ImportError:
            print("[INFO] matplotlib not installed; skipping confusion plots.")


# ---------------------------------------------------------------------
# 3. Duplicate/template analysis
# ---------------------------------------------------------------------

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duplicate_analysis(
    df,
    output_dir,
    near_duplicate_threshold=0.90,
    max_near_duplicate_conversations=10000,
):
    print("\n" + "=" * 70)
    print("3. DUPLICATE / TEMPLATE ANALYSIS")
    print("=" * 70)

    duplicate_dir = output_dir / "duplicate_template_analysis"
    duplicate_dir.mkdir(exist_ok=True)

    work = df[
        [
            "filename",
            "filepath",
            "true_class",
            "text",
        ]
    ].copy()

    # ------------------------------------------------------------
    # Exact duplicates
    # ------------------------------------------------------------

    work["exact_hash"] = work["text"].apply(sha256_text)

    exact_groups = (
        work.groupby("exact_hash")
        .agg(
            count=("filename", "count"),
            files=("filename", lambda x: " | ".join(x)),
            classes=("true_class", lambda x: " | ".join(sorted(set(x)))),
        )
        .reset_index()
    )

    exact_groups = exact_groups[
        exact_groups["count"] > 1
    ].sort_values("count", ascending=False)

    exact_groups.to_csv(
        duplicate_dir / "exact_duplicate_groups.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Normalized duplicates
    # ------------------------------------------------------------

    work["normalized_text"] = work["text"].apply(normalize_text)

    work["normalized_hash"] = work["normalized_text"].apply(
        sha256_text
    )

    norm_groups = (
        work.groupby("normalized_hash")
        .agg(
            count=("filename", "count"),
            files=("filename", lambda x: " | ".join(x)),
            classes=("true_class", lambda x: " | ".join(sorted(set(x)))),
        )
        .reset_index()
    )

    norm_groups = norm_groups[
        norm_groups["count"] > 1
    ].sort_values("count", ascending=False)

    norm_groups.to_csv(
        duplicate_dir / "normalized_duplicate_groups.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Template frequency
    # ------------------------------------------------------------

    template_stats = pd.DataFrame(
        {
            "metric": [
                "total_conversations",
                "exact_duplicate_groups",
                "conversations_in_exact_duplicate_groups",
                "normalized_duplicate_groups",
                "conversations_in_normalized_duplicate_groups",
                "exact_duplicate_conversations_excluding_one_per_group",
                "normalized_duplicate_conversations_excluding_one_per_group",
            ],
            "value": [
                len(work),
                len(exact_groups),
                int(exact_groups["count"].sum())
                if not exact_groups.empty
                else 0,
                len(norm_groups),
                int(norm_groups["count"].sum())
                if not norm_groups.empty
                else 0,
                int(
                    (exact_groups["count"] - 1).sum()
                )
                if not exact_groups.empty
                else 0,
                int(
                    (norm_groups["count"] - 1).sum()
                )
                if not norm_groups.empty
                else 0,
            ],
        }
    )

    template_stats.to_csv(
        duplicate_dir / "duplicate_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Near duplicates
    # ------------------------------------------------------------

    print(
        f"Computing TF-IDF near-duplicates "
        f"(threshold={near_duplicate_threshold:.2f})..."
    )

    n = len(work)

    if n <= max_near_duplicate_conversations:
        texts = work["normalized_text"].fillna("").tolist()

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
            norm="l2",
        )

        X = vectorizer.fit_transform(texts)

        # Sparse cosine similarity.
        similarity = cosine_similarity(X)

        near_pairs = []

        for i in range(n):
            # Only upper triangle
            for j in range(i + 1, n):
                sim = float(similarity[i, j])

                if sim >= near_duplicate_threshold:
                    near_pairs.append(
                        {
                            "file_a": work.iloc[i]["filename"],
                            "class_a": work.iloc[i]["true_class"],
                            "file_b": work.iloc[j]["filename"],
                            "class_b": work.iloc[j]["true_class"],
                            "cosine_similarity": sim,
                            "same_class": (
                                work.iloc[i]["true_class"]
                                == work.iloc[j]["true_class"]
                            ),
                        }
                    )

        near_df = pd.DataFrame(near_pairs)

        if not near_df.empty:
            near_df = near_df.sort_values(
                "cosine_similarity",
                ascending=False,
            )

        near_df.to_csv(
            duplicate_dir / "near_duplicate_pairs.csv",
            index=False,
        )

        print(
            f"Near-duplicate pairs >= {near_duplicate_threshold:.2f}: "
            f"{len(near_df)}"
        )

        # --------------------------------------------------------
        # Connected components / template clusters
        # --------------------------------------------------------

        adjacency = defaultdict(set)

        for _, row in near_df.iterrows():
            a = row["file_a"]
            b = row["file_b"]
            adjacency[a].add(b)
            adjacency[b].add(a)

        visited = set()
        clusters = []

        for node in adjacency:
            if node in visited:
                continue

            stack = [node]
            component = []

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(component) > 1:
                clusters.append(component)

        cluster_rows = []

        for cluster_id, members in enumerate(
            sorted(
                clusters,
                key=len,
                reverse=True,
            ),
            start=1,
        ):
            member_rows = work[
                work["filename"].isin(members)
            ]

            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "classes": " | ".join(
                        sorted(
                            member_rows["true_class"].unique()
                        )
                    ),
                    "files": " | ".join(sorted(members)),
                }
            )

        cluster_df = pd.DataFrame(cluster_rows)

        cluster_df.to_csv(
            duplicate_dir / "near_duplicate_clusters.csv",
            index=False,
        )

        # --------------------------------------------------------
        # Cross-class near duplicates
        # --------------------------------------------------------

        if not near_df.empty:
            cross_class = near_df[
                ~near_df["same_class"]
            ].copy()

            cross_class.to_csv(
                duplicate_dir / "cross_class_near_duplicate_pairs.csv",
                index=False,
            )
        else:
            cross_class = pd.DataFrame()

    else:
        print(
            f"[WARNING] {n:,} conversations exceeds "
            f"--max-near-duplicate-conversations "
            f"({max_near_duplicate_conversations:,}). "
            f"Skipping near-duplicate calculation."
        )

        near_df = pd.DataFrame()
        cross_class = pd.DataFrame()
        cluster_df = pd.DataFrame()

    # ------------------------------------------------------------
    # Split leakage if split column exists
    # ------------------------------------------------------------

    split_file = output_dir / "split_assignments.csv"

    if split_file.exists():
        split_df = pd.read_csv(split_file)

        if "filename" in split_df.columns and "split" in split_df.columns:
            work_split = work.merge(
                split_df[["filename", "split"]],
                on="filename",
                how="left",
            )

            work_split.to_csv(
                duplicate_dir / "conversation_split_assignments_used.csv",
                index=False,
            )

            # Exact normalized template across different splits
            split_template = (
                work_split.groupby("normalized_hash")
                .agg(
                    count=("filename", "count"),
                    splits=("split", lambda x: " | ".join(sorted(set(x)))),
                    files=("filename", lambda x: " | ".join(x)),
                )
                .reset_index()
            )

            leakage = split_template[
                split_template["splits"].str.contains(r"\|")
            ]

            leakage.to_csv(
                duplicate_dir / "cross_split_normalized_template_leakage.csv",
                index=False,
            )

            print(
                f"Cross-split normalized template groups: "
                f"{len(leakage)}"
            )

    # ------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------

    summary = {
        "total_conversations": int(len(work)),
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_group_members": (
            int(exact_groups["count"].sum())
            if not exact_groups.empty
            else 0
        ),
        "exact_duplicate_extra_instances": (
            int((exact_groups["count"] - 1).sum())
            if not exact_groups.empty
            else 0
        ),
        "normalized_duplicate_groups": int(len(norm_groups)),
        "normalized_duplicate_group_members": (
            int(norm_groups["count"].sum())
            if not norm_groups.empty
            else 0
        ),
        "normalized_duplicate_extra_instances": (
            int((norm_groups["count"] - 1).sum())
            if not norm_groups.empty
            else 0
        ),
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_pairs": (
            int(len(near_df))
            if not near_df.empty
            else 0
        ),
        "cross_class_near_duplicate_pairs": (
            int(len(cross_class))
            if not cross_class.empty
            else 0
        ),
        "near_duplicate_clusters": (
            int(len(cluster_df))
            if not cluster_df.empty
            else 0
        ),
        "largest_near_duplicate_cluster": (
            int(cluster_df["cluster_size"].max())
            if not cluster_df.empty
            else 0
        ),
    }

    with open(
        duplicate_dir / "duplicate_template_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"Exact duplicate groups:      {summary['exact_duplicate_groups']}")
    print(f"Normalized duplicate groups: {summary['normalized_duplicate_groups']}")
    print(f"Near-duplicate pairs:         {summary['near_duplicate_pairs']}")
    print(
        f"Cross-class near duplicates:  "
        f"{summary['cross_class_near_duplicate_pairs']}"
    )

    return summary


def has_class_folders(path):
    path = Path(path)
    return any((path / class_name).is_dir() for class_name in CLASSES)


def run_analysis_for_split(
    dataset_dir,
    output_dir,
    predictions,
    near_duplicate_threshold,
    max_near_duplicate_conversations,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCAM10K STRUCTURAL ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {Path(dataset_dir).resolve()}")
    print(f"Output:  {output_dir.resolve()}")

    print("\nLoading conversations...")
    df = load_dataset(dataset_dir)

    print(f"Found {len(df):,} conversations.")

    print("\nClass distribution:")
    print(
        df["true_class"]
        .value_counts()
        .reindex(CLASSES)
        .fillna(0)
        .astype(int)
        .to_string()
    )

    # Dataset manifest
    df[
        ["filename", "filepath", "true_class"]
    ].to_csv(
        output_dir / "dataset_manifest.csv",
        index=False,
    )

    # Run analyses
    length_analysis(df, output_dir)

    confusion_analysis(
        predictions,
        output_dir,
    )

    duplicate_analysis(
        df,
        output_dir,
        near_duplicate_threshold=near_duplicate_threshold,
        max_near_duplicate_conversations=max_near_duplicate_conversations,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE FOR SPLIT")
    print("=" * 70)
    print(f"Results written to: {output_dir.resolve()}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scam10K structural and quality analysis."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory (e.g., dataset_split/train or dataset_split)",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: analysis/count/<split_name>)",
    )

    parser.add_argument(
        "--predictions",
        nargs="*",
        default=[],
        help=(
            "One or more prediction CSVs. "
            "Each must contain true_class and predicted_class "
            "(or recognizable equivalents)."
        ),
    )

    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.90,
        help="TF-IDF cosine threshold for near duplicates. Default: 0.90",
    )

    parser.add_argument(
        "--max-near-duplicate-conversations",
        type=int,
        default=10000,
        help="Maximum corpus size for near-duplicate analysis.",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    if has_class_folders(dataset_path):
        split_name = dataset_path.name
        if args.output is None:
            output_dir = Path("analysis") / "count" / split_name
        else:
            out_p = Path(args.output)
            output_dir = out_p if out_p.name == split_name else out_p / split_name

        run_analysis_for_split(
            dataset_path,
            output_dir,
            args.predictions,
            args.near_duplicate_threshold,
            args.max_near_duplicate_conversations,
        )
    else:
        subdirs = [
            d for d in sorted(dataset_path.iterdir())
            if d.is_dir() and has_class_folders(d)
        ]
        if subdirs:
            for subdir in subdirs:
                split_name = subdir.name
                if args.output is None:
                    output_dir = Path("analysis") / "count" / split_name
                else:
                    out_p = Path(args.output)
                    output_dir = out_p if out_p.name == split_name else out_p / split_name

                run_analysis_for_split(
                    subdir,
                    output_dir,
                    args.predictions,
                    args.near_duplicate_threshold,
                    args.max_near_duplicate_conversations,
                )
        else:
            output_dir = Path(args.output) if args.output else Path("analysis") / "count"
            run_analysis_for_split(
                dataset_path,
                output_dir,
                args.predictions,
                args.near_duplicate_threshold,
                args.max_near_duplicate_conversations,
            )


if __name__ == "__main__":
    main()
