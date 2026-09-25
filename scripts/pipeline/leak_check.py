from pathlib import Path
import re
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def load_split(split_dir):
    rows = []

    for class_name in CLASSES:
        class_dir = Path(split_dir) / class_name

        if not class_dir.exists():
            continue

        for path in class_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")

            rows.append({
                "filename": path.name,
                "filepath": str(path),
                "true_class": class_name,
                "text": text,
            })

    return pd.DataFrame(rows)


def normalize(text):
    """
    Normalize text for detecting duplicates where superficial
    differences such as phone numbers, URLs, or whitespace exist.
    """
    text = text.lower()

    # Mask URLs
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)

    # Mask phone-number-like sequences
    text = re.sub(r"\+?\d[\d\s\-()]{7,}\d", "<PHONE>", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Check train/test duplicate and near-duplicate leakage."
    )

    parser.add_argument(
        "--train",
        default="dataset_split/train",
        help="Path to train split",
    )

    parser.add_argument(
        "--test",
        default="dataset_split/test",
        help="Path to test split",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="TF-IDF cosine similarity threshold",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TRAIN vs TEST LEAKAGE CHECK")
    print("=" * 70)

    train = load_split(args.train)
    test = load_split(args.test)

    print(f"Train conversations: {len(train):,}")
    print(f"Test conversations:  {len(test):,}")
    print()

    # ---------------------------------------------------------
    # 1. EXACT DUPLICATES
    # ---------------------------------------------------------

    train_exact = {
        text: i for i, text in enumerate(train["text"])
    }

    exact_matches = []

    for j, text in enumerate(test["text"]):
        if text in train_exact:
            i = train_exact[text]

            exact_matches.append({
                "train_filename": train.iloc[i]["filename"],
                "test_filename": test.iloc[j]["filename"],
                "train_class": train.iloc[i]["true_class"],
                "test_class": test.iloc[j]["true_class"],
            })

    # ---------------------------------------------------------
    # 2. NORMALIZED DUPLICATES
    # ---------------------------------------------------------

    train_normalized = {}

    for i, text in enumerate(train["text"]):
        key = normalize(text)
        train_normalized.setdefault(key, []).append(i)

    normalized_matches = []

    for j, text in enumerate(test["text"]):
        key = normalize(text)

        if key in train_normalized:
            for i in train_normalized[key]:
                normalized_matches.append({
                    "train_filename": train.iloc[i]["filename"],
                    "test_filename": test.iloc[j]["filename"],
                    "train_class": train.iloc[i]["true_class"],
                    "test_class": test.iloc[j]["true_class"],
                })

    # ---------------------------------------------------------
    # 3. TF-IDF NEAR DUPLICATES
    # ---------------------------------------------------------

    print("Computing TF-IDF train/test similarity...")

    all_texts = pd.concat(
        [train["text"], test["text"]],
        ignore_index=True
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200000,
    )

    tfidf = vectorizer.fit_transform(all_texts)

    train_matrix = tfidf[:len(train)]
    test_matrix = tfidf[len(train):]

    similarities = cosine_similarity(test_matrix, train_matrix)

    near_matches = []

    for test_idx in range(len(test)):
        train_indices = (
            similarities[test_idx] >= args.threshold
        ).nonzero()[0]

        for train_idx in train_indices:
            score = similarities[test_idx, train_idx]

            near_matches.append({
                "train_filename": train.iloc[train_idx]["filename"],
                "test_filename": test.iloc[test_idx]["filename"],
                "train_class": train.iloc[train_idx]["true_class"],
                "test_class": test.iloc[test_idx]["true_class"],
                "cosine_similarity": round(float(score), 4),
            })

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Exact train/test matches:       {len(exact_matches)}")
    print(
        f"Normalized train/test matches: {len(normalized_matches)}"
    )
    print(
        f"Near-duplicate train/test pairs: {len(near_matches)}"
    )

    if near_matches:
        near_matches.sort(
            key=lambda x: x["cosine_similarity"],
            reverse=True
        )

        print()
        print("NEAR-DUPLICATE TRAIN/TEST PAIRS")
        print("-" * 70)

        for match in near_matches:
            print(
                f"{match['cosine_similarity']:.4f} | "
                f"TRAIN: {match['train_filename']} "
                f"({match['train_class']}) | "
                f"TEST: {match['test_filename']} "
                f"({match['test_class']})"
            )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    output_dir = Path("analysis") / "leakage"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(exact_matches).to_csv(
        output_dir / "train_test_exact_matches.csv",
        index=False,
    )

    pd.DataFrame(normalized_matches).to_csv(
        output_dir / "train_test_normalized_matches.csv",
        index=False,
    )

    pd.DataFrame(near_matches).to_csv(
        output_dir / "train_test_near_duplicate_matches.csv",
        index=False,
    )

    summary = {
        "train_conversations": len(train),
        "test_conversations": len(test),
        "exact_train_test_matches": len(exact_matches),
        "normalized_train_test_matches": len(normalized_matches),
        "near_duplicate_train_test_pairs": len(near_matches),
        "near_duplicate_threshold": args.threshold,
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "train_test_leakage_summary.csv",
        index=False,
    )

    print()
    print("Results saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
