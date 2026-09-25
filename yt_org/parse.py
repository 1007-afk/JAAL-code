import os
import shutil
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM-classified conversational transcripts into a separate dataset."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to yt_dataset_9_hinglish"
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Path to llm_classification.csv. If omitted, uses dataset/results/llm_classification.csv"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output folder. If omitted, creates yt_conversations_llm beside the dataset."
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="Minimum LLM confidence. Default: 0.75"
    )

    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)

    if args.csv:
        csv_path = os.path.abspath(args.csv)
    else:
        csv_path = os.path.join(dataset, "results", "final_filter_results.csv")

    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.join(
            os.path.dirname(dataset),
            "yt_conversations_llm"
        )

    print("=" * 70)
    print("EXTRACTING LLM-CLASSIFIED CONVERSATIONS")
    print("=" * 70)

    print(f"Dataset: {dataset}")
    print(f"LLM CSV: {csv_path}")
    print(f"Output:  {output_dir}")
    print()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find LLM classification CSV:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    print("CSV columns:")
    print(list(df.columns))
    print()

    # ------------------------------------------------------------
    # Find important columns automatically
    # ------------------------------------------------------------

    label_candidates = [
        "llm_label",
        "label",
        "classification",
        "predicted_label"
    ]

    confidence_candidates = [
        "confidence",
        "llm_confidence"
    ]

    file_candidates = [
        "filepath",
        "file_path",
        "path",
        "filename",
        "file",
        "source_file"
    ]

    class_candidates = [
        "true_class",
        "class",
        "category",
        "label_class"
    ]

    label_col = next(
        (c for c in label_candidates if c in df.columns),
        None
    )

    confidence_col = next(
        (c for c in confidence_candidates if c in df.columns),
        None
    )

    file_col = next(
        (c for c in file_candidates if c in df.columns),
        None
    )

    class_col = next(
        (c for c in class_candidates if c in df.columns),
        None
    )

    if label_col is None:
        raise ValueError(
            "Could not find the LLM label column.\n"
            f"Available columns: {list(df.columns)}"
        )

    if file_col is None:
        raise ValueError(
            "Could not find the filename/path column.\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"Label column:      {label_col}")
    print(f"File column:       {file_col}")

    if confidence_col:
        print(f"Confidence column: {confidence_col}")

    if class_col:
        print(f"Class column:      {class_col}")

    print()

    # ------------------------------------------------------------
    # Normalize labels
    # ------------------------------------------------------------

    df["_label_clean"] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ------------------------------------------------------------
    # Select CONVERSATION
    # ------------------------------------------------------------

    conversations = df[
        df["_label_clean"] == "CONVERSATION"
    ].copy()

    print(f"LLM CONVERSATION candidates: {len(conversations)}")

    # ------------------------------------------------------------
    # Confidence filter
    # ------------------------------------------------------------

    if confidence_col:
        conversations["_confidence"] = pd.to_numeric(
            conversations[confidence_col],
            errors="coerce"
        )

        before = len(conversations)

        conversations = conversations[
            conversations["_confidence"] >= args.min_confidence
        ].copy()

        print(
            f"After confidence >= {args.min_confidence}: "
            f"{len(conversations)}"
        )

        print(
            f"Removed due to low confidence: "
            f"{before - len(conversations)}"
        )

    print()

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    copied = []
    missing = []

    # ------------------------------------------------------------
    # Copy files while preserving scam class folders
    # ------------------------------------------------------------

    for _, row in conversations.iterrows():

        source = str(row[file_col]).strip()

        # Handle absolute paths
        if os.path.isabs(source):
            source_path = source
        else:
            # First try relative to dataset
            source_path = os.path.join(dataset, source)

            # If that doesn't exist, search by basename
            if not os.path.exists(source_path):
                source_path = None

                basename = os.path.basename(source)

                for root, dirs, files in os.walk(dataset):
                    if basename in files:
                        source_path = os.path.join(root, basename)
                        break

        if not source_path or not os.path.exists(source_path):
            missing.append(source)
            continue

        # Determine original class folder
        if class_col and pd.notna(row[class_col]):
            scam_class = str(row[class_col]).strip()
        else:
            # Usually the parent folder is the class
            scam_class = os.path.basename(
                os.path.dirname(source_path)
            )

        class_output_dir = os.path.join(
            output_dir,
            scam_class
        )

        os.makedirs(class_output_dir, exist_ok=True)

        destination = os.path.join(
            class_output_dir,
            os.path.basename(source_path)
        )

        shutil.copy2(source_path, destination)

        copied.append({
            "true_class": scam_class,
            "filename": os.path.basename(source_path),
            "source_path": source_path,
            "destination": destination,
            "llm_label": row[label_col],
            "confidence": (
                row["_confidence"]
                if confidence_col
                else None
            )
        })

    # ------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------

    manifest_path = os.path.join(
        output_dir,
        "conversation_manifest.csv"
    )

    manifest = pd.DataFrame(copied)
    manifest.to_csv(manifest_path, index=False)

    # ------------------------------------------------------------
    # Save missing files
    # ------------------------------------------------------------

    if missing:
        missing_path = os.path.join(
            output_dir,
            "missing_files.txt"
        )

        with open(missing_path, "w", encoding="utf-8") as f:
            for item in missing:
                f.write(item + "\n")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"LLM CONVERSATION rows: {len(conversations)}")
    print(f"Files successfully copied: {len(copied)}")
    print(f"Files missing: {len(missing)}")
    print()

    if copied:
        print("Output distribution:")

        copied_df = pd.DataFrame(copied)

        counts = copied_df["true_class"].value_counts()

        for scam_class, count in counts.items():
            print(f"  {scam_class}: {count}")

    print()
    print(f"Output dataset: {output_dir}")
    print(f"Manifest:       {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
