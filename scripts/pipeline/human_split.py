from pathlib import Path
import random
import csv

# ============================================================
# CONFIGURATION
# ============================================================

# Entire dataset — all 9,677 synthetic conversations
DATASET_DIR = Path("dataset")

OUTPUT_DIR = Path("human_validation")
HUMAN_SET_DIR = OUTPUT_DIR / "human_set"

ANSWER_KEY = OUTPUT_DIR / "human_set_answer_key.csv"
METADATA_FILE = OUTPUT_DIR / "human_set_metadata.csv"
DISTRIBUTION_FILE = OUTPUT_DIR / "human_set_distribution.csv"

SAMPLE_FRACTION = 0.10
SEED = 42

# Only the 8 synthetic scam classes.
# YT is excluded.
CLASS_DIRS = [
    "1_banking_kyc_otp_fraud",
    "2_upi_wallet_fraud",
    "3_investment_task_scam",
    "4_digital_arrest_govt_impersonation",
    "5_loan_credit_app_scams",
    "6_delivery_customer_care_scams",
    "7_dating_romance_sextortion",
    "8_legacy_telecom_scams",
]


# ============================================================
# COLLECT ENTIRE DATASET
# ============================================================

def collect_files():

    samples = []

    for class_name in CLASS_DIRS:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for path in sorted(class_dir.glob("*.txt")):

            samples.append({
                "path": path,
                "class_name": class_name
            })

    return samples


# ============================================================
# STRATIFIED SAMPLING BY CLASS
# ============================================================

def stratified_sample(samples, fraction, seed):

    """
    Sample from the complete dataset.

    Stratification is performed ONLY by scam category.
    This preserves the class distribution of the full dataset.
    """

    rng = random.Random(seed)

    groups = {}

    for sample in samples:

        key = sample["class_name"]

        groups.setdefault(key, []).append(sample)

    selected = []

    for class_name, group in groups.items():

        group = group.copy()
        rng.shuffle(group)

        n = round(len(group) * fraction)

        # At least one example from every non-empty class
        n = max(1, n)

        selected.extend(group[:n])

    rng.shuffle(selected)

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HUMAN_SET_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Collect entire synthetic dataset
    # --------------------------------------------------------

    samples = collect_files()

    total = len(samples)

    print("=" * 70)
    print("FULL SYNTHETIC DATASET")
    print("=" * 70)

    print(f"Total conversations: {total}")

    if total != 9677:
        print(
            f"\nWARNING: Expected 9,677 conversations, "
            f"but found {total}."
        )

    # --------------------------------------------------------
    # Create stratified 10% sample
    # --------------------------------------------------------

    selected = stratified_sample(
        samples,
        SAMPLE_FRACTION,
        SEED
    )

    print()
    print("=" * 70)
    print("HUMAN VALIDATION SET")
    print("=" * 70)

    print(f"Sampling fraction: {SAMPLE_FRACTION:.0%}")
    print(f"Selected conversations: {len(selected)}")
    print(f"Random seed: {SEED}")

    # --------------------------------------------------------
    # Distribution containers
    # --------------------------------------------------------

    class_counts = {
        class_name: 0
        for class_name in CLASS_DIRS
    }

    # --------------------------------------------------------
    # Create human annotation files
    # --------------------------------------------------------

    answer_rows = []
    metadata_rows = []

    for idx, item in enumerate(selected, start=1):

        original_path = item["path"]
        class_name = item["class_name"]

        human_id = f"H{idx:04d}"

        # ----------------------------------------------------
        # Copy conversation to neutral ID
        # ----------------------------------------------------

        output_path = HUMAN_SET_DIR / f"{human_id}.txt"

        text = original_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        output_path.write_text(
            text,
            encoding="utf-8"
        )

        # ----------------------------------------------------
        # Count distribution
        # ----------------------------------------------------

        class_counts[class_name] += 1

        # ----------------------------------------------------
        # PRIVATE ANSWER KEY
        # DO NOT GIVE THIS FILE TO HUMAN ANNOTATORS.
        # ----------------------------------------------------

        answer_rows.append({
            "human_id": human_id,
            "ground_truth_class": class_name,
            "original_file": original_path.name,
            "original_path": str(original_path)
        })

        # ----------------------------------------------------
        # Metadata for researcher
        # ----------------------------------------------------

        metadata_rows.append({
            "human_id": human_id,
            "original_file": original_path.name
        })

    # ========================================================
    # SAVE PRIVATE ANSWER KEY
    # ========================================================

    with open(
        ANSWER_KEY,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "human_id",
                "ground_truth_class",
                "original_file",
                "original_path"
            ]
        )

        writer.writeheader()
        writer.writerows(answer_rows)

    # ========================================================
    # SAVE METADATA
    # ========================================================

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "human_id",
                "original_file"
            ]
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    # ========================================================
    # SAVE DISTRIBUTION
    # ========================================================

    with open(
        DISTRIBUTION_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class",
            "full_dataset_count",
            "human_set_count",
            "sampling_rate"
        ])

        for class_name in CLASS_DIRS:

            full_count = sum(
                1
                for sample in samples
                if sample["class_name"] == class_name
            )

            human_count = class_counts[class_name]

            rate = (
                human_count / full_count
                if full_count > 0
                else 0
            )

            writer.writerow([
                class_name,
                full_count,
                human_count,
                f"{rate:.4f}"
            ])

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    for class_name in CLASS_DIRS:

        full_count = sum(
            1
            for sample in samples
            if sample["class_name"] == class_name
        )

        human_count = class_counts[class_name]

        print(
            f"{class_name:45s} "
            f"{human_count:4d} / {full_count:4d} "
            f"({human_count / full_count:.2%})"
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"Human set      : {HUMAN_SET_DIR}")
    print(f"Answer key     : {ANSWER_KEY}")
    print(f"Metadata       : {METADATA_FILE}")
    print(f"Distribution   : {DISTRIBUTION_FILE}")

    print()
    print("IMPORTANT:")
    print(
        f"Give human annotators ONLY the files in "
        f"{HUMAN_SET_DIR}/"
    )
    print(
        "DO NOT give annotators the answer key or metadata."
    )


if __name__ == "__main__":
    main()
