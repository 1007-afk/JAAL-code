from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

# You are running this from:
# FraudCallDS/dataset/

DATASET_ROOT = Path("./dataset")
OUTPUT_ROOT = Path("../dataset_split")

SEED = 42
TEST_SIZE = 0.15

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


# ============================================================
# LOAD ALL FILES
# ============================================================

all_files = []
all_labels = []

for class_name in CLASSES:

    folder = DATASET_ROOT / class_name

    if not folder.exists():
        print(f"[WARNING] Folder not found: {folder}")
        continue

    files = list(folder.rglob("*.txt"))

    print(f"{class_name}: {len(files)} files")

    for f in files:
        all_files.append(f)
        all_labels.append(class_name)


print("\nTotal files:", len(all_files))


# ============================================================
# IMPORTANT:
# THIS IS THE EXACT SAME SPLIT AS YOUR ORIGINAL SCRIPT
# ============================================================

train_files, test_files, train_labels, test_labels = train_test_split(
    all_files,
    all_labels,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=all_labels
)


# ============================================================
# SPLIT ORIGINAL TRAINING SET INTO:
#
# 70% ORIGINAL DATA = TRAIN
# 15% ORIGINAL DATA = VALIDATION
#
# Since original train = 85%:
# validation fraction inside train = 15 / 85
# ============================================================

VAL_FRACTION_OF_TRAIN = 0.15 / 0.85

train_files, val_files, train_labels, val_labels = train_test_split(
    train_files,
    train_labels,
    test_size=VAL_FRACTION_OF_TRAIN,
    random_state=SEED,
    stratify=train_labels
)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_files(files, labels, split):

    output_dir = OUTPUT_ROOT / split

    for class_name in CLASSES:
        (output_dir / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

    for src, label in zip(files, labels):

        dst = output_dir / label / src.name

        # Handle duplicate filenames safely
        if dst.exists():

            counter = 1

            while True:
                new_name = f"{src.stem}_{counter}{src.suffix}"
                dst = output_dir / label / new_name

                if not dst.exists():
                    break

                counter += 1

        shutil.copy2(src, dst)


# ============================================================
# COPY DATA
# ============================================================

print("\nCopying TRAIN...")
copy_files(train_files, train_labels, "train")

print("Copying VALIDATION...")
copy_files(val_files, val_labels, "val")

print("Copying TEST...")
copy_files(test_files, test_labels, "test")


# ============================================================
# SUMMARY
# ============================================================

total = len(all_files)

print("\n" + "=" * 70)
print("FINAL DATASET SPLIT")
print("=" * 70)

print(f"Train:      {len(train_files):>6} ({len(train_files)/total*100:.2f}%)")
print(f"Validation: {len(val_files):>6} ({len(val_files)/total*100:.2f}%)")
print(f"Test:       {len(test_files):>6} ({len(test_files)/total*100:.2f}%)")
print(f"Total:      {total:>6}")

print("\nPer-class split:")
print("-" * 70)

for cls in CLASSES:

    tr = sum(1 for x in train_labels if x == cls)
    va = sum(1 for x in val_labels if x == cls)
    te = sum(1 for x in test_labels if x == cls)

    print(
        f"{cls:<45} "
        f"train={tr:<5} "
        f"val={va:<5} "
        f"test={te:<5}"
    )

print("\nSaved to:")
print(OUTPUT_ROOT.resolve())
