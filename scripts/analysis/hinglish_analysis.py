"""
Hinglish Language Analysis using Ollama LLM

Dataset:
    scam_10k_relabeled/

Expected structure:
    scam_10k_relabeled/
        1_banking_kyc_otp_fraud/
            *.txt
        2_upi_wallet_fraud/
            *.txt
        ...

LLM:
    gemma4:31b-cloud

Labels:
    EN    = English
    HI    = Romanized Hindi/Hindustani
    AMB   = Ambiguous English/Hindi
    OTHER = acronym, proper noun, slang, typo, etc.

Outputs:
    token_counts.csv
    token_language_map.json
    token_language_map.csv
    message_language_analysis.csv
    language_by_class.csv
    language_summary.json
"""



import os
import re
import json
import time
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm
import ollama

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the dataset directory"
)

args = parser.parse_args()
DATASET_DIR = args.dataset

# Automatically determine split name and output directory
SPLIT_NAME = os.path.basename(os.path.normpath(DATASET_DIR))
OUTPUT_DIR = os.path.join("analysis", "hinglish", SPLIT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Change this if your dataset folder has another name/location.


# Ollama model
MODEL = "gemma4:31b-cloud"

# Start with 100.
# Once stable, 300 or 500 is recommended.
BATCH_SIZE = 100

# Number of retry attempts for failed batches
MAX_RETRIES = 4

# Seconds between retries
RETRY_DELAY = 5

# Checkpoint file
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "token_language_map_checkpoint.json")

# Final output files
TOKEN_COUNTS_FILE = os.path.join(OUTPUT_DIR, "token_counts.csv")
TOKEN_MAP_JSON = os.path.join(OUTPUT_DIR, "token_language_map.json")
TOKEN_MAP_CSV = os.path.join(OUTPUT_DIR, "token_language_map.csv")
MESSAGE_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, "message_language_analysis.csv")
CLASS_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, "language_by_class.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "language_summary.json")

# Allowed labels
ALLOWED_LABELS = {"EN", "HI", "AMB", "OTHER"}


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")


def tokenize(text):
    """
    Extract alphabetic tokens.

    Example:
        "Please verify your KYC account"
        ->
        ["please", "verify", "your", "kyc", "account"]
    """
    return TOKEN_PATTERN.findall(text.lower())


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_txt_files(dataset_dir):
    """
    Find TXT files ONLY inside the eight scam-class folders.
    Does not scan .venv, scripts, generated files, etc.
    """

    class_folders = [
        "1_banking_kyc_otp_fraud",
        "2_upi_wallet_fraud",
        "3_investment_task_scam",
        "4_digital_arrest_govt_impersonation",
        "5_loan_credit_app_scams",
        "6_delivery_customer_care_scams",
        "7_dating_romance_sextortion",
        "8_legacy_telecom_scams",
    ]

    files = []

    for class_folder in class_folders:

        folder = os.path.join(
            dataset_dir,
            class_folder
        )

        if not os.path.isdir(folder):
            print(f"WARNING: folder not found: {folder}")
            continue

        for filename in os.listdir(folder):

            if filename.lower().endswith(".txt"):

                filepath = os.path.join(
                    folder,
                    filename
                )

                if os.path.isfile(filepath):
                    files.append(filepath)

    return sorted(files)

def get_class_name(filepath):
    """
    Extract scam class from parent folder.
    """

    parent = os.path.basename(os.path.dirname(filepath))

    return parent


# ============================================================
# READ DATASET
# ============================================================

def load_dataset():
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    files = find_txt_files(DATASET_DIR)

    if not files:
        raise FileNotFoundError(
            f"No .txt files found under: {DATASET_DIR}"
        )

    print(f"Found TXT files: {len(files):,}")

    messages = []

    for filepath in tqdm(files, desc="Reading messages"):
        try:
            with open(
                filepath,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                text = f.read().strip()

        except Exception as e:
            print(f"\nWarning: could not read {filepath}")
            print(e)
            continue

        if not text:
            continue

        class_name = get_class_name(filepath)

        messages.append(
            {
                "filepath": filepath,
                "class": class_name,
                "text": text,
            }
        )

    print(f"Loaded messages: {len(messages):,}")

    return messages


# ============================================================
# BUILD TOKEN COUNTS
# ============================================================

def build_token_counts(messages):
    print()
    print("=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)

    token_counter = Counter()

    message_tokens = []

    for msg in tqdm(messages, desc="Tokenizing"):
        tokens = tokenize(msg["text"])

        token_counter.update(tokens)

        message_tokens.append(tokens)

    print(f"Total token occurrences: {sum(token_counter.values()):,}")
    print(f"Unique tokens: {len(token_counter):,}")

    return token_counter, message_tokens


# ============================================================
# SAVE TOKEN COUNTS
# ============================================================

def save_token_counts(token_counter):

    rows = []

    for token, count in token_counter.most_common():

        rows.append(
            {
                "token": token,
                "count": count,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        TOKEN_COUNTS_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved: {TOKEN_COUNTS_FILE}")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    if not os.path.exists(CHECKPOINT_FILE):
        return {}

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # Validate checkpoint
        cleaned = {}

        for token, label in data.items():

            if label in ALLOWED_LABELS:
                cleaned[token] = label

        print(
            f"Loaded checkpoint: "
            f"{len(cleaned):,} classified tokens"
        )

        return cleaned

    except Exception as e:

        print("Could not load checkpoint:")
        print(e)

        return {}


def save_checkpoint(token_map):

    temp_file = CHECKPOINT_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            token_map,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temp_file,
        CHECKPOINT_FILE
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(tokens):

    token_text = json.dumps(
        tokens,
        ensure_ascii=False
    )

    prompt = f"""
You are a linguistic annotation system for a Hinglish scam-message
dataset from India.

Your task is NOT grammatical tagging.

You MUST classify every token into exactly ONE of these four labels.

LABEL DEFINITIONS:

EN
English word or clearly English lexical item.

HI
Romanized Hindi/Hindustani word or expression.

AMB
The token can reasonably be interpreted as either English or
Romanized Hindi depending on context, and the token itself does
not provide enough evidence to decide.

OTHER
Acronym, abbreviation, proper noun, company/person/place name,
brand, number artifact, slang, typo, URL artifact, technical
identifier, or token that is neither clearly English nor clearly
Romanized Hindi.

IMPORTANT EXAMPLES:

aapka -> HI
aapki -> HI
aap -> HI
hai -> HI
hain -> HI
nahi -> HI
nahin -> HI
mein -> HI
main -> HI
kya -> HI
kaise -> HI
karna -> HI
karo -> HI
milega -> HI
chahiye -> HI

account -> EN
verify -> EN
verification -> EN
please -> EN
bank -> EN
payment -> EN
money -> EN
your -> EN
click -> EN
call -> EN
receive -> EN

otp -> OTHER
upi -> OTHER
kyc -> OTHER
pan -> OTHER
aadhaar -> OTHER
amazon -> OTHER
paytm -> OTHER

IMPORTANT RULES:

1. Return ONLY JSON.
2. Do NOT use markdown.
3. Do NOT put ```json around the response.
4. Every input token MUST appear as a JSON key.
5. Every value MUST be exactly:
   EN, HI, AMB, or OTHER
6. Do NOT return noun, verb, pronoun, adjective, etc.
7. Do NOT explain your decisions.
8. Preserve the exact spelling of each token.
9. Do not omit any token.
10. Do not add tokens that were not provided.

TOKENS TO CLASSIFY:

{token_text}
"""

    return prompt


# ============================================================
# CLEAN MODEL RESPONSE
# ============================================================

def clean_json_response(content):

    content = content.strip()

    # Remove markdown fences if model ignores instruction
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"^```\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    content = content.strip()

    # Sometimes models put extra text before/after JSON.
    # Try to isolate the outermost JSON object.
    first_brace = content.find("{")
    last_brace = content.rfind("}")

    if first_brace != -1 and last_brace != -1:
        content = content[first_brace:last_brace + 1]

    return content


# ============================================================
# VALIDATE CLASSIFICATION
# ============================================================

def validate_result(result, tokens):

    if not isinstance(result, dict):

        raise ValueError(
            "Model response is not a JSON object."
        )

    requested = set(tokens)
    returned = set(result.keys())

    missing = requested - returned
    extra = returned - requested

    if missing:

        raise ValueError(
            f"Missing {len(missing)} tokens. "
            f"Examples: {list(missing)[:10]}"
        )

    if extra:

        raise ValueError(
            f"Model returned {len(extra)} unexpected tokens. "
            f"Examples: {list(extra)[:10]}"
        )

    for token in tokens:

        label = result[token]

        if label not in ALLOWED_LABELS:

            raise ValueError(
                f"Invalid label: "
                f"{token!r} -> {label!r}"
            )

    return True


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(tokens):

    prompt = build_prompt(tokens)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    content = response["message"]["content"]

    cleaned = clean_json_response(content)

    try:

        result = json.loads(cleaned)

    except json.JSONDecodeError as e:

        print()
        print("MODEL RETURNED INVALID JSON")
        print("-" * 70)
        print(content[:3000])
        print("-" * 70)

        raise ValueError(
            f"Invalid JSON returned by model: {e}"
        )

    validate_result(
        result,
        tokens
    )

    return result


# ============================================================
# CLASSIFY VOCABULARY
# ============================================================

def classify_vocabulary(tokens):

    print()
    print("=" * 70)
    print("LLM VOCABULARY CLASSIFICATION")
    print("=" * 70)

    token_map = load_checkpoint()

    remaining = [
        token
        for token in tokens
        if token not in token_map
    ]

    print(f"Unique tokens:           {len(tokens):,}")
    print(f"Batch size:              {BATCH_SIZE:,}")

    total_batches = (
        len(remaining) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"LLM batches required:     "
        f"{total_batches:,}"
    )

    print(
        f"Remaining tokens:        "
        f"{len(remaining):,}"
    )

    if not remaining:

        print()
        print("All tokens already classified.")
        return token_map

    for batch_index in range(
        0,
        len(remaining),
        BATCH_SIZE
    ):

        batch = remaining[
            batch_index:
            batch_index + BATCH_SIZE
        ]

        batch_number = (
            batch_index // BATCH_SIZE
        ) + 1

        print()
        print("-" * 70)
        print(
            f"Batch {batch_number}/{total_batches}"
        )

        print(
            f"Progress: "
            f"{len(token_map):,}/{len(tokens):,}"
        )

        success = False

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                print(
                    f"  Ollama batch: "
                    f"{len(batch):,} tokens "
                    f"(attempt {attempt})"
                )

                result = call_ollama(batch)

                token_map.update(result)

                save_checkpoint(token_map)

                print(
                    f"  SUCCESS: "
                    f"{len(result):,} tokens classified"
                )

                print(
                    f"  Total classified: "
                    f"{len(token_map):,}/"
                    f"{len(tokens):,}"
                )

                success = True

                break

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

                if attempt < MAX_RETRIES:

                    print(
                        f"  Retrying in "
                        f"{RETRY_DELAY} seconds..."
                    )

                    time.sleep(
                        RETRY_DELAY
                    )

                else:

                    print()
                    print(
                        "Batch failed after "
                        f"{MAX_RETRIES} attempts."
                    )

        if not success:

            print()
            print(
                "Stopping so the checkpoint "
                "is preserved."
            )

            raise RuntimeError(
                f"Failed batch {batch_number}"
            )

    print()
    print(
        f"Classification complete: "
        f"{len(token_map):,} tokens"
    )

    return token_map


# ============================================================
# SAVE TOKEN LANGUAGE MAP
# ============================================================

def save_token_language_map(
    token_counter,
    token_map
):

    # JSON
    with open(
        TOKEN_MAP_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            token_map,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved: {TOKEN_MAP_JSON}"
    )

    # CSV
    rows = []

    for token, count in token_counter.most_common():

        rows.append(
            {
                "token": token,
                "count": count,
                "label": token_map.get(
                    token,
                    "UNCLASSIFIED"
                ),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        TOKEN_MAP_CSV,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {TOKEN_MAP_CSV}"
    )


# ============================================================
# OCCURRENCE-LEVEL LANGUAGE STATISTICS
# ============================================================

def calculate_occurrence_statistics(
    token_counter,
    token_map
):

    counts = Counter()

    total = 0

    for token, frequency in token_counter.items():

        label = token_map.get(
            token,
            "OTHER"
        )

        counts[label] += frequency
        total += frequency

    stats = {}

    for label in [
        "EN",
        "HI",
        "AMB",
        "OTHER"
    ]:

        count = counts[label]

        percentage = (
            count / total * 100
            if total
            else 0
        )

        stats[label] = {
            "token_occurrences": count,
            "percentage": percentage
        }

    stats["TOTAL"] = {
        "token_occurrences": total,
        "percentage": 100.0
    }

    return stats


# ============================================================
# TYPE-LEVEL LANGUAGE STATISTICS
# ============================================================

def calculate_type_statistics(
    token_counter,
    token_map
):

    counts = Counter()

    total = len(token_counter)

    for token in token_counter:

        label = token_map.get(
            token,
            "OTHER"
        )

        counts[label] += 1

    stats = {}

    for label in [
        "EN",
        "HI",
        "AMB",
        "OTHER"
    ]:

        count = counts[label]

        percentage = (
            count / total * 100
            if total
            else 0
        )

        stats[label] = {
            "unique_token_types": count,
            "percentage": percentage
        }

    stats["TOTAL"] = {
        "unique_token_types": total,
        "percentage": 100.0
    }

    return stats


# ============================================================
# MESSAGE-LEVEL LANGUAGE ANALYSIS
# ============================================================

def classify_message(
    tokens,
    token_map
):

    if not tokens:

        return {
            "en_pct": 0,
            "hi_pct": 0,
            "amb_pct": 0,
            "other_pct": 0,
            "language": "EMPTY"
        }

    counts = Counter()

    for token in tokens:

        label = token_map.get(
            token,
            "OTHER"
        )

        counts[label] += 1

    total = len(tokens)

    en_pct = counts["EN"] / total * 100
    hi_pct = counts["HI"] / total * 100
    amb_pct = counts["AMB"] / total * 100
    other_pct = counts["OTHER"] / total * 100

    # Message-level classification
    #
    # We intentionally don't call every non-English
    # message "Hinglish".
    #
    # A mixed message requires actual English + Hindi.
    if en_pct >= 80:

        language = "ENGLISH_DOMINANT"

    elif hi_pct >= 80:

        language = "HINDI_DOMINANT"

    elif en_pct > 0 and hi_pct > 0:

        language = "HINGLISH_MIXED"

    elif amb_pct >= 50:

        language = "AMBIGUOUS"

    else:

        language = "OTHER_DOMINANT"

    return {
        "en_pct": en_pct,
        "hi_pct": hi_pct,
        "amb_pct": amb_pct,
        "other_pct": other_pct,
        "language": language
    }


# ============================================================
# MESSAGE ANALYSIS
# ============================================================

def analyze_messages(
    messages,
    message_tokens,
    token_map
):

    print()
    print("=" * 70)
    print("MESSAGE-LEVEL LANGUAGE ANALYSIS")
    print("=" * 70)

    rows = []

    for msg, tokens in tqdm(
        zip(messages, message_tokens),
        total=len(messages),
        desc="Analyzing messages"
    ):

        stats = classify_message(
            tokens,
            token_map
        )

        rows.append(
            {
                "filepath": msg["filepath"],
                "class": msg["class"],
                "text": msg["text"],
                "token_count": len(tokens),

                "en_pct": stats["en_pct"],
                "hi_pct": stats["hi_pct"],
                "amb_pct": stats["amb_pct"],
                "other_pct": stats["other_pct"],

                "language": stats["language"]
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        MESSAGE_ANALYSIS_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {MESSAGE_ANALYSIS_FILE}"
    )

    return df


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

def analyze_by_class(
    message_df
):

    print()
    print("=" * 70)
    print("PER-SCAM-CLASS LANGUAGE ANALYSIS")
    print("=" * 70)

    rows = []

    for class_name, group in message_df.groupby(
        "class"
    ):

        total_messages = len(group)

        language_counts = (
            group["language"]
            .value_counts()
            .to_dict()
        )

        row = {
            "class": class_name,
            "messages": total_messages,

            "avg_en_pct": group["en_pct"].mean(),
            "avg_hi_pct": group["hi_pct"].mean(),
            "avg_amb_pct": group["amb_pct"].mean(),
            "avg_other_pct": group["other_pct"].mean(),

            "english_dominant": language_counts.get(
                "ENGLISH_DOMINANT",
                0
            ),

            "hindi_dominant": language_counts.get(
                "HINDI_DOMINANT",
                0
            ),

            "hinglish_mixed": language_counts.get(
                "HINGLISH_MIXED",
                0
            ),

            "ambiguous": language_counts.get(
                "AMBIGUOUS",
                0
            ),

            "other_dominant": language_counts.get(
                "OTHER_DOMINANT",
                0
            ),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add percentage columns
    for col in [
        "english_dominant",
        "hindi_dominant",
        "hinglish_mixed",
        "ambiguous",
        "other_dominant"
    ]:

        df[col + "_pct"] = (
            df[col] /
            df["messages"] *
            100
        )

    df = df.sort_values(
        "class"
    )

    df.to_csv(
        CLASS_ANALYSIS_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved: {CLASS_ANALYSIS_FILE}"
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    messages,
    token_counter,
    token_map,
    message_df
):

    occurrence_stats = (
        calculate_occurrence_statistics(
            token_counter,
            token_map
        )
    )

    type_stats = (
        calculate_type_statistics(
            token_counter,
            token_map
        )
    )

    message_language_counts = (
        message_df["language"]
        .value_counts()
        .to_dict()
    )

    total_messages = len(message_df)

    message_language_percentages = {}

    for label, count in (
        message_language_counts.items()
    ):

        message_language_percentages[label] = (
            count /
            total_messages *
            100
            if total_messages
            else 0
        )

    summary = {

        "dataset": {
            "messages": len(messages),
            "unique_tokens": len(token_counter),
            "total_token_occurrences":
                sum(token_counter.values())
        },

        "llm": {
            "model": MODEL,
            "batch_size": BATCH_SIZE
        },

        "occurrence_level": occurrence_stats,

        "type_level": type_stats,

        "message_level": {
            "counts":
                message_language_counts,
            "percentages":
                message_language_percentages
        }
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved: {SUMMARY_FILE}"
    )

    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(summary):

    print()
    print("=" * 70)
    print("FINAL LANGUAGE SUMMARY")
    print("=" * 70)

    dataset = summary["dataset"]

    print()
    print(
        f"Messages:               "
        f"{dataset['messages']:,}"
    )

    print(
        f"Total token occurrences:"
        f" {dataset['total_token_occurrences']:,}"
    )

    print(
        f"Unique tokens:          "
        f"{dataset['unique_tokens']:,}"
    )

    print()
    print("OCCURRENCE-LEVEL")
    print("-" * 70)

    occurrence = summary[
        "occurrence_level"
    ]

    for label in [
        "EN",
        "HI",
        "AMB",
        "OTHER"
    ]:

        item = occurrence[label]

        print(
            f"{label:<8}"
            f"{item['token_occurrences']:>12,}"
            f"   "
            f"{item['percentage']:>7.2f}%"
        )

    print()
    print("TYPE-LEVEL")
    print("-" * 70)

    type_stats = summary[
        "type_level"
    ]

    for label in [
        "EN",
        "HI",
        "AMB",
        "OTHER"
    ]:

        item = type_stats[label]

        print(
            f"{label:<8}"
            f"{item['unique_token_types']:>12,}"
            f"   "
            f"{item['percentage']:>7.2f}%"
        )

    print()
    print("MESSAGE-LEVEL")
    print("-" * 70)

    msg_counts = summary[
        "message_level"
    ]["counts"]

    msg_pcts = summary[
        "message_level"
    ]["percentages"]

    for label in [
        "ENGLISH_DOMINANT",
        "HINDI_DOMINANT",
        "HINGLISH_MIXED",
        "AMBIGUOUS",
        "OTHER_DOMINANT"
    ]:

        count = msg_counts.get(
            label,
            0
        )

        pct = msg_pcts.get(
            label,
            0
        )

        print(
            f"{label:<20}"
            f"{count:>8,}"
            f"   "
            f"{pct:>7.2f}%"
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print("HINGLISH LANGUAGE ANALYSIS")
    print("Ollama + Gemma 4")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model:   {MODEL}")
    print(
        f"Batch:   {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    messages = load_dataset()

    # --------------------------------------------------------
    # 2. Tokenize
    # --------------------------------------------------------

    token_counter, message_tokens = (
        build_token_counts(messages)
    )

    # --------------------------------------------------------
    # 3. Save token frequencies
    # --------------------------------------------------------

    save_token_counts(
        token_counter
    )

    # --------------------------------------------------------
    # 4. Vocabulary
    # --------------------------------------------------------

    vocabulary = list(
        token_counter.keys()
    )

    # --------------------------------------------------------
    # 5. LLM classification
    # --------------------------------------------------------

    token_map = classify_vocabulary(
        vocabulary
    )

    # --------------------------------------------------------
    # 6. Save token mapping
    # --------------------------------------------------------

    save_token_language_map(
        token_counter,
        token_map
    )

    # --------------------------------------------------------
    # 7. Message analysis
    # --------------------------------------------------------

    message_df = analyze_messages(
        messages,
        message_tokens,
        token_map
    )

    # --------------------------------------------------------
    # 8. Per-class analysis
    # --------------------------------------------------------

    analyze_by_class(
        message_df
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    summary = create_summary(
        messages,
        token_counter,
        token_map,
        message_df
    )

    # --------------------------------------------------------
    # 10. Print final results
    # --------------------------------------------------------

    print_summary(
        summary
    )

    elapsed = time.time() - start_time

    print()
    print(
        f"Total runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("Output files:")
    print(
        f"  - {TOKEN_COUNTS_FILE}"
    )
    print(
        f"  - {TOKEN_MAP_JSON}"
    )
    print(
        f"  - {TOKEN_MAP_CSV}"
    )
    print(
        f"  - {MESSAGE_ANALYSIS_FILE}"
    )
    print(
        f"  - {CLASS_ANALYSIS_FILE}"
    )
    print(
        f"  - {SUMMARY_FILE}"
    )

    print()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
