
#!/usr/bin/env python3
"""
Scam Dataset Upscaler
─────────────────────
Reads existing scam conversation files from a source folder,
then uses an Ollama model to generate new synthetic conversations
until each class reaches the target count.

Usage:
    python upscale_scam_dataset.py \
        --source scam_semantic_9 \
        --dest   scam_semantic_12k \
        --model  gemma3:27b \
        --target 2000 \
        [--resumable]

    The --resumable flag (default ON) means:
        • Progress is tracked in <dest>/_progress.json
        • If the script crashes/stops, re-running it skips already-generated files
        • Pass --no-resumable to disable and regenerate everything

Format expected in source files (one conversation per file, .txt):
    Scammer: <message>
    Receiver: <message>
    Scammer: <message>
    ...

Format generated in output files: same as above.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import ollama
except ImportError:
    print("ERROR: 'ollama' package not found. Install it with:  pip install ollama")
    sys.exit(1)


# ─── Class configuration (from the screenshot) ────────────────────────────────
# Only classes that need upscaling are listed (class 8 is excluded since it's at target).
# ─── Class configuration (from the screenshot) ────────────────────────────────
# Keys = EXACT folder names in scam_semantic_9
CLASS_TARGETS = {
    "2_upi_wallet_fraud":                     1000,
    "4_digital_arrest_govt_impersonation":    1000,
    "5_loan_credit_app_scams":                1000,
    "6_delivery_customer_care_scams":         1000,
    "7_dating_romance_sextortion":            1000,
    # "8_legacy_telecom_scams": 3031  ← already at target, skip
}

CLASS_DESCRIPTIONS = {
    "1_banking_kyc_otp_fraud": (
        "a bank/KYC/OTP scam where the scammer impersonates a bank representative, "
        "asking the victim to share OTP, Aadhaar number, PAN card, or bank account "
        "details to 'verify' or 'update' their KYC. The scammer creates urgency by "
        "threatening account suspension."
    ),
    "2_upi_wallet_fraud": (
        "a UPI or digital wallet fraud where the scammer tricks the victim into "
        "sending money via UPI apps (PhonePe, GPay, Paytm, etc.) by posing as a "
        "buyer, or by sending fake 'collect' requests, or by claiming the victim "
        "won a cashback/prize that requires a small 'processing fee'."
    ),
    "3_investment_task_scam": (
        "an investment or task-based scam (also called part-time job scam) where "
        "the scammer promises high returns on cryptocurrency/stock investments or "
        "asks the victim to complete simple online tasks (like rating apps, liking "
        "videos) for payment, then disappears after collecting deposits or charges "
        "withdrawal fees."
    ),
    "4_digital_arrest_govt_impersonation": (
        "a 'digital arrest' or government impersonation scam where the scammer "
        "pretends to be from the CBI, ED, TRAI, police, or customs department, "
        "claiming the victim is involved in drug trafficking, money laundering, or "
        "illegal parcel delivery. They demand money to 'settle' the case and keep "
        "the victim on a video call to prevent them from seeking help."
    ),
    "5_loan_credit_app_scams": (
        "a loan app or extortion scam where the scammer represents a predatory "
        "instant-loan app, harasses the borrower with abusive calls, threatens to "
        "send morphed photos to contacts, or demands excessive repayment amounts far "
        "beyond the principal."
    ),
    "6_delivery_customer_care_scams": (
        "a delivery or customer care scam where the scammer poses as an Amazon, "
        "Flipkart, Swiggy, Zomato, or courier agent claiming there's an issue with "
        "a package. They ask for OTP, payment, or remote access to 'resolve' the issue."
    ),
    "7_dating_romance_sextortion": (
        "a dating app romance scam or sextortion where the scammer builds a "
        "romantic relationship online, then either asks for money (emergency, travel, "
        "visa), or lures the victim into sending intimate photos and later threatens "
        "to leak them unless paid."
    ),
}
# ─── Utility helpers ──────────────────────────────────────────────────────────

def load_existing_conversations(class_dir: Path) -> list[str]:
    """Return all conversation texts found in a class directory."""
    conversations = []
    if not class_dir.exists():
        return conversations
    for f in class_dir.glob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8").strip()
            if text:
                conversations.append(text)
        except Exception:
            pass
    return conversations


def sample_seed_conversations(
    all_conversations: list[str],
    already_used_indices: set[int],
    n: int = 3,
) -> tuple[list[str], list[int]]:
    """
    Randomly pick `n` conversations that haven't been used as seeds recently.
    Returns (selected_texts, selected_indices).
    If all conversations have been used, resets the pool.
    """
    available = [i for i in range(len(all_conversations)) if i not in already_used_indices]
    if len(available) < n:
        # Reset: allow re-use so generation can continue
        available = list(range(len(all_conversations)))
    
    chosen_indices = random.sample(available, min(n, len(available)))
    chosen_texts = [all_conversations[i] for i in chosen_indices]
    return chosen_texts, chosen_indices


def build_prompt(class_name: str, seed_conversations: list[str]) -> str:
    description = CLASS_DESCRIPTIONS.get(class_name, class_name.replace("_", " "))
    seeds_text = "\n\n---EXAMPLE---\n".join(seed_conversations)

    prompt = f"""You are generating synthetic training data for a scam detection AI.

The scam type is: {description}

Here are {len(seed_conversations)} example conversations from this category:

---EXAMPLE---
{seeds_text}
---END EXAMPLES---

Now generate ONE new realistic scam conversation of this type.

STRICT RULES:
1. The scammer ALWAYS speaks first.
2. Each turn is labeled exactly as:
   Scammer: <message>
   Receiver: <message>
3. The conversation MUST be exactly 5 to 6 turns total (5-6 Scammer lines + 5-6 Receiver lines).
4. STOP generating after the 6th Scammer turn. Do not write more than 6 exchanges.
5. Write ONLY the conversation — no title, no explanation, no metadata, no preamble.
6. Make it distinct from the examples above — vary the scenario details, names, amounts, platform names, and tactics used.
7. The conversation should feel authentic, using natural Indian English / Hinglish as appropriate.
8. Do NOT include any markdown formatting, asterisks, or bold text.

Output the conversation now:"""
    return prompt

def parse_conversation(raw: str) -> Optional[str]:
    """
    Validate and clean the raw LLM output.
    Returns cleaned conversation string or None if it looks invalid.
    """
    raw = raw.strip()
    # Must start with "Scammer:"
    if not re.match(r"(?i)^scammer\s*:", raw):
        # Try to find where it starts
        match = re.search(r"(?i)(scammer\s*:.*)", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        else:
            return None

    # Normalize label casing
    raw = re.sub(r"(?i)\bscammer\s*:", "Scammer:", raw)
    raw = re.sub(r"(?i)\breceiver\s*:", "Receiver:", raw)

    # Count turns
    scammer_turns = len(re.findall(r"^Scammer:", raw, re.MULTILINE))
    receiver_turns = len(re.findall(r"^Receiver:", raw, re.MULTILINE))

    if scammer_turns < 3 or receiver_turns < 2:
        return None  # Too short / malformed

    return raw


def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_progress(progress_file: Path, progress: dict) -> None:
    progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")


# ─── Core generation loop ─────────────────────────────────────────────────────
def upscale_class(
    class_name: str,
    source_class_dir: Path,
    dest_class_dir: Path,
    target_count: int,
    model: str,
    progress: dict,
    progress_file: Path,
    resumable: bool,
) -> None:
    import math
    dest_class_dir.mkdir(parents=True, exist_ok=True)

    # Load source (seed) conversations
    source_convs = load_existing_conversations(source_class_dir)
    if not source_convs:
        print(f"  [WARN] No source conversations found in {source_class_dir}. Skipping.")
        return

    current_source_count = len(source_convs)
    print(f"  Source conversations: {current_source_count}")

    # Count already-generated files in dest
    existing_dest = load_existing_conversations(dest_class_dir)
    already_generated = len(existing_dest)
    print(f"  Already generated in dest: {already_generated}")

    # ─── Round-robin batch logic ───────────────────────────────────────────────
   # ─── Round-robin batch logic ───────────────────────────────────────────────
    per_run_batch = 300
    final_target = target_count  # 2000

    # Hard stop if already at final target
    total_available = current_source_count + already_generated
    if total_available >= final_target:
        print(f"  ✓ Already at final target ({total_available}/{final_target}). Skipping.")
        return

    # Find the global minimum across all dest folders (so all classes stay in sync)
    global_min = min(
        len(list((dest_class_dir.parent / cn).glob("*.txt")))
        if (dest_class_dir.parent / cn).exists() else 0
        for cn in CLASS_TARGETS
    )

    # This class should only generate up to global_min + per_run_batch
    this_run_target = global_min + per_run_batch

    # Don't overshoot final target
    max_dest_allowed = final_target - current_source_count
    this_run_target = min(this_run_target, max_dest_allowed)

    needed = this_run_target - already_generated
    if needed <= 0:
        print(f"  ✓ This class is ahead ({already_generated} generated) — skipping until others catch up.")
        return

    print(f"  Dest so far : {already_generated}")
    print(f"  This run    : up to {this_run_target} in dest (+{needed} new)")
    print(f"  Final target: {final_target} total (src + dest)")
    # ──────────────────────────────────────────────────────────────────────────
    # Track which source conversations were recently used as seeds (for variance)
    class_progress_key = class_name
    used_seed_indices: set[int] = set(progress.get(class_progress_key, {}).get("used_seeds", []))
    generated_so_far: int = progress.get(class_progress_key, {}).get("generated", 0)

    # File counter: start after any existing dest files
    file_counter = already_generated + 1

    failures = 0
    max_consecutive_failures = 10

    for i in range(needed):
        # Sample seed conversations (avoid repeating recent ones)
        n_seeds = random.randint(2, 3)
        seed_texts, seed_indices = sample_seed_conversations(source_convs, used_seed_indices, n=n_seeds)
        used_seed_indices.update(seed_indices)

        # If we've used half the pool, allow reuse (reset tracking)
        if len(used_seed_indices) > len(source_convs) * 0.6:
            used_seed_indices = set()

        prompt = build_prompt(class_name, seed_texts)

        # Call Ollama
        attempt = 0
        conversation = None
        while attempt < 3 and conversation is None:
            try:
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={"num_predict": 400}
                )
                raw_text = response.get("response", "").strip()
                conversation = parse_conversation(raw_text)
                if conversation is None:
                    print(f"    [retry {attempt+1}] Output malformed, retrying...")
                    attempt += 1
                    time.sleep(1)
            except Exception as e:
                print(f"    [ERROR] Ollama call failed: {e}")
                attempt += 1
                time.sleep(3)

        if conversation is None:
            failures += 1
            print(f"    [SKIP] Could not generate valid conversation after 3 attempts.")
            if failures >= max_consecutive_failures:
                print(f"  [ABORT] Too many consecutive failures ({failures}). Stopping this class.")
                break
            continue

        failures = 0  # reset on success

        # Save to dest
        out_file = dest_class_dir / f"{class_name}_{file_counter:05d}.txt"
        out_file.write_text(conversation, encoding="utf-8")
        file_counter += 1
        generated_so_far += 1

        # Save progress checkpoint
        if resumable:
            progress[class_progress_key] = {
                "generated": generated_so_far,
                "used_seeds": list(used_seed_indices),
            }
            save_progress(progress_file, progress)

        if (i + 1) % 10 == 0 or (i + 1) == needed:
            print(f"  Progress: {i+1}/{needed} generated for {class_name}")

    print(f"  ✓ Done with {class_name}. Dest now has {file_counter - 1} generated files.")
    print(f"     Total available (src + dest): {current_source_count + file_counter - 1}/{final_target}")
    print(f"     Run again to continue next batch of {per_run_batch}.")# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upscale scam conversation dataset using an Ollama model."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to source dataset folder (e.g. scam_semantic_9)"
    )
    parser.add_argument(
        "--dest", required=True,
        help="Path to destination folder for generated data (e.g. scam_semantic_12k)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Ollama model name to use (e.g. gemma3:27b)"
    )
    parser.add_argument(
        "--target", type=int, default=2000,
        help="Target number of total conversations per class (default: 2000)"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Optionally limit to specific class names (space-separated). "
             "Default: all classes in CLASS_TARGETS."
    )
    parser.add_argument(
        "--resumable", action=argparse.BooleanOptionalAction, default=True,
        help="Track progress so interrupted runs can be resumed (default: enabled). "
             "Use --no-resumable to disable."
    )
    parser.add_argument(
        "--skip-at-target", action="store_true", default=True,
        help="Skip classes that already have >= target conversations (default: True)"
    )

    args = parser.parse_args()

    source_root = Path(args.source)
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    progress_file = dest_root / "_progress.json"
    progress = load_progress(progress_file) if args.resumable else {}

    # Determine which classes to process
    if args.classes:
        classes_to_process = {
            k: CLASS_TARGETS.get(k, args.target)
            for k in args.classes
        }
    else:
        classes_to_process = CLASS_TARGETS

    # Verify model is available
    print(f"Checking Ollama model '{args.model}'...")
    try:
        ollama.show(args.model)
        print(f"  ✓ Model '{args.model}' is available.\n")
    except Exception as e:
        print(f"  [ERROR] Model '{args.model}' not found or Ollama not running: {e}")
        print("  Make sure Ollama is running and the model is pulled:  ollama pull <model>")
        sys.exit(1)

    print(f"Source : {source_root.resolve()}")
    print(f"Dest   : {dest_root.resolve()}")
    print(f"Model  : {args.model}")
    print(f"Target : {args.target} conversations per class")
    print(f"Resumable: {args.resumable}")
    print("=" * 60)

    for class_name, target in classes_to_process.items():
        effective_target = args.target  # CLI target overrides table if provided
        print(f"\n[Class] {class_name}  (target: {effective_target})")

        source_class_dir = source_root / class_name
        dest_class_dir   = dest_root   / class_name

        if not source_class_dir.exists():
            # Try case-insensitive match
            matches = [d for d in source_root.iterdir() if d.name.lower() == class_name.lower()]
            if matches:
                source_class_dir = matches[0]
                print(f"  (matched source folder: {source_class_dir.name})")
            else:
                print(f"  [WARN] Source folder not found: {source_class_dir}. Skipping.")
                continue

        upscale_class(
            class_name=class_name,
            source_class_dir=source_class_dir,
            dest_class_dir=dest_class_dir,
            target_count=effective_target,
            model=args.model,
            progress=progress,
            progress_file=progress_file,
            resumable=args.resumable,
        )

    print("\n" + "=" * 60)
    print("All classes processed. Summary:")
    for class_name in classes_to_process:
        dest_class_dir = dest_root / class_name
        gen_count = len(list(dest_class_dir.glob("*.txt"))) if dest_class_dir.exists() else 0
        src_count = len(list((source_root / class_name).glob("*.txt"))) \
            if (source_root / class_name).exists() else 0
        total = src_count + gen_count
        status = "✓" if total >= args.target else "⚠ BELOW TARGET"
        print(f"  {class_name:<35} src={src_count}  new={gen_count}  total={total}  {status}")

    if args.resumable:
        print(f"\nProgress saved to: {progress_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()
