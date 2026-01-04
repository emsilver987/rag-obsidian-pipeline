import json
import requests
import sys

# ---------------------------
# CONFIG
# ---------------------------
META_PATH = "metadata.json"
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b-instruct"

VALID_SPLITS = {"Push", "Pull", "Legs", "Mixed"}

# ---------------------------
# LLM CALL
# ---------------------------
def classify_split(workout_text):
    prompt = f"""
You are classifying a workout into one of four categories.

Rules:
- Push = chest, shoulders, triceps dominant
- Pull = back, biceps, rear delts dominant
- Legs = quads, hamstrings, glutes, calves dominant
- Mixed = no clear dominance

Return ONLY one word from:
Push, Pull, Legs, Mixed

Workout log:
{workout_text}
"""

    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return res.json().get("response", "").strip()

# ---------------------------
# MAIN
# ---------------------------
def main(write=False):
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    updated = 0
    skipped = 0

    for entry in metadata:
        # ---- Filters ----
        if entry.get("type") != "workouts":
            continue
        if not entry.get("date"):
            continue
        if "split" in entry:
            skipped += 1
            continue

        print("this is a valid workout")

        workout_text = entry.get("text", "").strip()
        if not workout_text:
            skipped += 1
            continue

        print(f"\nClassifying: {entry.get('file')} ({entry.get('date')})")

        split = classify_split(workout_text)

        if split not in VALID_SPLITS:
            print(f"⚠️ Invalid split returned: '{split}' — skipping")
            skipped += 1
            continue

        print(f"→ Classified as: {split}")

        if write:
            entry["split"] = split
            updated += 1

    if write:
        with open(META_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✅ Updated {updated} entries.")
    else:
        print(f"\nℹ️ Dry run complete. {updated} would be updated, {skipped} skipped.")
        print("Run with --write to persist changes.")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    write_flag = "--write" in sys.argv
    main(write=write_flag)

