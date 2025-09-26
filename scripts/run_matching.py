"""Run the matching pipeline on a specified CSV file.

Pseudocode:
1) Configure input CSV path (edit INPUT_CSV as needed)
2) Load participants via matching.matcher.read_participants
3) Run matching.matcher.match_participants to produce matches DataFrame
4) Save results to OUTPUT_CSV and print a brief summary

Notes:
- This script expects embeddings columns to exist:
  'buddy_preferences_embedding' and 'personal_summary_embedding'.
  If missing, run your feature engineering/embedding generation first.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import pandas as pd

# Ensure project root (parent of scripts/) is on sys.path for package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching.matcher import read_participants, match_participants
from matching.feature_engineering import prepare_feature_columns
from matching.ingest import resolve_aliases
from dotenv import load_dotenv


# Edit this path to point at the CSV you want to match on
INPUT_CSV = Path("data/private_mlops_marchvc.csv")
OUTPUT_CSV = Path("data/private_mlops_marchvc_matches.csv")


def main() -> None:
    """Entry point to run the matching pipeline on INPUT_CSV.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        KeyError: If required columns are missing for matching.
    """
    load_dotenv()
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # 1) Load participants
    print(f"[1/5] Loading participants from {INPUT_CSV}...")
    participants_df: pd.DataFrame = read_participants(str(INPUT_CSV))
    print(f"       Loaded {len(participants_df)} rows.")

    # 2) Feature engineering to produce embeddings and tiers
    try:
        model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        print(f"[2/5] Running feature engineering (model={model})...")
        fe_df = prepare_feature_columns(participants_df, model=model, batch_size=20)
        print("       Feature engineering complete.")
    except Exception as e:
        print(f"Warning: feature engineering failed ({e}). Proceeding with original DataFrame.")
        fe_df = participants_df

    # 2b) Normalize columns expected by matcher
    print("[3/5] Normalizing columns for matcher expectations...")
    df = fe_df.copy()
    alias_map = resolve_aliases(df)
    # Embedding columns mapping
    if "personal_summary_embedding" not in df.columns and "profile_embedding" in df.columns:
        df = df.rename(columns={"profile_embedding": "personal_summary_embedding"})
    if "buddy_preferences_embedding" not in df.columns and "preference_embedding" in df.columns:
        df = df.rename(columns={"preference_embedding": "buddy_preferences_embedding"})
    # Participant id normalization (use aliases first, then fallbacks)
    if "participant_id" not in df.columns:
        id_alias = alias_map.get("id")
        if id_alias is not None and id_alias in df.columns:
            df["participant_id"] = df[id_alias].astype(str)
        else:
            id_candidates = [
                "participant_id",
                "synthetic_id",
                "respondent_id",
                "Respondent ID",
                "source_participant_id",
                "Participant ID",
                "id",
            ]
            for col in id_candidates:
                if col in df.columns:
                    df["participant_id"] = df[col].astype(str)
                    break

    # buddy_preference normalization for ordering (use alias mapping)
    if "buddy_preference" not in df.columns:
        bp_alias = alias_map.get("buddy_preference")
        if bp_alias is not None and bp_alias in df.columns:
            df["buddy_preference"] = df[bp_alias]
    # Ensure required numeric tiers exist (if missing, fill with defaults)
    if "career_stage_level" not in df.columns:
        df["career_stage_level"] = 3  # neutral mid-tier default
    if "region_tier" not in df.columns:
        df["region_tier"] = 3  # neutral mid-tier default

    # Coerce to numeric and fill missing with defaults to avoid NoneType math
    df["career_stage_level"] = (
        pd.to_numeric(df["career_stage_level"], errors="coerce").fillna(3).astype(int)
    )
    df["region_tier"] = (
        pd.to_numeric(df["region_tier"], errors="coerce").fillna(3).astype(int)
    )

    # 3) Run matching (handles ordering internally)
    try:
        print(f"[4/5] Matching participants (top_n=20) on {len(df)} rows...")
        def progress(pairs_done: int, total_pairs: int, a_id: str, b_id: str, score: float) -> None:
            # Print every 5 pairs and on the final pair
            if total_pairs <= 0:
                return
            if (pairs_done % 5 == 0) or (pairs_done == total_pairs):
                pct = int(100 * pairs_done / total_pairs)
                print(f"   - [{pairs_done}/{total_pairs} | {pct}%] last: {a_id} ↔ {b_id} (score={score:.3f})")
        matches_df: pd.DataFrame = match_participants(df, top_n=20, progress_fn=progress)
    except KeyError as e:
        missing_info = (
            "Missing required columns. Ensure embeddings exist: "
            "'buddy_preferences_embedding', 'personal_summary_embedding'"
        )
        print(missing_info)
        raise

    # 4) Save results
    print(f"[5/5] Saving results to {OUTPUT_CSV}...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(OUTPUT_CSV, index=False)

    # 5) Report summary
    pairs = len(matches_df)
    print(f"Done. Wrote {pairs} matches to {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

