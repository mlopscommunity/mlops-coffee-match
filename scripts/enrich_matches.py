"""Enrich match results with participant details and render Markdown.

This module reads a matches CSV (output of `scripts/run_matching.py`) and the
participants CSV used for matching, joins human-readable fields for each side of
the pair, and produces:

- An enriched CSV with flattened A/B participant details
- A Markdown report suitable for quick human review

Note: No LLM calls here; enrichment is purely local to preserve privacy.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any
import pandas as pd
import argparse
import json
from datetime import datetime


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the CSV contents.
    """
    # read the CSV into a DataFrame
    return pd.read_csv(path)


def normalize_participant_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a `participant_id` column as string for joins.

    Tries common identifier columns in order: `participant_id`, `synthetic_id`,
    `respondent_id`, `id`. If none exist, raises a ValueError.

    Args:
        df: Participants DataFrame.

    Returns:
        Copy of the DataFrame with a normalized `participant_id` column.
    """
    # detect/derive participant_id
    out = df.copy()
    if "participant_id" in out.columns:
        out["participant_id"] = out["participant_id"].astype(str)
        return out
    for col in ["synthetic_id", "respondent_id", "id"]:
        if col in out.columns:
            out["participant_id"] = out[col].astype(str)
            return out
    raise ValueError("No suitable identifier column found to create `participant_id`.")


def select_readable_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Project participant DataFrame to a minimal set of readable columns.

    The projection aims to capture fields helpful for human review while staying
    compact. Columns are renamed to a stable schema to simplify downstream joins.

    Preferred fields (if present):
        - participant_id (required)
        - name or synthetic_name
        - company, role, location or regional_location
        - career_stage
        - summary, buddy_preferences

    Args:
        df: Normalized participant DataFrame (must include `participant_id`).

    Returns:
        A projected DataFrame with standardized column names.
    """
    # choose best-available names for fields
    out = df.copy()
    out_cols = {"participant_id": "participant_id"}
    if "name" in out.columns:
        out_cols["name"] = "name"
    elif "synthetic_name" in out.columns:
        out_cols["synthetic_name"] = "name"
    for src, dst in [
        ("company", "company"),
        ("role", "role"),
        ("regional_location", "location"),
        ("location", "location"),
        ("career_stage", "career_stage"),
        ("summary", "summary"),
        ("buddy_preferences", "buddy_preferences"),
    ]:
        if src in out.columns and dst not in out_cols.values():
            out_cols[src] = dst
    return out[list(out_cols.keys())].rename(columns=out_cols)


def enrich_matches(matches_df: pd.DataFrame, people_df: pd.DataFrame) -> pd.DataFrame:
    """Join match rows with A/B participant details for enrichment.

    Args:
        matches_df: DataFrame from `run_matching.py` (A_id, B_id, score, etc.).
        people_df: Projected participants DataFrame from `select_readable_fields`.

    Returns:
        Enriched DataFrame with flattened A_* and B_* columns plus match metadata.
    """
    required = {
        "participant_A_id",
        "participant_B_id",
        "match_score",
        "llm_justification",
        "icebreaker_topics",
    }
    missing = required - set(matches_df.columns)
    if missing:
        raise KeyError(f"Matches CSV missing required columns: {sorted(missing)}")

    base = matches_df.copy()

    # Normalize icebreakers to JSON string for CSV stability
    def to_json_list(val: Any) -> str:
        if isinstance(val, list):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("[") and s.endswith("]"):
                # looks like JSON already
                return s
            # fallback: split by '|' or ',' heuristically
            parts = [p.strip() for p in (s.split("|") if "|" in s else s.split(",")) if p.strip()]
            return json.dumps(parts, ensure_ascii=False)
        return json.dumps([], ensure_ascii=False)

    base["icebreaker_topics"] = base["icebreaker_topics"].apply(to_json_list)

    # Prepare right-side DataFrame for A/B joins
    right = people_df.copy()

    # Join A side
    a_join = base.merge(
        right.add_prefix("A_"),
        how="left",
        left_on="participant_A_id",
        right_on="A_participant_id",
    )
    if "A_participant_id" in a_join.columns:
        a_join = a_join.drop(columns=["A_participant_id"])  # redundant after join

    # Join B side
    ab_join = a_join.merge(
        right.add_prefix("B_"),
        how="left",
        left_on="participant_B_id",
        right_on="B_participant_id",
    )
    if "B_participant_id" in ab_join.columns:
        ab_join = ab_join.drop(columns=["B_participant_id"])  # redundant after join

    # Optional: report unmatched ids
    # (leave blanks; do not fail hard)

    # Assemble final columns in a readable order
    final_cols: List[str] = [
        "participant_A_id",
        "participant_B_id",
        "match_score",
        "llm_justification",
        "icebreaker_topics",
        "intro_message",
        # Interleaved A/B fields for side-by-side comparison
        "A_name", "B_name",
        "A_role", "B_role",
        "A_company", "B_company",
        "A_location", "B_location",
        "A_career_stage", "B_career_stage",
        "A_summary", "B_summary",
        "A_buddy_preferences", "B_buddy_preferences",
    ]

    # Ensure all final columns exist (create empty ones if missing)
    for c in final_cols:
        if c not in ab_join.columns:
            ab_join[c] = None

    return ab_join[final_cols]


def render_markdown(enriched_df: pd.DataFrame, out_path_md: Path) -> None:
    """Render a human-readable Markdown report from the enriched matches.

    The report includes: score, A/B identity lines, concise summaries, LLM
    justification, icebreakers, and intro message (if present).

    Args:
        enriched_df: Output of `enrich_matches`.
        out_path_md: Destination file path for the Markdown report.
    """
    # open file and write a simple header and per-pair sections
    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(enriched_df)
    lines.append(f"# Matches Report\n")
    lines.append(f"Generated: {ts}\n")
    lines.append(f"Total pairs: {total}\n\n")

    def parse_icebreakers(s: Any) -> List[str]:
        if isinstance(s, list):
            return [str(x) for x in s]
        if isinstance(s, str):
            st = s.strip()
            if st.startswith("[") and st.endswith("]"):
                try:
                    arr = json.loads(st)
                    if isinstance(arr, list):
                        return [str(x) for x in arr]
                except Exception:
                    pass
            # fallback: comma split
            return [p.strip() for p in st.split(",") if p.strip()]
        return []

    df = enriched_df.copy()
    if "match_score" in df.columns:
        df = df.sort_values("match_score", ascending=False)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        score = row.get("match_score")
        score_str = f"{float(score):.2f}" if pd.notna(score) else "-"
        lines.append(f"## Pair {i} — Score: {score_str}\n")

        a_line = " | ".join(
            [
                str(row.get("A_name") or "Unknown"),
                str(row.get("A_role") or ""),
                f"@ {row.get('A_company')}" if row.get("A_company") else "",
                str(row.get("A_location") or ""),
                str(row.get("A_career_stage") or ""),
            ]
        ).strip(" | ")
        b_line = " | ".join(
            [
                str(row.get("B_name") or "Unknown"),
                str(row.get("B_role") or ""),
                f"@ {row.get('B_company')}" if row.get("B_company") else "",
                str(row.get("B_location") or ""),
                str(row.get("B_career_stage") or ""),
            ]
        ).strip(" | ")
        lines.append(f"Seeker (A): {a_line}\n")
        lines.append(f"Buddy  (B): {b_line}\n")

        def trunc(v: Any, n: int = 400) -> str:
            s = str(v or "").strip()
            return (s[: n - 1] + "…") if len(s) > n else s

        lines.append("")
        lines.append(f"A summary: {trunc(row.get('A_summary'))}\n")
        lines.append(f"A preferences: {trunc(row.get('A_buddy_preferences'))}\n")

        just = trunc(row.get("llm_justification"), 400)
        lines.append("")
        lines.append(f"Justification: {just}\n")

        ice = parse_icebreakers(row.get("icebreaker_topics"))
        if ice:
            lines.append("Icebreakers:")
            for it in ice:
                lines.append(f"- {it}")
            lines.append("")

        intro = trunc(row.get("intro_message"), 600)
        if intro:
            lines.append("> " + intro.replace("\n", "\n> "))
            lines.append("")

    out_path_md.parent.mkdir(parents=True, exist_ok=True)
    out_path_md.write_text("\n".join(lines), encoding="utf-8")
    return None


def main() -> None:
    """CLI entrypoint: read CSVs, enrich matches, write CSV and Markdown.

    Expected environment:
        - matches CSV path
        - participants CSV path
        - output directory

    Steps:
        1) Load matches and participants
        2) Normalize participants id and project readable fields
        3) Enrich and write CSV
        4) Render Markdown report
    """
    # parse args (paths), coordinate calls to helpers, print progress
    parser = argparse.ArgumentParser(description="Enrich matches with participant details and render report")
    parser.add_argument("--matches", type=Path, required=True, help="Path to matches CSV (from run_matching.py)")
    parser.add_argument("--participants", type=Path, required=True, help="Path to participants CSV used for matching")
    parser.add_argument("--out-dir", type=Path, default=Path("data_examples"), help="Output directory for enriched files")
    args = parser.parse_args()

    print(f"[1/4] Loading matches: {args.matches}")
    matches_df = load_csv(args.matches)

    print(f"[2/4] Loading participants: {args.participants}")
    people_raw = load_csv(args.participants)
    people_norm = normalize_participant_ids(people_raw)
    people_proj = select_readable_fields(people_norm)

    print("[3/4] Enriching matches…")
    enriched = enrich_matches(matches_df, people_proj)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "matches_enriched.csv"
    print(f"Writing enriched CSV: {out_csv}")
    enriched.to_csv(out_csv, index=False)

    out_md = args.out_dir / "matches_report.md"
    print(f"[4/4] Rendering Markdown report: {out_md}")
    render_markdown(enriched, out_md)
    print("Done.")
    return None


if __name__ == "__main__":
    main()


