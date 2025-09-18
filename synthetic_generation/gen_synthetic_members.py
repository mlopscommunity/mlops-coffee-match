#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
"""Utilities to sample participant records for synthetic data generation.

This module loads the private survey CSV into a pandas DataFrame and samples
rows without replacement to feed into an LLM for synthetic record creation.

CLI usage example:
  python synthetic_generation/gen_synthetic_members.py \
    --csv data/private_mlops_marchvc.csv \
    --n 5 \
    --seed 42 \
    --jsonl-out samples.jsonl

Compare sampled vs. generated locally:
  python synthetic_generation/gen_synthetic_members.py \
    --csv data/private_mlops_marchvc.csv \
    --n 5 \
    --seed 42 \
    --compare

Notes:
- Dependencies are managed with uv (use: `uv add pandas python-dotenv`).
- Environment variables can be configured via a .env file.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from dotenv import load_dotenv
from synthetic_generation.synth_models import SyntheticParticipants
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def load_participants(csv_path: Path) -> pd.DataFrame:
    """Load participants CSV into a DataFrame.

    Args:
        csv_path: Path to the CSV file containing private survey responses.

    Returns:
        A pandas DataFrame with the survey data.

    Raises:
        FileNotFoundError: If the csv_path does not exist.
        ValueError: If the file exists but is empty or cannot be parsed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to read CSV at {csv_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV at {csv_path} contains no rows")

    return df


# Canonical column rename mapping for CSV headers → pythonic names
RENAME_MAP: Dict[str, str] = {
    "Unnamed: 0": "submission_id",
    "Respondent ID": "respondent_id",
    "Submitted at": "submitted_at",
    "Your email address": "email",
    "Your name ": "name",
    "Is your Slack handle different from your name? \nE.g. Name :Dale Bastianz  Slack: Bastian ": "slack_name_different",
    "What is your Slack Handle?": "slack_handle",
    "Your LinkedIn URL": "linkedin_url",
    "Where do you work? ": "company",
    "Which option best represents your role?": "role",
    "How would you characterize your current career stage?": "career_stage",
    "Where are you based?": "location",
    "Do you have a preference on who your buddy should be?": "buddy_preference",
    "Tweet-sized summary of yourself": "summary",
    "What are your skills? In which fields do you specialize?": "skills",
    "Describe what you want your buddy to be like.": "buddy_preferences",
    "Keep me registered for all future rounds.": "keep_registered",
    "Keep me registered for all future rounds. (Yes)": "keep_registered_yes",
    "Keep me registered for all future rounds. (Ask me again next time)": "keep_registered_ask_next_time",
}

# PII columns in canonical and original forms for safety (not sent to LLM)
PII_COLUMNS_CANONICAL = {"email", "slack_handle", "linkedin_url", "company", "name"}
PII_COLUMNS_ORIGINAL = {
    "Your email address",
    "What is your Slack Handle?",
    "Your LinkedIn URL",
    "Where do you work? ",
    "Your name ",
}

# Fields we allow as LLM input examples
ALLOWED_LLM_FIELDS = {
    "role",
    "career_stage",
    "location",
    "buddy_preference",
    "summary",
    "skills",
    "buddy_preferences",
    "submission_id",
    "respondent_id",
}


def apply_column_renames(df: pd.DataFrame, apply: bool) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP) if apply else df


def select_allowed_fields(records: List[Dict[str, Any]], allowed: set[str]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for rec in records:
        selected.append({k: v for k, v in rec.items() if k in allowed})
    return selected


def sample_participants(
    df: pd.DataFrame, num_samples: int, seed: int | None = None
) -> pd.DataFrame:
    """Sample participants without replacement.

    Args:
        df: Source DataFrame to sample from.
        num_samples: Number of rows to sample (without replacement).
        seed: Optional random seed for reproducibility.

    Returns:
        A DataFrame containing the sampled rows.

    Raises:
        ValueError: If num_samples is not in [1, len(df)].
    """
    total_rows = len(df)
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if num_samples > total_rows:
        raise ValueError(
            f"num_samples ({num_samples}) cannot exceed number of rows ({total_rows})"
        )

    return df.sample(n=num_samples, replace=False, random_state=seed)


def records_for_llm(df: pd.DataFrame) -> List[dict[str, Any]]:
    """Convert a DataFrame to JSON-serializable records for LLM prompting.

    Converts NaN values to None to ensure valid JSON.

    Args:
        df: Input DataFrame.

    Returns:
        List of record dictionaries ready to be serialized as JSON.
    """
    # Ensure JSON serializable values (convert NaN to None)
    cleaned = df.where(pd.notnull(df), None)
    return cast(List[Dict[str, Any]], cleaned.to_dict(orient="records"))


def generate_with_openai_min(
    sampled_records: List[Dict[str, Any]],
    model: str,
    num_generated: int,
    structured_output: bool = True,
) -> List[Dict[str, Any]]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Use 'uv add openai'.")
    client = OpenAI()
    msgs = [
        {"role": "system", "content": "Generate anonymized synthetic participants as a JSON array only."},
        {
            "role": "user",
            "content": (
                "Create {num} records similar to these examples, with new fake names and lightly paraphrased fields.\n"
                "Do not include emails, Slack handles, LinkedIn, company names.\n"
                "Respond with ONLY a JSON array of objects.\n\n"
                f"{json.dumps({"examples": sampled_records}, ensure_ascii=False)}"
            ).replace("{num}", str(int(num_generated))),
        },
    ]
    if structured_output:
        parsed = client.responses.parse(  # type: ignore[attr-defined]
            model=model,
            input=cast(Any, msgs),
            text_format=SyntheticParticipants,
        )
        items = getattr(parsed, "output_parsed", None)
        if items is None:
            return []
        seq = getattr(items, "root", items)
        out: List[Dict[str, Any]] = []
        for p in seq:
            if hasattr(p, "model_dump"):
                out.append(p.model_dump())  # type: ignore[no-any-return]
            elif isinstance(p, dict):
                out.append(cast(Dict[str, Any], p))
        return out
    resp = client.chat.completions.create(model=model, messages=msgs)  # type: ignore[arg-type]
    content = resp.choices[0].message.content or "[]"
    data = json.loads(content)
    if isinstance(data, list):
        return cast(List[Dict[str, Any]], data)
    return []


def write_jsonl(records: List[dict[str, Any]], out_path: Path) -> None:
    """Write records to a JSONL file (one JSON object per line).

    Args:
        records: List of dictionaries to serialize.
        out_path: Destination file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample participant records from a CSV to seed LLM-based synthetic generation."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/private_mlops_marchvc.csv",
        help="Path to the private participants CSV (default: data/private_mlops_marchvc.csv)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of rows to sample without replacement (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    parser.add_argument(
        "--jsonl-out",
        type=str,
        default=None,
        help=(
            "Optional path to write JSONL of sampled records. If omitted, prints JSON to stdout."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print both sampled and locally generated synthetic records in one JSON object",
    )
    parser.add_argument(
        "--generated-n",
        type=int,
        default=None,
        help="Number of synthetic records to generate for comparison (defaults to n)",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API to generate synthetic records instead of local placeholder",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use when --use-openai is set (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--structured-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OpenAI structured output (JSON Schema) for generation (default: true)",
    )
    parser.add_argument(
        "--apply-rename",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply canonical column rename mapping after reading CSV (default: true)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write generated records as a CSV (e.g., data/synthetic_generated.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    # Load environment variables from .env if present
    load_dotenv()

    args = parse_args()
    csv_path = Path(args.csv)
    df = load_participants(csv_path)
    if args.apply_rename:
        df = apply_column_renames(df, apply=True)

    sampled_df = sample_participants(df, num_samples=args.n, seed=args.seed)
    records = records_for_llm(sampled_df)
    # Strip PII and non-allowed fields before LLM examples
    records_for_examples = select_allowed_fields(records, ALLOWED_LLM_FIELDS)

    if args.compare:
        gen_n = args.generated_n if args.generated_n is not None else len(records)
        if args.use_openai:
            if OpenAI is None:
                raise RuntimeError(
                    "openai package not available. Install with: uv add openai"
                )
            allowed_models = {"gpt-5", "gpt-5-mini"}
            if args.openai_model not in allowed_models:
                raise ValueError(
                    f"Unsupported model '{args.openai_model}'. Allowed: {sorted(allowed_models)}"
                )
            generated = generate_with_openai_min(
                sampled_records=records_for_examples,
                model=args.openai_model,
                num_generated=gen_n,
                structured_output=args.structured_output,
            )
        else:
            generated = records
        if args.out_csv:
            try:
                import pandas as _pd
                _df = _pd.DataFrame(generated)
                Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
                _df.to_csv(args.out_csv, index=False)
            except Exception as _exc:  # noqa: BLE001
                print(f"Failed to write CSV to {args.out_csv}: {_exc}")
        combined = {"sampled": records, "generated_count": len(generated)}
        if not args.out_csv:
            combined["generated"] = generated
        print(json.dumps(combined, ensure_ascii=False, indent=2))
        return

    if args.jsonl_out:
        write_jsonl(records, Path(args.jsonl_out))
        return

    print(json.dumps(records, ensure_ascii=False, indent=2))


def generate_with_openai(
    sampled_records: List[Dict[str, Any]],
    model: str,
    num_generated: int,
    seed: Optional[int] = None,
    structured_output: bool = True,
) -> List[Dict[str, Any]]:
    # Backwards-compatible alias to the minimal generator; seed is unused here
    return generate_with_openai_min(
        sampled_records=sampled_records,
        model=model,
        num_generated=num_generated,
        structured_output=structured_output,
    )


if __name__ == "__main__":
    main()


