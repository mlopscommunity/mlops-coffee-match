#!/usr/bin/env python3
"""Sample participant records for synthetic data generation."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from dotenv import load_dotenv
try:
    from .synth_models import SyntheticParticipantsResponse
except ImportError:
    from synth_models import SyntheticParticipantsResponse
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore


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

# Fields we allow as LLM input examples (excludes PII)
ALLOWED_LLM_FIELDS = {
    "role", "career_stage", "location", "buddy_preference", 
    "summary", "skills", "buddy_preferences", "submission_id", "respondent_id"
}


def apply_column_renames(df: pd.DataFrame, apply: bool) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP) if apply else df


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


def prepare_llm_records(df: pd.DataFrame, allowed_fields: set[str]) -> List[Dict[str, Any]]:
    """Prepare DataFrame records for LLM prompting.
    
    Filters to allowed fields and converts NaN values to None for JSON serialization.

    Args:
        df: Input DataFrame.
        allowed_fields: Set of column names to include in output.

    Returns:
        List of record dictionaries ready for LLM prompting.
    """
    # Filter to allowed fields and convert NaN to None
    filtered_df = df[list(allowed_fields)].where(pd.notnull(df[list(allowed_fields)]), None)
    return cast(List[Dict[str, Any]], filtered_df.to_dict(orient="records"))


def generate_with_openai_min(
    sampled_records: List[Dict[str, Any]],
    model: str,
    num_generated: int,
    structured_output: bool = True,
) -> List[Dict[str, Any]]:
    """Generate synthetic participants using OpenAI API."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Use 'uv add openai'.")
    
    client = OpenAI()
    
    system_prompt = f"""You are an AI assistant that generates synthetic participant data for research purposes.

Your task:
- Create {num_generated} synthetic participant records based on the provided examples
- Use completely new fake names and lightly paraphrase other fields
- Exclude all PII: emails, Slack handles, LinkedIn URLs, company names, real names
- Maintain the same data structure and field types as the examples
- Ensure synthetic data is realistic but not identifiable
- Output as a JSON object with a "participants" array containing the records

Examples to base your generation on:
{json.dumps(sampled_records, ensure_ascii=False, indent=2)}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate {num_generated} synthetic participant records now."},
    ]
    
    if structured_output:
        # Use structured output with Pydantic model
        parsed = client.responses.parse(  # type: ignore[attr-defined]
            model=model,
            input=cast(Any, messages),
            text_format=SyntheticParticipantsResponse,
        )
        items = getattr(parsed, "output_parsed", None)
        if items is None:
            return []
        
        # Extract the participants list and convert to dicts
        participants = getattr(items, "participants", [])
        return [p.model_dump() if hasattr(p, "model_dump") else p for p in participants]
    
    # Regular completion fallback
    resp = client.chat.completions.create(model=model, messages=messages)  # type: ignore[arg-type]
    content = resp.choices[0].message.content or "[]"
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "participants" in data:
            return data["participants"]
        elif isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        return []


def generate_with_anthropic(
    sampled_records: List[Dict[str, Any]],
    model: str,
    num_generated: int,
) -> List[Dict[str, Any]]:
    """Generate synthetic participants using Anthropic API."""
    if anthropic is None:
        raise RuntimeError("anthropic package not installed. Use 'uv add anthropic'.")
    
    client = anthropic.Anthropic()
    
    system_prompt = f"""You are an AI assistant that generates synthetic participant data for research purposes.

Your task:
- Create {num_generated} synthetic participant records based on the provided examples
- Use completely new fake names and lightly paraphrase other fields
- Exclude all PII: emails, Slack handles, LinkedIn URLs, company names, real names
- Maintain the same data structure and field types as the examples
- Ensure synthetic data is realistic but not identifiable
- Output as a JSON object with a "participants" array containing the records

Examples to base your generation on:
{json.dumps(sampled_records, ensure_ascii=False, indent=2)}"""
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Generate {num_generated} synthetic participant records now."}]
        )
        
        content = response.content[0].text
        data = json.loads(content)
        if isinstance(data, dict) and "participants" in data:
            return data["participants"]
        elif isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        return []


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic participant data from CSV samples.")
    parser.add_argument("--csv", default="data/private_mlops_marchvc.csv", help="Input CSV path")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--compare", action="store_true", help="Show comparison output")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True, help="LLM provider to use")
    parser.add_argument("--model", choices=["gpt-5", "gpt-5-mini", "claude-sonnet-4"], required=True, help="Model to use")
    parser.add_argument("--out", default="synthetic_participants.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    # Load environment variables from .env if present
    load_dotenv()

    args = parse_args()
    csv_path = Path(args.csv)
    df = load_participants(csv_path)
    df = apply_column_renames(df, apply=True)

    sampled_df = sample_participants(df, num_samples=args.n, seed=args.seed)
    records_for_examples = prepare_llm_records(sampled_df, ALLOWED_LLM_FIELDS)

    # Generate synthetic data (LLM required)
    if args.provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not available. Install with: uv add openai")
        
        if args.model not in ["gpt-5", "gpt-5-mini"]:
            raise ValueError(f"Model {args.model} not supported for OpenAI provider. Use gpt-5 or gpt-5-mini.")
        
        generated = generate_with_openai_min(
            sampled_records=records_for_examples,
            model=args.model,
            num_generated=args.n,
            structured_output=True,
        )
    elif args.provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic package not available. Install with: uv add anthropic")
        
        if args.model != "claude-sonnet-4":
            raise ValueError(f"Model {args.model} not supported for Anthropic provider. Use claude-sonnet-4.")
        
        generated = generate_with_anthropic(
            sampled_records=records_for_examples,
            model=args.model,
            num_generated=args.n,
        )
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    # Convert to DataFrame and save as CSV
    df_generated = pd.DataFrame(generated)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_generated.to_csv(args.out, index=False)
    
    if args.compare:
        # Show comparison in JSON format
        combined = {
            "sampled": records_for_examples, 
            "generated": generated,
            "generated_count": len(generated)
        }
        print(json.dumps(combined, ensure_ascii=False, indent=2))
    else:
        # Just show the generated data
        print(json.dumps(generated, ensure_ascii=False, indent=2))


def generate_with_openai(
    sampled_records: List[Dict[str, Any]],
    model: str,
    num_generated: int,
    seed: Optional[int] = None,
    structured_output: bool = True,
) -> List[Dict[str, Any]]:
    """Backwards-compatible alias for generate_with_openai_min."""
    return generate_with_openai_min(
        sampled_records=sampled_records,
        model=model,
        num_generated=num_generated,
        structured_output=structured_output,
    )


if __name__ == "__main__":
    main()


