#!/usr/bin/env python3
"""Generate synthetic participant data."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import shortuuid
from dotenv import load_dotenv
from openai import OpenAI
# from synth_models import SyntheticParticipantsResponse


def load_participants(csv_path: Path) -> pd.DataFrame:
    """Load participants CSV into a DataFrame."""
    return pd.read_csv(csv_path)


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


def sample_participants(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """Sample participants without replacement."""
    return df.sample(n=num_samples, replace=False)


def prepare_llm_records(df: pd.DataFrame, allowed_fields: set[str]) -> List[Dict[str, Any]]:
    """Prepare DataFrame records for LLM prompting."""
    filtered_df = df[list(allowed_fields)].where(pd.notnull(df[list(allowed_fields)]), None)
    return filtered_df.to_dict(orient="records")


def generate_synthetic_id() -> str:
    """Generate a short UUID for synthetic participant identification."""
    return shortuuid.uuid()


def generate_timestamped_filename(prefix: str, extension: str) -> str:
    """Generate a timestamped filename for output files.
    
    Args:
        prefix: Filename prefix (e.g., "synthetic_participants")
        extension: File extension (e.g., "csv")
        
    Returns:
        Timestamped filename (e.g., "synthetic_participants_20241220_143022.csv")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def add_synthetic_metadata(records: List[Dict[str, Any]], source_ids: List[str]) -> List[Dict[str, Any]]:
    """Add synthetic metadata to generated records."""
    return [
        {
            "synthetic_id": generate_synthetic_id(),
            "source_participant_id": source_ids[i % len(source_ids)],
            **record
        }
        for i, record in enumerate(records)
    ]


def generate_synthetic(records: List[Dict[str, Any]], source_ids: List[str], num_generated: int, provider: str, model: str) -> List[Dict[str, Any]]:
    """Generate synthetic participants using specified provider and model."""
    prompt = f"""Generate {num_generated} synthetic participant records based on these examples. 
Use fake names and paraphrase other fields. Exclude PII. Output as JSON with "participants" array.

Examples: {json.dumps(records, ensure_ascii=False, indent=2)}"""
    
    if provider == "openai":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content or "[]"
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    data = json.loads(content)
    raw_records = data.get("participants", data) if isinstance(data, dict) else data
    return add_synthetic_metadata(raw_records, source_ids)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic participant data.")
    parser.add_argument("--total", type=int, required=True, help="Number of synthetic participants to generate")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True, help="LLM provider")
    parser.add_argument("--model", required=True, help="Model name (gpt-5, gpt-5-mini, or claude-sonnet-4)")
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    load_dotenv()
    args = parse_args()
    
    # Load data
    df = load_participants(Path("data/private_mlops_marchvc.csv"))
    df = apply_column_renames(df, apply=True)

    # Setup output
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filename = generate_timestamped_filename("synthetic_participants", "csv")
    output_path = data_dir / filename

    print(f"Generating {args.total} synthetic participants...")
    all_generated: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    while len(all_generated) < args.total:
        remaining = args.total - len(all_generated)
        batch_size = min(args.batch_size, remaining)
        
        # Sample without replacement
        available_df = df[~df['respondent_id'].isin(used_ids)]
        if len(available_df) == 0:
            print("No more source participants available")
            break
            
        sampled_df = available_df.sample(n=min(batch_size, len(available_df)))
        used_ids.update(sampled_df['respondent_id'])
        
        # Generate
        records = prepare_llm_records(sampled_df, ALLOWED_LLM_FIELDS)
        source_ids = sampled_df['respondent_id'].tolist()
        generated = generate_synthetic(records, source_ids, batch_size, args.provider, args.model)
        
        all_generated.extend(generated)
        print(f"Generated {len(generated)} (total: {len(all_generated)})")

    # Save
    pd.DataFrame(all_generated).to_csv(output_path, index=False)
    print(f"Saved {len(all_generated)} synthetic participants to {output_path}")


if __name__ == "__main__":
    main()


