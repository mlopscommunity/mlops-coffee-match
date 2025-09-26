from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


FIELD_ALIASES: Dict[str, List[str]] = {
    "id": ["source_participant_id", "Respondent ID", "synthetic_id", "Participant ID"],
    "email": ["email", "Your email address"],
    "name": ["synthetic_name", "Your name", "Name"],
    "company": ["company", "Where do you work?"],
    "role": ["role", "Which option best represents your role?"],
    "career_stage": ["career_stage", "How would you characterize your current career stage?"],
    "location": ["location", "Where are you based?"],
    "buddy_preference": ["buddy_preference", "Do you have a preference on who your buddy should be?"],
    "summary": ["summary", "Tweet-sized summary of yourself"],
    "skills": ["skills", "What are your skills? In which fields do you specialize?"],
    "buddy_preferences": ["buddy_preferences", "Describe what you want your buddy to be like."],
    "linkedin": ["linkedin", "Your LinkedIn URL"],
    "slack_handle": ["slack_handle", "What is your Slack Handle?"],
}


def get_alias_column(df: pd.DataFrame, key: str) -> Optional[str]:
    for candidate in FIELD_ALIASES.get(key, []):
        if candidate in df.columns:
            return candidate
    return None


def get_alias_series(df: pd.DataFrame, key: str, default: str = "") -> pd.Series:
    col = get_alias_column(df, key)
    if col is not None:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def resolve_aliases(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {key: get_alias_column(df, key) for key in FIELD_ALIASES}


def clean_coffee_df(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleanup that preserves the original CoffeeMatch schema."""

    out = df.copy()
    out.columns = [col.strip() if isinstance(col, str) else col for col in out.columns]
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("\n", " ")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace({"nan": None, "None": None, "": None})
            )
    return out


def clean_coffee_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return clean_coffee_df(df)


def ensure_standard_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard string fields exist and are clean for downstream LLM prompts.

    Creates/normalizes the following columns if missing: 'role', 'summary', 'company',
    'buddy_preference', 'buddy_preferences', and 'region'. Values are coerced to
    trimmed strings with NaNs/None replaced by empty strings.

    Args:
        df: Input DataFrame.

    Returns:
        A copy of the DataFrame with the standardized columns present and cleaned.
    """
    out = df.copy()
    alias_map = resolve_aliases(out)

    def _copy_if_missing(target: str, alias_key: str) -> None:
        if target not in out.columns:
            alias_col = alias_map.get(alias_key)
            if alias_col is not None and alias_col in out.columns:
                out[target] = out[alias_col]

    # Create columns from aliases when absent
    _copy_if_missing("role", "role")
    _copy_if_missing("summary", "summary")
    _copy_if_missing("company", "company")
    _copy_if_missing("buddy_preference", "buddy_preference")
    _copy_if_missing("buddy_preferences", "buddy_preferences")

    # Map regional_location (feature engineered) to region if region missing
    if "region" not in out.columns and "regional_location" in out.columns:
        out["region"] = out["regional_location"]

    # Coerce to clean strings for LLM payload fields
    for col in [
        "role",
        "summary",
        "company",
        "buddy_preference",
        "buddy_preferences",
        "region",
    ]:
        if col not in out.columns:
            out[col] = ""
        series = out[col]
        # Convert non-strings, replace NaN/None-like tokens, and strip
        series = series.astype(str)
        series = series.replace({"nan": "", "None": "", "NaN": "", "NULL": ""})
        series = series.str.replace("\n", " ")
        series = series.str.replace(r"\s+", " ", regex=True)
        out[col] = series.str.strip()

    return out
