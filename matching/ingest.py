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
