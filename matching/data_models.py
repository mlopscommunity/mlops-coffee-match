from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Participant(BaseModel):
    """
    Represents a single participant in the coffee chat matching program.
    """

    synthetic_id: str
    source_participant_id: Optional[str] = None
    synthetic_name: str
    company: Optional[str] = None
    role: Optional[str] = None
    career_stage: Optional[str] = None
    location: Optional[str] = None
    buddy_preference: Optional[str] = None
    buddy_preferences: Optional[str] = None
    summary: Optional[str] = None
    skills: Optional[str] = None

    # Feature Engineered Fields
    profile_embedding: Optional[List[float]] = None
    preference_embedding: Optional[List[float]] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class Match(BaseModel):
    """
    Represents a single match between two participants.
    """

    participant_a_id: str
    participant_b_id: str
    llm_justification: str
    match_score: Optional[float] = None
