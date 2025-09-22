#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
"""Pydantic models for structured synthetic participant data.

These models define the schema we expect from LLM-based generation and help
validate and normalize outputs before writing them to disk.
"""

from typing import List, Optional, Literal
from pydantic import RootModel

from pydantic import BaseModel, Field


Role = Literal[
    "ML Engineer",
    "ML Scientist/Researcher",
    "Software Engineer",
    "Data Scientist",
    "Founder",
    "Management",
    "Sales and Marketing",
    "Data Engineer",
    "Student",
    "Other",
]

CareerStage = Literal[
    "Undergrad/New Grad",
    "Graduate Student",
    "1–3 Years",
    "3–5 Years",
    "5–10 Years",
    "10+ Years",
]

BuddyPreference = Literal[
    "No preference",
    "Buddy in a similar role",
]


class SyntheticParticipant(BaseModel):
    """Schema for a single synthetic participant record.

    Notes:
    - Excludes PII fields (email, Slack, LinkedIn, original name).
    - Includes company and location for context (synthetic versions).
    - `skills` is normalized to a list of strings.
    - Includes tracking fields for synthetic data management.
    """

    synthetic_id: str = Field(..., description="Unique identifier for this synthetic participant")
    source_participant_id: str = Field(..., description="Original respondent_id this was based on")
    synthetic_name: str = Field(..., min_length=1, description="Fabricated, non-PII name")

    role: Optional[Role] = Field(default=None, description="Participant role")
    career_stage: Optional[CareerStage] = Field(default=None, description="Career stage")
    location: Optional[str] = Field(
        default=None, description="Any level description of their location, city, country, region, etc."
    )
    company: Optional[str] = Field(default=None, description="Company or organization (synthetic)")
    buddy_preference: Optional[BuddyPreference] = Field(default=None, description="Buddy preference")

    summary: Optional[str] = Field(default=None, description="Short self-summary")
    skills: Optional[str] = Field(default=None, description="Free written list of skills/tags, can be list or free text/speakable text")
    buddy_preferences: Optional[str] = Field(
        default=None, description="Free-text description of buddy preferences"
    )


class SyntheticParticipantsResponse(BaseModel):
    """Response wrapper for synthetic participants array."""
    participants: List[SyntheticParticipant] = Field(..., description="List of synthetic participants")



