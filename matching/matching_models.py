# pydantic models for the matching system
from typing import List, Optional
from pydantic import BaseModel, Field

class Match(BaseModel):
    """Canonical match record produced by the matcher pipeline.

    Fields:
        participant_id: The seeker (origin) participant identifier.
        buddy_match_id: The selected buddy's participant identifier.
        match_score: Cosine similarity score in [0, 1] between seeker preferences and buddy profile.
        match_justification: Short, human-readable rationale for the pairing.
        icebreaker_topics: Optional list of conversation starters for the pair.
    """

    participant_id: str
    buddy_match_id: str
    match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between 0.0 and 1.0",
    )
    match_justification: str
    icebreaker_topics: List[str] = Field(default_factory=list, description="Conversation starters for the pair")


class LLMMatchRecommendation(BaseModel):
    """Subset returned by the LLM for final selection.

    Fields:
        buddy_match_id: Selected buddy's participant identifier (must be in shortlist).
        match_justification: Reasoning for why this is a good match.
        icebreaker_topics: Optional list of conversation starters.
        intro_message: Optional draft intro the organizer can send to both.
    """

    buddy_match_id: str
    match_justification: str
    icebreaker_topics: List[str] = Field(default_factory=list)
    intro_message: Optional[str] = None
    