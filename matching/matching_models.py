# pydantic models for the matching system
from typing import List
from pydantic import BaseModel, Field

class Match(BaseModel):
    participant_id: str
    buddy_match_id: str
    match_score: float = Field(default=0, description="2 for a great match, 1 for a good match, 0 for a bad match")
    match_justification: str
    icebreaker_topics: List[str]
    