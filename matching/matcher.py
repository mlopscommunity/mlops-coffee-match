"""
This is the primary agent that will be used to match participants.
It will be responsible for:

- Ordering the participants by priority
- Looping through the remaining participants to:
    - Find the best n matches for the participant
    - Choosing the best match
    - Updating the assignment pool
    - Continue until all participants have been matched
- After all participants have been matched, it will run a post-matching review and optimization step
    - This will be done by a separate LLM that will be responsible for:
        - Reviewing the matches
        - Identifying any matches that are weak, illogical, or have poor justifications
        - Flagging them for re-matching
        - Re-assigning them to new partners

"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Dict, List, Callable
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from .matching_models import Match, LLMMatchRecommendation


def read_participants(participant_file_location: str) -> pd.DataFrame:
    """
    Read the participants from a CSV file and return as a pandas DataFrame.

    Args:
        participant_file_location (str): Path to the participant CSV file.

    Returns:
        pd.DataFrame: DataFrame containing participant data.
    """
    participants_df = pd.read_csv(participant_file_location)
    return participants_df

def order_participants(participants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Order the participants by priority.
    
    Priority order:
    1. buddy_preference (descending - "Buddy in a similar role" > "No preference")
    2. career_stage_level (ascending - earlier career stages get priority)
    3. region_tier (ascending - lower tier numbers get priority)
    
    Args:
        participants_df (pd.DataFrame): DataFrame containing participant data.
        
    Returns:
        pd.DataFrame: DataFrame sorted by priority.
    """
    return participants_df.sort_values(
        by=["buddy_preference", "career_stage_level", "region_tier"],
        ascending=[False, True, True]
    )

def _get_candidate_pool(
    participant: pd.Series,
    all_participants: pd.DataFrame,
    min_candidates: int = 20
) -> pd.DataFrame:
    """
    Get a pool of potential candidates for a single participant.

    Applies categorical filters and relaxes them if not enough candidates are found.

    Pseudocode:
    1. Exclude the participant from the candidate pool.
    2. Start with strict filters (career_stage_level +/- 1, region_tier +/- 1).
    3. If candidate count < min_candidates, gradually relax filters:
       - Expand career_stage_level range to +/- 2.
       - Expand region_tier range to +/- 2.
    4. Return the filtered DataFrame of candidates.
    """
    # Exclude the participant from the candidate pool
    candidates = all_participants[all_participants['participant_id'] != participant['participant_id']]

    career_stage_level = participant['career_stage_level']
    region_tier = participant['region_tier']

    # Define relaxation levels and iterate through them
    for level in [1, 2]:
        mask = (
            candidates['career_stage_level'].between(career_stage_level - level, career_stage_level + level) &
            candidates['region_tier'].between(region_tier - level, region_tier + level)
        )
        filtered_candidates = candidates[mask]

        if len(filtered_candidates) >= min_candidates:
            return filtered_candidates

    # If the loop completes, return the last (most relaxed) set of candidates
    return filtered_candidates


def _score_candidates(
    participant: pd.Series, candidates: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate similarity scores for a pool of candidates.

    Compares the participant's preference embedding to each candidate's profile embedding.

    Pseudocode:
    1. Get the participant's preference_embedding.
    2. Get the candidates' profile_embeddings.
    3. Calculate cosine similarity between the participant's embedding and all candidate embeddings.
    4. Add a 'similarity_score' column to the candidates DataFrame.
    5. Return the DataFrame with scores.
    """
    # Ensure there are candidates to score
    if candidates.empty:
        return candidates.copy().assign(similarity_score=pd.Series(dtype="float64"))

    # Get the participant's preference embedding and reshape for cosine_similarity
    participant_embedding = np.array(participant["buddy_preferences_embedding"]).reshape(
        1, -1
    )

    # Stack candidate profile embeddings into a 2D array
    candidate_embeddings = np.vstack(candidates["personal_summary_embedding"].tolist())

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        participant_embedding, candidate_embeddings
    ).flatten()

    # Add scores to a copy of the candidates DataFrame
    scored_candidates = candidates.copy()
    scored_candidates["similarity_score"] = similarity_scores

    return scored_candidates


## Note: Removed unused top_funnel_matches to avoid batch recomputation and keep the
## greedy per-seeker flow clear. llm_as_a_matcher handles shortlist for the current seeker.

def llm_as_a_matcher(
    matches_df: pd.DataFrame,
    top_n: int = 20,
) -> Tuple[Match, Optional[str]]:
    """Select a single match for the next seeker using filters, similarity, and LLM.

    This function assumes the input DataFrame is already ordered via order_participants.
    It will select the first row as the seeker, construct a filtered candidate pool,
    compute cosine similarity to create a shortlist, optionally call an LLM to choose
    among the shortlist, and then return a validated Match plus an optional intro message.

    The caller is responsible for removing both the seeker and the chosen buddy from the
    available pool before the next iteration.
    """
    # Preconditions & validation
    if matches_df.empty:
        raise ValueError("No available participants to match.")

    required_cols = {
        "participant_id",
        "career_stage_level",
        "region_tier",
        "buddy_preferences_embedding",
        "personal_summary_embedding",
    }
    missing = required_cols - set(matches_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Select seeker (row 0)
    seeker = matches_df.iloc[0]
    seeker_id = str(seeker["participant_id"])  # normalize to str for downstream comparisons

    # Build candidate pool (excludes seeker internally)
    candidates = _get_candidate_pool(seeker, matches_df, min_candidates=top_n)
    if candidates.empty:
        # Fallback: any other participant
        candidates = matches_df[matches_df["participant_id"] != seeker_id]
        if candidates.empty:
            # Only one person available; cannot form a pair
            raise RuntimeError("Only one participant available; cannot form a pair.")

    # Score and shortlist
    scored = _score_candidates(seeker, candidates)  # adds 'similarity_score'
    shortlist = (
        scored.sort_values("similarity_score", ascending=False)
        .head(top_n)
        .copy()
    )

    # If shortlist is empty (e.g., missing/invalid embeddings), pick deterministic fallback
    if shortlist.empty:
        chosen_buddy_id = str(candidates.iloc[0]["participant_id"])
        justification = "Fallback selection due to unavailable embeddings."
        icebreakers: list[str] = []
        intro_message: Optional[str] = None
        match = Match(
            participant_id=seeker_id,
            buddy_match_id=chosen_buddy_id,
            match_score=0.0,
            match_justification=justification,
            icebreaker_topics=icebreakers,
        )
        return match, intro_message

    # Prepare compact payload for potential LLM call (strings are pre-cleaned upstream)
    seeker_info = {
        "id": seeker_id,
        "role": seeker.get("role", ""),
        "region": seeker.get("region", ""),
        "summary": seeker.get("summary", ""),
        "company": seeker.get("company", ""),
        "buddy_preference": seeker.get("buddy_preference", ""),
        "buddy_preferences": seeker.get("buddy_preferences", ""),
    }
    candidates_info = []
    for _, row in shortlist.iterrows():
        candidates_info.append(
            {
                "id": str(row["participant_id"]),
                "role": row.get("role", ""),
                "region": row.get("region", ""),
                "summary": row.get("summary", ""),
                "company": row.get("company", ""),
                "buddy_preference": row.get("buddy_preference", ""),
                "buddy_preferences": row.get("buddy_preferences", ""),
                "similarity": float(row["similarity_score"]),
            }
        )

    # Call LLM chooser; on any error we will fallback to highest similarity
    rec: Optional[LLMMatchRecommendation] = None
    try:
        raw = call_llm_choose_buddy(seeker_info, candidates_info)
        if raw:
            rec = LLMMatchRecommendation.model_validate(raw)
    except Exception:
        rec = None

    shortlist_ids = set(shortlist["participant_id"].astype(str))
    if rec is None or rec.buddy_match_id not in shortlist_ids:
        chosen_buddy_id = str(shortlist.iloc[0]["participant_id"])  # highest similarity
        justification = (
            "Auto-selected by highest similarity due to missing/invalid LLM choice."
        )
        icebreakers = []
        intro_message = None
    else:
        chosen_buddy_id = rec.buddy_match_id
        justification = rec.match_justification
        icebreakers = rec.icebreaker_topics
        intro_message = rec.intro_message

    # Compute match_score from shortlist
    chosen_row = shortlist.loc[shortlist["participant_id"].astype(str) == chosen_buddy_id]
    match_score = (
        float(chosen_row["similarity_score"].iloc[0]) if not chosen_row.empty else 0.0
    )

    match = Match(
        participant_id=seeker_id,
        buddy_match_id=chosen_buddy_id,
        match_score=match_score,
        match_justification=justification,
        icebreaker_topics=icebreakers,
    )
    return match, intro_message

def match_participants(
    participants_df: pd.DataFrame,
    top_n: int = 20,
    progress_fn: Optional[Callable[[int, int, str, str, float], None]] = None,
) -> pd.DataFrame:
    """Run the full matching loop over available participants.

    This orchestrator orders participants using `order_participants`, then repeatedly
    calls `llm_as_a_matcher` to select a buddy for the current seeker (the first row
    of the ordered working DataFrame). After each selection, it removes both the
    seeker and the chosen buddy from the working pool and continues until fewer than
    two participants remain. If one participant remains at the end, it may form a
    triad with the most recent pair.

    Args:
        participants_df: Input DataFrame containing the available participants and
            required columns: `participant_id`, `career_stage_level`, `region_tier`,
            `buddy_preferences_embedding`, `personal_summary_embedding`. Additional
            human-readable fields (e.g., `role`, `region`, `summary`) are used for
            LLM prompting.
        top_n: Maximum number of shortlisted candidates per seeker to consider when
            delegating final selection to the LLM.

    Returns:
        pd.DataFrame: A tidy DataFrame of matches with columns:
            - `participant_A_id`
            - `participant_B_id`
            - `match_score`
            - `llm_justification`
            - `icebreaker_topics`
            - `intro_message` (optional, may be None)
    """
    # Preconditions
    if participants_df is None or len(participants_df) < 2:
        return pd.DataFrame(
            columns=[
                "participant_A_id",
                "participant_B_id",
                "match_score",
                "llm_justification",
                "icebreaker_topics",
                "intro_message",
            ]
        )

    # Order participants once
    work_df = order_participants(participants_df).reset_index(drop=True)

    results: list[dict] = []
    total_pairs = len(work_df) // 2
    pairs_done = 0

    # Iterate until fewer than 2 participants remain
    while len(work_df) >= 2:
        match, intro = llm_as_a_matcher(work_df, top_n=top_n)

        results.append(
            {
                "participant_A_id": match.participant_id,
                "participant_B_id": match.buddy_match_id,
                "match_score": match.match_score,
                "llm_justification": match.match_justification,
                "icebreaker_topics": match.icebreaker_topics,
                "intro_message": intro,
            }
        )

        # Remove both ids from the working pool
        to_remove = {str(match.participant_id), str(match.buddy_match_id)}
        work_df = work_df[~work_df["participant_id"].astype(str).isin(to_remove)].reset_index(
            drop=True
        )

        # Progress callback
        pairs_done += 1
        if progress_fn is not None:
            try:
                progress_fn(pairs_done, total_pairs, str(match.participant_id), str(match.buddy_match_id), float(match.match_score))
            except Exception:
                # Ignore progress callback errors to avoid breaking the run
                pass

    # Optional: triad handling can be implemented later if desired
    return pd.DataFrame(results)


def call_llm_choose_buddy(
    seeker: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Choose the best buddy from a shortlist using OpenAI Responses API.

    This helper prepares a compact, PII-safe payload from the `seeker` and
    `candidates`, calls the OpenAI Responses API with structured outputs, and
    validates the result against `LLMMatchRecommendation`.

    Args:
        seeker: Minimal seeker context with keys like `id`, `role`, `region`, `summary`.
        candidates: Shortlisted candidates; each item should include `id`, `role`,
            `region`, `summary`, and a numeric `similarity`.
        model: Optional model name; defaults to env `OPENAI_MODEL` or a sensible default.
        temperature: Decoding temperature; prefer low for deterministic choices.
        max_retries: Number of times to retry on transient failures.

    Returns:
        Dict compatible with `LLMMatchRecommendation`:
        `{ "buddy_match_id": str, "match_justification": str,
           "icebreaker_topics": List[str], "intro_message": Optional[str] }`.

    Pseudocode:
        1) Build system prompt with clear selection rules and JSON-only output.
        2) Build user payload: { seeker, candidates } (compact).
        3) client = OpenAI(); use Responses API parse with text_format=LLMMatchRecommendation.
        4) Retry up to max_retries on transient errors.
        5) On success: return parsed.model_dump(); on error: return {} to trigger fallback.
    """
    from openai import OpenAI
    import json
    import time

    chosen_model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    SYSTEM_PROMPT = (
        "You are a careful pairing assistant for a virtual coffee program. "
        "Given a seeker and a shortlist of candidates, choose exactly one buddy from the provided ids. "
        "When writing justifications and intros, focus on concrete profile evidence: role/skills, "
        "background/summary themes, company, preference alignment, and location/region context. "
        "Do NOT mention numeric similarity scores or that someone is the 'highest similarity'. "
        "Do NOT include raw participant ids or any identifier strings in natural language text. "
        "Write human-friendly sentences that reference themes (e.g., platform MLOps, model monitoring, startup ops, EU timezone overlap). "
        "Do not invent personal names or use placeholder tokens like [seeker name] or [buddy name]. "
        "If names are not provided, write a short, generic intro that explains why they were paired "
        "using themes like roles, skills, interests, or regions. "
        "Respond ONLY with the structured fields defined by the schema."
    )

    # Compact user payload
    user_payload = {
        "seeker": seeker,
        "candidates": candidates,
        "instructions": [
            "Pick exactly one candidate whose id appears in candidates.",
            "Keep justification to 1-2 sentences that reference preferences, background/summary, roles/skills, and region/timezone where relevant.",
            "Never mention similarity scores and never cite raw ids in the text.",
            "Include 2-4 concise icebreaker topics derived from overlapping interests/skills, complementary experience, or regional context.",
            "Do not fabricate names or use placeholder tokens; if names are unknown, write a friendly generic intro explaining the pairing themes.",
        ],
    }

    client = OpenAI()
    messages: Any = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            parsed = client.responses.parse(  # type: ignore[call-arg]
                model=str(chosen_model),
                input=messages,
                text_format=LLMMatchRecommendation,  # type: ignore[arg-type]
            )
            if getattr(parsed, "output_parsed", None) is None:  # type: ignore[attr-defined]
                raise ValueError("Structured parse returned None")
            rec: LLMMatchRecommendation = parsed.output_parsed  # type: ignore[assignment]
            return rec.model_dump()
        except Exception:
            if attempt < max_retries:
                time.sleep(0.8 * attempt)
                continue
            return {}
    # Final fallback (should not reach here due to returns above)
    return {}