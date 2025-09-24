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
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
    candidate_embeddings = np.vstack(candidates["personal_summary_embedding"].values)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        participant_embedding, candidate_embeddings
    ).flatten()

    # Add scores to a copy of the candidates DataFrame
    scored_candidates = candidates.copy()
    scored_candidates["similarity_score"] = similarity_scores

    return scored_candidates


def top_funnel_matches(
    participants_df: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """
    For each participant, find the top 10-20 potential matches.

    Orchestrates the process of getting and scoring candidates.
    
    Pseudocode:
    1. Initialize an empty list to store top matches for each participant.
    2. For each participant in the DataFrame:
       a. Get a candidate pool using _get_candidate_pool().
       b. Score the candidates using _score_candidates().
       c. Rank the scored candidates by 'similarity_score' in descending order.
       d. Select the top 20 candidates and add them to the list.
    3. Concatenate all top matches into a single DataFrame.
    4. Return the final DataFrame.
    """
    all_top_matches = []
    for _, participant in participants_df.iterrows():
        # a. Get a candidate pool
        candidate_pool = _get_candidate_pool(
            participant=participant, all_participants=participants_df
        )

        # b. Score the candidates
        scored_candidates = _score_candidates(
            participant=participant, candidates=candidate_pool
        )

        # c. Rank and d. Select top N
        top_matches = scored_candidates.sort_values(
            by="similarity_score", ascending=False
        ).head(top_n)

        # Add origin participant ID for tracking
        top_matches = top_matches.assign(origin_participant_id=participant["participant_id"])
        all_top_matches.append(top_matches)

    # 3. Concatenate all top matches
    if not all_top_matches:
        return pd.DataFrame()

    return pd.concat(all_top_matches).reset_index(drop=True)

def llm_as_a_matcher(matches_df: pd.DataFrame) -> pd.DataFrame:
    return None

def match_participants(participants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the participants.
    """
    return participants_df