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
import pandas as pd

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

def match_participants(participants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the participants.
    """
    return participants_df