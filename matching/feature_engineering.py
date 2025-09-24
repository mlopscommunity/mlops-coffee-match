from typing import Any, List

import pandas as pd
import pydantic
from dotenv import load_dotenv
from openai import OpenAI

from .ingest import FIELD_ALIASES, get_alias_series, resolve_aliases

class SubregionList(pydantic.BaseModel):
    subregions: List[str]

region_tiers = {
    "North America": 1,
    "Western Europe": 2,
    "Eastern Europe": 2,
    "East Asia": 3,
    "South Asia": 4,
    "Southeast Asia": 4,
    "Middle East": 4,
    "South America": 5,
    "Northern Africa": 5,
    "Central Asia": 5,
    "Australia/New Zealand": 5,
    "Sub-Saharan Africa": 5,
    "Oceania": 6,
    "Antarctica": 6
}

career_stage_level = {
  "Undergrad / New Grad": 1,
  "Graduate Student": 2,
  "1 - 3 Years of Experience": 3,
  "3 - 5 Years of Experience": 4,
  "5 - 10 Years of Experience": 5,
  "10+ Years of Experience": 6,
}

# Note: No heuristic helpers. We expect strict dropdown values for career_stage.

def gen_regional_locations(locations: pd.Series, model: str) -> pd.Series:
    """
    Generate regional locations for a given pandas Series of locations.

    Args:
        locations (pd.Series): Series containing location strings.
        model (str): Model identifier to use for regionalization.

    Returns:
        pd.Series: Series of regionalized location strings.
    """
    # Placeholder: Replace with actual logic to generate regional locations
    # For now, just return the input as-is
    
    import json

    # Define a system prompt to instruct the model to map locations to a set of global subregions.
    SYSTEM_PROMPT = (
        "You are a helpful assistant that maps a list of locations to their corresponding global subregions. "
        "Use the following subregions: North America, South America, Western Europe, Eastern Europe, "
        "Northern Africa, Sub-Saharan Africa, Middle East, Central Asia, South Asia, East Asia, Southeast Asia, "
        "Oceania, Australia/New Zealand, and Antarctica. "
        "Return a JSON object with key 'subregions' whose value is a list of subregion names, "
        "one for each input location, in the same order."
    )

    try:
        client = OpenAI()
    except Exception as e:
        print(f"Warning: OpenAI client init failed for regionalization ({e}). Returning original locations.")
        return locations

    # Build messages input per Responses API structured outputs guide
    input_payload = json.dumps(locations.tolist())
    messages: Any = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_payload},
    ]

    import time
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            # Structured outputs using parse() with a Pydantic model and messages
            parsed = client.responses.parse(  # type: ignore[call-arg]
                model=model,
                input=messages,
                text_format=SubregionList,  # type: ignore[arg-type]
            )
            if getattr(parsed, "output_parsed", None) is None:  # type: ignore[attr-defined]
                raise ValueError("Structured parse returned None")
            subregion_list: SubregionList = parsed.output_parsed  # type: ignore[assignment]
            return pd.Series(subregion_list.subregions, index=locations.index)
        except Exception as e:
            if attempt < max_attempts:
                time.sleep(0.8 * attempt)
                continue
            print(f"Warning: OpenAI regionalization failed ({e}). Returning original locations for this batch.")
            return locations
    # Final fallback if loop exits without returning
    return locations

def normalize_location(df: pd.DataFrame, model: str, location: str, batch_size: int) -> pd.Series:
    """
    Uses gen_regional_locations to generate regional locations for a given location column.

    Args:
        df (pd.DataFrame): DataFrame containing the location column.
        model (str): Model identifier to use for regionalization.
        location (str): Name of the column containing location strings.
        batch_size (int): Number of rows to process per batch.

    Returns:
        pd.Series: Series of regionalized location strings.
    """
    # Ensure the DataFrame contains the required location column
    if location not in df.columns:
        raise ValueError(f"DataFrame must contain the column '{location}'.")

    # Only process non-null locations for efficiency
    locations = df[location]

    # Process in batches to avoid memory issues with large DataFrames
    result = []
    for i in range(0, len(locations), batch_size):
        batch = locations.iloc[i:i + batch_size]
        regionalized = gen_regional_locations(batch, model)
        result.append(regionalized)

    # Concatenate the results back into a single Series (preserve original indices)
    return pd.concat(result)


def _combine_text_fields(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    pieces = []
    for key in keys:
        series = get_alias_series(df, key)
        if pd.api.types.is_object_dtype(series) or series.dtype == "O":
            pieces.append(series.fillna("").astype(str))
        else:
            pieces.append(series.astype(str).fillna(""))
    if not pieces:
        return pd.Series([" "] * len(df), index=df.index)
    stacked = pd.concat(pieces, axis=1)
    combined = stacked.apply(
        lambda row: " | ".join([part for part in row if isinstance(part, str) and part.strip()]) or " ",
        axis=1,
    )
    return combined.str.replace("\n", " ").str.replace(r"\s+", " ", regex=True).str.strip().replace({"": " "})


def embed_personal_summary(df: pd.DataFrame, batch_size: int = 100) -> pd.Series:
    """Embed combined profile text using 'text-embedding-3-small'."""

    model = "text-embedding-3-small"
    combined = _combine_text_fields(df, ["company", "role", "summary", "skills"])

    texts = combined.tolist()
    embeddings: List[Any] = [None] * len(texts)
    idx = combined.index

    try:
        client = OpenAI()
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            batch_embeds = [item.embedding for item in resp.data]
            embeddings[start:start + len(batch_embeds)] = batch_embeds
        return pd.Series(embeddings, index=idx, name="profile_embedding")
    except Exception as e:
        print(f"Warning: OpenAI profile embeddings failed ({e}). Will rely on TF-IDF fallback downstream.")
        return pd.Series([None] * len(idx), index=idx, name="profile_embedding")


def embed_buddy_preferences(df: pd.DataFrame, batch_size: int = 100) -> pd.Series:
    """Embed buddy preference text."""

    model = "text-embedding-3-small"
    texts_series = _combine_text_fields(df, ["buddy_preferences"])
    texts = texts_series.tolist()
    embeddings: List[Any] = [None] * len(texts)
    idx = texts_series.index

    try:
        client = OpenAI()
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            batch_embeds = [item.embedding for item in resp.data]
            embeddings[start:start + len(batch_embeds)] = batch_embeds
        return pd.Series(embeddings, index=idx, name="preference_embedding")
    except Exception as e:
        print(f"Warning: OpenAI preference embeddings failed ({e}). Will rely on TF-IDF fallback downstream.")
        return pd.Series([None] * len(idx), index=idx, name="preference_embedding")


def prepare_feature_columns(
    df: pd.DataFrame, model: str = "gpt-5-mini", batch_size: int = 20
) -> pd.DataFrame:
    """
    Produce feature columns per plan:
      - profile_embedding: from combined profile text (summary + skills + role + company)
      - preference_embedding: from buddy_preferences
      - regional_location: normalized from location
      - region_tier: numeric tier level for regional_location (1-6)
      - career_stage_level: numeric level for career_stage (1-6)
    Returns a new DataFrame with added columns.
    """
    print("Starting feature preparation...")
    out = df.copy()
    alias_map = resolve_aliases(out)

    print("Generating profile embeddings...")
    out["profile_embedding"] = embed_personal_summary(out, batch_size=batch_size)

    print("Generating buddy preference embeddings...")
    out["preference_embedding"] = embed_buddy_preferences(out, batch_size=batch_size)

    loc_col = alias_map.get("location")
    if loc_col is not None:
        print("Normalizing locations...")
        out["regional_location"] = normalize_location(
            df=out, model=model, location=loc_col, batch_size=batch_size
        )
    else:
        out["regional_location"] = pd.Series([None] * len(out), index=out.index)

    # Map regional locations to tier levels
    if "regional_location" in out.columns:
        print("Mapping regional locations to tier levels...")
        out["region_tier"] = out["regional_location"].map(region_tiers)
    else:
        out["region_tier"] = pd.Series([None] * len(out), index=out.index)

    # Map career stages to level numbers
    if "career_stage" in out.columns:
        print("Mapping career stages to level numbers...")
        out["career_stage_level"] = out["career_stage"].map(career_stage_level)
    else:
        out["career_stage_level"] = pd.Series([None] * len(out), index=out.index)

    print("Feature preparation complete.")
    return out


if __name__ == "__main__":
    import os
    from pathlib import Path

    try:
        # Load environment from .env if present
        load_dotenv()

        # Helpful check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your environment or a .env file."
            )

        project_root = Path(__file__).resolve().parents[1]
        csv_path = (
            project_root / "data_examples" / "synthetic_participants_20250922_151750.csv"
        )
        df = pd.read_csv(csv_path)

        # Prepare features on the full dataframe
        print(f"\nProcessing {len(df)} rows from {csv_path.name}...")
        df_to_process = df.copy()

        model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

        featured_df = prepare_feature_columns(df_to_process, model=model, batch_size=20)

        print("\n--- Feature Engineering Results ---")
        print(featured_df.head())

        if (
            "profile_embedding" in featured_df.columns
            and featured_df["profile_embedding"].notna().any()
        ):
            print("\nProfile embedding sample (first 8 dims):")
            print(featured_df["profile_embedding"].iloc[0][:8])

        if (
            "preference_embedding" in featured_df.columns
            and featured_df["preference_embedding"].notna().any()
        ):
            print("\nPreference embedding sample (first 8 dims):")
            print(featured_df["preference_embedding"].iloc[0][:8])

        if "regional_location" in featured_df.columns:
            print("\nLocation normalization sample:")
            print(featured_df[["location", "regional_location"]].head())

        if "region_tier" in featured_df.columns:
            print("\nRegion tier mapping sample:")
            print(featured_df[["regional_location", "region_tier"]].head())

        if "career_stage_level" in featured_df.columns:
            print("\nCareer stage level mapping sample:")
            print(featured_df[["career_stage", "career_stage_level"]].head())

        # Save the featured DataFrame to a new CSV
        output_filename = f"{csv_path.stem}_featured.csv"
        output_path = project_root / "data_examples" / output_filename
        featured_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved featured data to:\n{output_path}")

    except Exception as e:
        print(f"Error during feature engineering test: {e}")