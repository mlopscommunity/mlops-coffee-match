from typing import Any, List
from openai import OpenAI
import pydantic
import pandas as pd
from dotenv import load_dotenv

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

# --- Helpers to convert tiers into numeric features ---
def _infer_career_stage_level(series: pd.Series) -> pd.Series:
    """Map a career-stage-like text column to numeric levels.
    Uses career_stage_level keys first, then simple heuristics (years extraction, student hints).
    Returns an int Series with fallback 3 (1-3 yrs bucket).
    """
    mapped = series.map(career_stage_level)
    if mapped.notna().any():
        return mapped.fillna(3).astype(int)

    import re

    def to_level(v: Any) -> int:
        s = str(v or "").strip().lower()
        # fuzzy containment against known labels
        for k, lvl in career_stage_level.items():
            if s in k.lower() or k.lower() in s:
                return lvl
        m = re.search(r"(\d+)\s*\+?", s)
        if m:
            y = int(m.group(1))
            if y >= 10:
                return 6
            if y >= 5:
                return 5
            if y >= 3:
                return 4
            if y >= 1:
                return 3
            return 2
        if any(tok in s for tok in ["student", "undergrad", "grad"]):
            return 2
        return 3

    return series.apply(to_level).astype(int)

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

    client = OpenAI()

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
            print(f"Error during OpenAI regionalization: {e}")
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


def embed_personal_summary(df: pd.DataFrame, batch_size: int = 100) -> pd.Series:
    """
    Embed combined text from columns: company, role, summary, skills using 'text-embedding-3-small'.
    Returns a Series of embeddings aligned to df.index.
    """
    client = OpenAI()
    model = "text-embedding-3-small"

    # Combine columns safely and compactly
    cols = ["company", "role", "summary", "skills"]
    combined = (
        df.reindex(columns=cols, fill_value="").astype(str)
          .agg(" | ".join, axis=1)
          .str.replace("\n", " ")
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
    )
    # Replace empty strings to avoid API error
    combined[combined == ""] = " "

    texts = combined.tolist()
    embeddings: List[Any] = [None] * len(texts)
    idx = combined.index

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embeds = [item.embedding for item in resp.data]
        embeddings[start:start + len(batch_embeds)] = batch_embeds

    return pd.Series(embeddings, index=idx, name="embedding")


def embed_buddy_preferences(df: pd.DataFrame, batch_size: int = 100) -> pd.Series:
    """
    Embed the `buddy_preferences` column using 'text-embedding-3-small'.
    Returns a Series of embeddings aligned to df.index.
    """
    if "buddy_preferences" not in df.columns:
        raise ValueError("DataFrame must contain the column 'buddy_preferences'.")

    client = OpenAI()
    model = "text-embedding-3-small"

    texts_series = (
        df["buddy_preferences"].fillna("").astype(str)
          .str.replace("\n", " ")
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
    )
    # Replace empty strings to avoid API error
    texts_series[texts_series == ""] = " "

    texts = texts_series.tolist()
    embeddings: List[Any] = [None] * len(texts)
    idx = texts_series.index

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embeds = [item.embedding for item in resp.data]
        embeddings[start:start + len(batch_embeds)] = batch_embeds

    return pd.Series(embeddings, index=idx, name="buddy_preferences_embedding")


def prepare_feature_columns(
    df: pd.DataFrame, model: str = "gpt-5-mini", batch_size: int = 20
) -> pd.DataFrame:
    """
    Produce feature columns per plan:
      - profile_embedding: from combined profile text (summary + skills + role + company)
      - preference_embedding: from buddy_preferences
      - regional_location: normalized from location
    Returns a new DataFrame with added columns.
    """
    print("Starting feature preparation...")
    out = df.copy()

    print("Generating profile embeddings...")
    out["profile_embedding"] = embed_personal_summary(out, batch_size=batch_size)

    if "buddy_preferences" in out.columns:
        print("Generating buddy preference embeddings...")
        out["preference_embedding"] = embed_buddy_preferences(out, batch_size=batch_size)
    else:
        out["preference_embedding"] = pd.Series([None] * len(out), index=out.index)

    if "location" in out.columns:
        print("Normalizing locations...")
        out["regional_location"] = normalize_location(
            df=out, model=model, location="location", batch_size=batch_size
        )
    else:
        out["regional_location"] = pd.Series([None] * len(out), index=out.index)

    # --- Numeric tiers and composite priority ---
    # Region tier (lower tier number = higher priority). Unknown -> worst tier.
    max_region = max(region_tiers.values()) if len(region_tiers) else 6
    out["region_tier"] = (
        out["regional_location"].map(region_tiers).fillna(max_region).astype(int)
    )
    # Normalize to [0,1] with tier 1 -> 1.0 priority
    denom = (max_region - 1) if (max_region - 1) != 0 else 1
    out["region_priority_norm"] = 1 - (out["region_tier"] - 1) / denom

    # Career stage numeric level from any present column, then normalize
    stage_col = None
    for c in [
        "career_stage",
        "career_stage_bucket",
        "experience_bucket",
        "years_of_experience_bucket",
    ]:
        if c in out.columns:
            stage_col = c
            break
    source_series = out[stage_col] if stage_col else pd.Series([None] * len(out), index=out.index)
    out["career_stage_level_num"] = _infer_career_stage_level(source_series)
    max_stage = max(career_stage_level.values()) if len(career_stage_level) else 6
    out["career_stage_norm"] = out["career_stage_level_num"].astype(float) / max_stage

    # Composite priority for pre-sorting candidate pools
    STAGE_W, REGION_W = 0.7, 0.3
    out["priority_score"] = (
        STAGE_W * out["career_stage_norm"] + REGION_W * out["region_priority_norm"]
    )

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

        # Save the featured DataFrame to a new CSV
        output_filename = f"{csv_path.stem}_featured.csv"
        output_path = project_root / "data_examples" / output_filename
        featured_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved featured data to:\n{output_path}")

    except Exception as e:
        print(f"Error during feature engineering test: {e}")