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

    texts = texts_series.tolist()
    embeddings: List[Any] = [None] * len(texts)
    idx = texts_series.index

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embeds = [item.embedding for item in resp.data]
        embeddings[start:start + len(batch_embeds)] = batch_embeds

    return pd.Series(embeddings, index=idx, name="buddy_preferences_embedding")


def prepare_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce feature columns per plan:
      - profile_embedding: from combined profile text (summary + skills + role + company)
      - preference_embedding: from buddy_preferences
    Returns a new DataFrame with added columns.
    """
    out = df.copy()
    out["profile_embedding"] = embed_personal_summary(out)
    if "buddy_preferences" in out.columns:
        out["preference_embedding"] = embed_buddy_preferences(out)
    else:
        out["preference_embedding"] = pd.Series([None] * len(out), index=out.index)
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
        csv_path = project_root / "data" / "synthetic_participants_20250919_120846.csv"
        df = pd.read_csv(csv_path)
        
        # Location normalization test (commented out)
        # model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        # batch_size = 20
        # regional_series = normalize_location(
        #     df=df,
        #     model=model,
        #     location="location",
        #     batch_size=batch_size,
        # )
        # df["regional_location"] = regional_series
        # print(df[["location", "regional_location"]].head(15).to_string(index=False))

        # Embeddings tests on a small sample
        sample = df.head(5)

        emb_summary = embed_personal_summary(sample, batch_size=20)
        print(f"Summary embeddings: {len(emb_summary)} rows. First len: {len(emb_summary.iloc[0])}")
        print("First summary vec (8 dims):", emb_summary.iloc[0][:8])

        if "buddy_preferences" in sample.columns:
            emb_prefs = embed_buddy_preferences(sample, batch_size=20)
            print(f"Buddy prefs embeddings: {len(emb_prefs)} rows. First len: {len(emb_prefs.iloc[0])}")
            print("First buddy prefs vec (8 dims):", emb_prefs.iloc[0][:8])
    except Exception as e:
        print(f"Error testing normalize_location: {e}")