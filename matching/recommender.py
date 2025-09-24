from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ProfileEmb = List[float]


@dataclass(frozen=True)
class ScoreWeights:
    w_mutual_pref: float = 0.40
    w_profile_sim: float = 0.20
    w_role: float = 0.15
    w_career_stage: float = 0.15
    w_location: float = 0.10


def _safe_cosine(a: Optional[ProfileEmb], b: Optional[ProfileEmb]) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=float).reshape(1, -1)
    vb = np.asarray(b, dtype=float).reshape(1, -1)
    # handle size mismatch by trimming to min length
    m = min(va.shape[1], vb.shape[1])
    if m == 0:
        return 0.0
    if va.shape[1] != m:
        va = va[:, :m]
    if vb.shape[1] != m:
        vb = vb[:, :m]
    sim = cosine_similarity(va, vb)[0, 0]
    # convert [-1,1] -> [0,1]
    return float((sim + 1.0) / 2.0)


def _text_fallback_embeddings(df: pd.DataFrame) -> Tuple[List[ProfileEmb], List[ProfileEmb]]:
    """If profile/preference embeddings are missing, build TF-IDF vectors as fallback."""
    # profile text
    profile_text = (
        df.reindex(columns=["company", "role", "summary", "skills"], fill_value="")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.replace("\n", " ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    ).tolist()

    pref_text = (
        df.get("buddy_preferences", pd.Series([""] * len(df), index=df.index))
        .astype(str)
        .str.replace("\n", " ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    ).tolist()

    vec = TfidfVectorizer(max_features=2048)
    prof_mat = vec.fit_transform(profile_text).astype(np.float32)
    pref_vec = TfidfVectorizer(max_features=2048)
    pref_mat = pref_vec.fit_transform(pref_text).astype(np.float32)
    prof = [row.toarray().ravel().tolist() for row in prof_mat]
    pref = [row.toarray().ravel().tolist() for row in pref_mat]
    return prof, pref


def _location_proximity(a_region: Any, b_region: Any) -> float:
    if pd.isna(a_region) or pd.isna(b_region):
        return 0.3  # weak neutral
    if a_region == b_region:
        return 1.0
    # unknown regions are treated as distant
    return 0.4


def _career_stage_affinity(a: int, b: int, max_gap_soft: int = 3) -> float:
    if a is None or b is None:
        return 0.5
    gap = abs(int(a) - int(b))
    if gap == 0:
        return 1.0
    if gap >= max_gap_soft:
        return 0.2
    # linear decay within soft window
    return max(0.2, 1.0 - gap / max_gap_soft)


def _role_match_boost(a_role: Optional[str], b_role: Optional[str]) -> float:
    if not a_role or not b_role:
        return 0.0
    a = str(a_role).strip().lower()
    b = str(b_role).strip().lower()
    if a == b:
        return 1.0
    # coarse families
    families = {
        "ml engineer": "technical",
        "machine learning engineer": "technical",
        "data scientist": "technical",
        "software engineer": "technical",
        "data engineer": "technical",
        "ml researcher": "research",
        "ml scientist": "research",
        "product manager": "pm",
        "founder": "leadership",
        "management": "leadership",
        "student": "student",
    }
    fa = next((v for k, v in families.items() if k in a), None)
    fb = next((v for k, v in families.items() if k in b), None)
    if fa and fb and fa == fb:
        return 0.6
    return 0.1 if any(t in a for t in b.split()) else 0.0


def _maybe_parse_vec(val: Any) -> Optional[List[float]]:
    """Convert stringified list to list[float] if needed; pass through lists/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            import json
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [float(x) for x in arr]
            except Exception:
                return None
        # treat other strings as None to fall back to TF-IDF later
        return None
    return None


def _ensure_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Try to parse any existing embedding strings to lists
    if "profile_embedding" in out.columns:
        out["profile_embedding"] = out["profile_embedding"].apply(_maybe_parse_vec)
    if "preference_embedding" in out.columns:
        out["preference_embedding"] = out["preference_embedding"].apply(_maybe_parse_vec)

    if "profile_embedding" not in out.columns or out["profile_embedding"].isna().all():
        prof, pref = _text_fallback_embeddings(out)
        out["profile_embedding"] = prof
        out["preference_embedding"] = pref
    elif "preference_embedding" not in out.columns or out["preference_embedding"].isna().all():
        # build only pref if missing
        _, pref = _text_fallback_embeddings(out)
        out["preference_embedding"] = pref
    return out


# ---- Tier fallbacks (region/career stage/priority) ----
def _infer_career_stage_level(series: pd.Series) -> pd.Series:
    """Best-effort mapping of career stage text to numeric levels.
    Mirrors the logic in feature_engineering._infer_career_stage_level but kept local to avoid tight coupling.
    Levels: 1 (Undergrad) .. 6 (10+ years).
    """
    mapping = {
        "Undergrad / New Grad": 1,
        "Graduate Student": 2,
        "1 - 3 Years of Experience": 3,
        "3 - 5 Years of Experience": 4,
        "5 - 10 Years of Experience": 5,
        "10+ Years of Experience": 6,
    }
    mapped = series.map(mapping)
    if mapped.notna().any():
        return mapped.fillna(3).astype(int)

    import re

    def to_level(v: Any) -> int:
        s = str(v or "").strip().lower()
        for k, lvl in mapping.items():
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


def _ensure_tiers_and_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has career_stage_level_num and priority_score.
    Uses regional_location (if available) to compute region_tier and priority.
    """
    out = df.copy()
    # career stage numeric
    if "career_stage_level_num" not in out.columns:
        stage_col = next((c for c in [
            "career_stage",
            "career_stage_bucket",
            "experience_bucket",
            "years_of_experience_bucket",
        ] if c in out.columns), None)
        source = out[stage_col] if stage_col else pd.Series([None]*len(out), index=out.index)
        out["career_stage_level_num"] = _infer_career_stage_level(source)

    # region tier and priority
    if "priority_score" not in out.columns:
        # Map regional_location to a coarse tier; unknown -> worst
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
            "Antarctica": 6,
        }
        max_region = max(region_tiers.values()) if region_tiers else 6
        if "regional_location" in out.columns:
            out["region_tier"] = (
                out["regional_location"].map(region_tiers).fillna(max_region).astype(int)
            )
        else:
            out["region_tier"] = max_region
        denom = (max_region - 1) if (max_region - 1) != 0 else 1
        out["region_priority_norm"] = 1 - (out["region_tier"] - 1) / denom

        max_stage = 6
        out["career_stage_norm"] = out["career_stage_level_num"].astype(float) / max_stage
        STAGE_W, REGION_W = 0.7, 0.3
        out["priority_score"] = (
            STAGE_W * out["career_stage_norm"] + REGION_W * out["region_priority_norm"]
        )

    return out


def score_pair(a_row: pd.Series, b_row: pd.Series, weights: ScoreWeights) -> Tuple[float, Dict[str, float]]:
    # preference alignment in both directions
    pref_a_to_b = _safe_cosine(a_row.get("preference_embedding"), b_row.get("profile_embedding"))
    pref_b_to_a = _safe_cosine(b_row.get("preference_embedding"), a_row.get("profile_embedding"))
    mutual_pref = (pref_a_to_b + pref_b_to_a) / 2.0

    profile_sim = _safe_cosine(a_row.get("profile_embedding"), b_row.get("profile_embedding"))

    location = _location_proximity(a_row.get("regional_location"), b_row.get("regional_location"))
    stage = _career_stage_affinity(a_row.get("career_stage_level_num"), b_row.get("career_stage_level_num"))
    role = _role_match_boost(a_row.get("role"), b_row.get("role"))

    score = (
        weights.w_mutual_pref * mutual_pref
        + weights.w_profile_sim * profile_sim
        + weights.w_location * location
        + weights.w_career_stage * stage
        + weights.w_role * role
    )

    return score, {
        "mutual_pref": mutual_pref,
        "profile_sim": profile_sim,
        "location": location,
        "stage": stage,
        "role": role,
    }


def top_k_for_index(df: pd.DataFrame, idx: int, k: int = 15, weights: Optional[ScoreWeights] = None) -> pd.DataFrame:
    """Return top-k candidates for participant at position idx, with component scores."""
    if weights is None:
        weights = ScoreWeights()

    df2 = _ensure_embeddings(df)
    df2 = _ensure_tiers_and_priority(df2)
    if "priority_score" in df2.columns:
        # pre-sort to cut search space, keep 200 best by priority excluding self
        pool = df2.drop(index=idx).sort_values("priority_score", ascending=False).head(max(200, k * 5))
    else:
        pool = df2.drop(index=idx)

    a_row = df2.loc[idx]
    scores: List[Tuple[int, float, Dict[str, float]]] = []
    for j, b_row in pool.iterrows():
        s, comps = score_pair(a_row, b_row, weights)
        scores.append((j, s, comps))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:k]
    recs = []
    for j, s, comps in top:
        row = df2.loc[j].copy()
        row["match_score"] = s
        for k_, v_ in comps.items():
            row[f"score_{k_}"] = v_
        recs.append(row)
    return pd.DataFrame(recs)


def greedy_global_pairing(
    df: pd.DataFrame,
    weights: Optional[ScoreWeights] = None,
    k_pool: int = 30,
    strategy: str = "priority-first",
) -> List[Tuple[int, int, float]]:
    """Produce distinct 1:1 pairs.

    Strategies:
    - "priority-first" (default): sort participants by priority_score desc; for each, pick the
      best available candidate (highest match_score) from its top-k pool, then remove both.
    - "edge": build all top-k edges and take the globally best edges greedily.

    Returns list of tuples (i, j, score) in the order they were assigned.
    """
    if weights is None:
        weights = ScoreWeights()
    df2 = _ensure_embeddings(df)
    df2 = _ensure_tiers_and_priority(df2)
    n = len(df2)

    if strategy == "edge":
        # build candidate edges by top-k around each node
        edges: List[Tuple[int, int, float]] = []
        for i in range(n):
            cand_df = top_k_for_index(df2, i, k=min(k_pool, n - 1), weights=weights)
            for j, row in cand_df.iterrows():
                jj = int(j) if isinstance(j, (int, np.integer)) else df2.index[df2.index.get_loc(j)]
                edges.append((i, jj, float(row["match_score"])) )
        # sort edges by score descending and add greedily if both endpoints unused
        edges.sort(key=lambda t: t[2], reverse=True)
        used = set()
        pairs: List[Tuple[int, int, float]] = []
        for i, j, s in edges:
            if i in used or j in used:
                continue
            pairs.append((i, j, s))
            used.add(i)
            used.add(j)
            if len(used) >= n - (n % 2):
                break
        return pairs

    # priority-first strategy
    order = df2.sort_values("priority_score", ascending=False).index.tolist()
    used = set()
    pairs: List[Tuple[int, int, float]] = []
    for i in order:
        if i in used:
            continue
        cand_df = top_k_for_index(df2, i, k=min(k_pool, n - 1), weights=weights)
        # choose first candidate not used yet
        chosen = None
        for j, row in cand_df.iterrows():
            jj = int(j) if isinstance(j, (int, np.integer)) else df2.index[df2.index.get_loc(j)]
            if jj in used or jj == i:
                continue
            chosen = (i, jj, float(row["match_score"]))
            break
        if chosen is None:
            # fall back: scan all others to find best available
            best_s, best_j = -1.0, None
            a_row = df2.loc[i]
            for jj in df2.index:
                if jj in used or jj == i:
                    continue
                s, _ = score_pair(a_row, df2.loc[jj], weights)
                if s > best_s:
                    best_s, best_j = s, jj
            if best_j is not None:
                chosen = (i, best_j, best_s)
        if chosen is not None:
            i_, j_, s_ = chosen
            pairs.append((i_, j_, s_))
            used.add(i_)
            used.add(j_)
        if len(used) >= n - (n % 2):
            break
    return pairs
