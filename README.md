# MLOps Virtual Coffee 1:1 Matching Recommendation System

## Project Overview

The MLOps Community regularly organizes **Virtual Coffee 1:1s** and similar buddy programs. Currently, member matching is done manually, which is time-consuming and unsustainable as participation grows. This project explores building an **AI-powered recommendation system** that can automate matching while preserving fairness, quality, and diversity.

## Problem Statement

Organizers spend hours manually pairing members based on survey responses. With \~80–100+ entries per round, this process becomes inefficient and error-prone. The key challenge is to recommend matches that:

* Reflect participants’ stated preferences and interests.
* Ensure broad coverage so no one is left with low-quality matches.
* Scale smoothly as community size grows.

## Synthetic Dataset Plan

Because real survey data contains personal information, we will use a **synthetic dataset** in the public repo:

* **Structure**: Mirror the form fields (role, career stage, years of experience, skills, summary, buddy preferences, participation flags).
* **Names**: Use fabricated but realistic names (not identifiers like `participant_042`).
* **Free-text fields**: Paraphrased by an LLM to look natural but not personally identifiable.
* **Excluded fields**: Slack handles, LinkedIn URLs, company names, or emails.
* **Distribution**: Sampling without replacement ×2 over the real data for representativeness, then LLM-based variation.
* **Noise injection**: Controlled variation in location buckets, experience levels, and role titles to avoid one-to-one correspondence.

## Approach

The system can be thought of in four main steps:

### 1. Exploratory Data Analysis (EDA) & Preprocessing

* Inspect distributions of roles, experience, locations, and text lengths.
* Normalize categorical fields (career stage, role taxonomy, regions).
* Clean and standardize free-text (skills, preferences, summaries).
* Handle missing or inconsistent entries.

### 2. Feature Engineering

* **Use current survey structure as inputs**:

  * Role (categorical dropdown: ML Engineer, ML Scientist/Researcher, Software Engineer, Data Scientist, Founder, Management, Sales and Marketing, Data Engineer, Student, Other).
  * Career stage (dropdown: Undergrad/New Grad, Graduate Student, 1–3 Years, 3–5 Years, 5–10 Years, 10+ Years).
  * Buddy preference (dropdown: No preference, or Buddy in a similar role).
  * Free‑text fields: skills, summaries, detailed buddy preferences.
* **Feature ideas**:

  * One‑hot or embedding encodings for dropdown fields.
  * Semantic embeddings for free‑text inputs (skills, summaries, buddy preferences).
  * Possible weighting/flagging for hard vs soft constraints (e.g., “must be similar role” vs. open).
* Combine these into participant vectors that blend categorical structure with semantic content.

### 3. Recommendation Scoring

* Compute similarity scores between participants using embeddings + categorical alignment.
* Apply heuristic weighting (e.g., preferences > skills > role > experience > location).
* Boost matches that satisfy expressed buddy preferences.

### 4. Assignment Optimization

* Build a pairwise score matrix of potential matches.
* Solve for **distinct 1:1 pairings** using optimization methods:

  * Maximum-weight perfect matching (graph-based).
  * ILP formulation for flexible constraints.
* Apply fairness/diversity controls to prevent “popular” participants from dominating matches:

  * Popularity penalties.
  * Mixing rules (e.g., career stage diversity).
  * Avoiding repeat pairs across rounds.

## Assignment & Fairness (Distinct 1:1 Pairing) — High-Level

**Goal**: Convert pairwise scores into a global set of distinct 1:1 matches that balances quality and fairness, while also allowing for **backup recommendations** in case some participants don’t show.

**Core idea**: Build a score matrix and solve a global pairing problem, then supplement with alternative matches.

* **Primary match assignment**: Solve for a maximum‑weight perfect matching, small ILP, or greedy+repair to assign each participant a distinct buddy.
* **Backup/secondary matches**: For each participant, store 1–2 additional candidates ranked by score. These serve as alternates if the primary pair falls through.
* **Fairness levers** (activate as needed):

  * Popularity moderation so universally high‑scoring profiles don’t dominate.
  * Diversity nudges (career‑stage mixing, role complementarity) as soft constraints.
  * Hard vs. soft preference handling; avoidance of repeat pairs across rounds.
* **Odd counts & guardrails**:

  * Handle odd participant counts with a dummy “bye” or occasional triads when approved.
  * Minimum quality thresholds with graceful fallback if infeasible.

### Light‑Touch Evaluation

Track a few simple indicators to guide iteration (not to over‑fit):

* Median/quantile match score.
* % of participants with key preferences satisfied.
* Exposure of “popular” profiles (how often they appear in final matches vs. candidate lists).
* Experience/role balance across pairs.
* **Backup utility**: how often alternates are used, and how their scores compare to primaries.

## Current Implementation (WIP)

High‑level flow for synthetic data generation and inspection:

- CSV → DataFrame → sample without replacement → LLM generation → structured validation
- Scripts under `synthetic_generation/`:
  - `gen_synthetic_members.py`: CLI to read, sample, and optionally generate via OpenAI
  - `synth_models.py`: Pydantic models for structured outputs (input/output alignment)
- Scripts under `matching/`:
  - `feature_engineering.py`: Prepares data by creating embeddings and normalizing locations.
  - `matcher.py`: Contains the core logic for pairing participants based on engineered features.
  - `data_models.py`: Defines Pydantic models for participants and matches.

What it does now:
- Reads `data/private_mlops_marchvc.csv` into a DataFrame
- Applies a canonical rename mapping to pythonic keys (toggle with `--no-apply-rename`)
- Samples N rows without replacement
- Builds example records for the LLM with PII excluded
- Calls OpenAI (optional) and parses structured outputs into Pydantic models
- Prints sampled + generated, can also write generated rows to CSV

PII excluded from LLM: `email`, `slack_handle`, `linkedin_url`, `company`, `name`.

Example renames (subset):
- `"Unnamed: 0" → "submission_id"`
- `"Respondent ID" → "respondent_id"`
- `"Which option best represents your role?" → "role"`
- `"How would you characterize your current career stage?" → "career_stage"`
- `"Where are you based?" → "location"`
- `"Tweet-sized summary of yourself" → "summary"`
- `"What are your skills? In which fields do you specialize?" → "skills"`
- `"Describe what you want your buddy to be like." → "buddy_preferences"`

### CLI usage

Run as a module (ensures package imports):

```bash
uv run python -m synthetic_generation.gen_synthetic_members --n 5 --seed 42 --compare
```

Generate via OpenAI (structured output) and print:

```bash
uv run python -m synthetic_generation.gen_synthetic_members \
  --n 5 \
  --compare \
  --use-openai \
  --openai-model gpt-5-mini
```

Write generated rows to CSV under `data/`:

```bash
uv run python -m synthetic_generation.gen_synthetic_members \
  --n 5 --compare --use-openai \
  --out-csv data/synthetic_generated.csv
```

Flags:
- `--apply-rename/--no-apply-rename`: toggle canonical header renames (default on)
- `--structured-output/--no-structured-output`: toggle OpenAI structured parsing (default on)

### Environment

- Python 3.12+
- Managed with `uv` (dependencies in `pyproject.toml`)

Setup:

```bash
uv sync
source .venv/bin/activate
export OPENAI_API_KEY=...  # or place in .env
```
