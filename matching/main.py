from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print
from rich.table import Table

from .feature_engineering import prepare_feature_columns
from .ingest import FIELD_ALIASES, clean_coffee_csv, resolve_aliases
from .recommender import ScoreWeights, greedy_global_pairing, top_k_for_index


app = typer.Typer(help="MLOps Coffee Match CLI")


def _load_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	return df


@app.command()
def features(
	csv_path: Path = typer.Argument(..., help="Input CSV"),
	out_path: Optional[Path] = typer.Option(None, help="Where to write featured CSV"),
	openai_model: str = typer.Option("gpt-5-mini", help="OpenAI model for location normalization"),
	batch_size: int = typer.Option(20, help="Batch size for embeddings/normalization"),
):
	"""Generate feature columns (embeddings, region tiers, career stage tiers, priority)."""
	df = _load_csv(csv_path)
	feat = prepare_feature_columns(df, model=openai_model, batch_size=batch_size)
	out = out_path or csv_path.with_name(f"{csv_path.stem}_featured.csv")
	feat.to_csv(out, index=False)
	print(f"[green]Wrote features to[/green] {out}")


@app.command()
def clean(
	csv_path: Path = typer.Argument(..., help="Raw CoffeeMatch CSV export"),
	out_path: Optional[Path] = typer.Option(None, help="Where to write cleaned CSV"),
):
	"""Normalize CoffeeMatch survey export into the engine schema."""
	df = clean_coffee_csv(csv_path)
	out = out_path or csv_path.with_name(f"{csv_path.stem}_cleaned.csv")
	df.to_csv(out, index=False)
	print(f"[green]Wrote cleaned data to[/green] {out}")


@app.command()
def process(
	csv_path: Path = typer.Argument(..., help="Raw CoffeeMatch CSV export"),
	pairs_out: Optional[Path] = typer.Option(None, help="Write final pairs CSV to this path"),
	openai_model: str = typer.Option("gpt-5-mini", help="OpenAI model for location normalization"),
	batch_size: int = typer.Option(20, help="Batch size for embeddings/normalization"),
	k_pool: int = typer.Option(30, help="Candidate pool per participant"),
	strategy: str = typer.Option("priority-first", help="Pairing strategy: 'priority-first' or 'edge'"),
	keep_intermediate: bool = typer.Option(False, "--keep-intermediate/--no-keep-intermediate", help="Persist the cleaned CSV to disk"),
):
	"""End-to-end: clean -> features -> pairs for CoffeeMatch CSV."""
	# 1) Clean
	cleaned = clean_coffee_csv(csv_path)
	if keep_intermediate:
		cleaned_path = csv_path.with_name(f"{csv_path.stem}_cleaned.csv")
		cleaned.to_csv(cleaned_path, index=False)
		print(f"[green]Cleaned ->[/green] {cleaned_path}")
	else:
		print("[green]Cleaned in-memory (not written)[/green]")

	# 2) Feature engineering
	featured = prepare_feature_columns(cleaned, model=openai_model, batch_size=batch_size)
	featured_path = csv_path.with_name(f"{csv_path.stem}_featured.csv")
	featured.to_csv(featured_path, index=False)
	print(f"[green]Featured ->[/green] {featured_path}")

	# 3) Pairing
	pairs = greedy_global_pairing(featured, k_pool=k_pool, strategy=strategy)
	rows = []
	alias_map = resolve_aliases(featured)

	def _val(row: pd.Series, key: str) -> str:
		col = alias_map.get(key)
		if col and col in row.index:
			return str(row.get(col) or "")
		for candidate in FIELD_ALIASES.get(key, []):
			if candidate in row.index:
				return str(row.get(candidate) or "")
		return ""

	for i, (a, b, s) in enumerate(pairs, start=1):
		ar = featured.loc[a]
		br = featured.loc[b]
		row_payload = {
			"pair_index": i,
			"a_index": a,
			"b_index": b,
			"match_score": float(s),
			"strategy": strategy,
			"a_name": _val(ar, "name") or f"row_{a}",
			"b_name": _val(br, "name") or f"row_{b}",
			"a_role": _val(ar, "role"),
			"b_role": _val(br, "role"),
			"a_company": _val(ar, "company"),
			"b_company": _val(br, "company"),
			"a_location": _val(ar, "location"),
			"b_location": _val(br, "location"),
			"a_slack_handle": _val(ar, "slack_handle"),
			"b_slack_handle": _val(br, "slack_handle"),
			"a_email": _val(ar, "email"),
			"b_email": _val(br, "email"),
			"a_linkedin": _val(ar, "linkedin"),
			"b_linkedin": _val(br, "linkedin"),
			"a_career_stage": _val(ar, "career_stage"),
			"b_career_stage": _val(br, "career_stage"),
			"a_buddy_preference": _val(ar, "buddy_preference"),
			"b_buddy_preference": _val(br, "buddy_preference"),
			"a_buddy_preferences": _val(ar, "buddy_preferences"),
			"b_buddy_preferences": _val(br, "buddy_preferences"),
		}
		rows.append(row_payload)
	pairs_df = pd.DataFrame(rows)
	pairs_out = pairs_out or csv_path.with_name(f"{csv_path.stem}_pairs.csv")
	pairs_df.to_csv(pairs_out, index=False)
	print(f"[bold]Generated {len(pairs_df)} pairs[/bold] -> {pairs_out}")


@app.command()
def recommend(
	csv_path: Path = typer.Argument(..., help="Featured CSV (or raw; TF-IDF fallback used)"),
	who_index: int = typer.Option(0, help="Row index to recommend for"),
	top_k: int = typer.Option(15, help="Number of candidates to show"),
):
	df = _load_csv(csv_path)
	alias_map = resolve_aliases(df)
	recs = top_k_for_index(df, who_index, k=top_k)
	# Render a small table
	name_col = alias_map.get("name") or "synthetic_name"
	role_col = alias_map.get("role") or "role"
	location_col = alias_map.get("location") or "location"
	cols = [name_col, role_col, location_col, "regional_location", "match_score"]
	table = Table(*cols)
	for _, r in recs.iterrows():
		table.add_row(*(str(r.get(c, "")) for c in cols))
	print(table)


@app.command()
def pair(
	csv_path: Path = typer.Argument(..., help="Featured CSV (or raw; TF-IDF fallback used)"),
	k_pool: int = typer.Option(30, help="Candidate pool size per participant"),
	strategy: str = typer.Option("priority-first", help="Pairing strategy: 'priority-first' or 'edge'"),
	out_path: Optional[Path] = typer.Option(None, help="Write pairs to this CSV"),
):
	df = _load_csv(csv_path)
	alias_map = resolve_aliases(df)
	pairs = greedy_global_pairing(df, k_pool=k_pool, strategy=strategy)
	print(f"[bold]Generated {len(pairs)} pairs[/bold]")

	def _val(row: pd.Series, key: str) -> str:
		col = alias_map.get(key)
		if col and col in row.index:
			return str(row.get(col) or "")
		for candidate in FIELD_ALIASES.get(key, []):
			if candidate in row.index:
				return str(row.get(candidate) or "")
		return ""

	rows = []
	for i, (a, b, s) in enumerate(pairs, start=1):
		a_row = df.loc[a]
		b_row = df.loc[b]
		a_name = _val(a_row, "name") or f"row_{a}"
		b_name = _val(b_row, "name") or f"row_{b}"
		print(f"{i:02d}. {a_name} ↔ {b_name}  (score={s:.3f})")
		row_payload = {
			"pair_index": i,
			"a_index": a,
			"b_index": b,
			"match_score": float(s),
			"strategy": strategy,
			"a_name": a_name,
			"b_name": b_name,
			"a_role": _val(a_row, "role"),
			"b_role": _val(b_row, "role"),
			"a_company": _val(a_row, "company"),
			"b_company": _val(b_row, "company"),
			"a_location": _val(a_row, "location"),
			"b_location": _val(b_row, "location"),
			"a_slack_handle": _val(a_row, "slack_handle"),
			"b_slack_handle": _val(b_row, "slack_handle"),
			"a_email": _val(a_row, "email"),
			"b_email": _val(b_row, "email"),
			"a_linkedin": _val(a_row, "linkedin"),
			"b_linkedin": _val(b_row, "linkedin"),
			"a_career_stage": _val(a_row, "career_stage"),
			"b_career_stage": _val(b_row, "career_stage"),
			"a_buddy_preference": _val(a_row, "buddy_preference"),
			"b_buddy_preference": _val(b_row, "buddy_preference"),
			"a_buddy_preferences": _val(a_row, "buddy_preferences"),
			"b_buddy_preferences": _val(b_row, "buddy_preferences"),
		}
		rows.append(row_payload)
	if out_path:
		out_df = pd.DataFrame(rows)
		out_df.to_csv(out_path, index=False)
		print(f"[green]Saved pairs to[/green] {out_path}")


if __name__ == "__main__":
	app()
