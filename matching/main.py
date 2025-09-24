from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print
from rich.table import Table

from .feature_engineering import prepare_feature_columns
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
def recommend(
	csv_path: Path = typer.Argument(..., help="Featured CSV (or raw; TF-IDF fallback used)"),
	who_index: int = typer.Option(0, help="Row index to recommend for"),
	top_k: int = typer.Option(15, help="Number of candidates to show"),
):
	df = _load_csv(csv_path)
	recs = top_k_for_index(df, who_index, k=top_k)
	# Render a small table
	cols = ["synthetic_name", "role", "location", "regional_location", "match_score"]
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
	pairs = greedy_global_pairing(df, k_pool=k_pool, strategy=strategy)
	print(f"[bold]Generated {len(pairs)} pairs[/bold]")
	rows = []
	for i, (a, b, s) in enumerate(pairs, start=1):
		a_row = df.loc[a]
		b_row = df.loc[b]
		a_name = a_row.get("synthetic_name", f"row_{a}")
		b_name = b_row.get("synthetic_name", f"row_{b}")
		print(f"{i:02d}. {a_name} ↔ {b_name}  (score={s:.3f})")
		rows.append({
			"pair_index": i,
			"a_index": a,
			"a_name": a_name,
			"a_role": a_row.get("role", ""),
			"a_location": a_row.get("location", ""),
			"b_index": b,
			"b_name": b_name,
			"b_role": b_row.get("role", ""),
			"b_location": b_row.get("location", ""),
			"match_score": float(s),
			"strategy": strategy,
		})
	if out_path:
		out_df = pd.DataFrame(rows)
		out_df.to_csv(out_path, index=False)
		print(f"[green]Saved pairs to[/green] {out_path}")


if __name__ == "__main__":
	app()
