#!/usr/bin/env python3
"""
Top-k binder design selection by worst weighted metric rank.

Implements Algorithm 2 from BoltzGen paper:
  For each design, compute its rank on each metric, divide by the metric's
  weight, then score the design by its worst (max) weighted rank.
  Select the k designs with the smallest such score.
  (see https://hannes-stark.com/assets/boltzgen.pdf)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Paper-defined metric configs and weights (see algorithm 2)
# direction: "max" = higher is better, "min" = lower is better

METRIC_DEFS = {
    "design_iiptm":                {"direction": "max"},
    "design_ptm":                  {"direction": "max"},
    "min_design_to_target_pae":    {"direction": "min"},
    "plip_hbonds_refolded":        {"direction": "max"},
    "plip_saltbridge_refolded":    {"direction": "max"},
    "delta_sasa_refolded":         {"direction": "max"},
}

PROTEIN_WEIGHTS = {
    "design_iiptm":                1.0,
    "design_ptm":                  2.0,
    "min_design_to_target_pae":    1.0,
    "plip_hbonds_refolded":        2.0,
    "plip_saltbridge_refolded":    2.0,
    "delta_sasa_refolded":         2.0,
}


def rank_binders(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    k: int = 50,
    max_liability_violations: int | None = None,
    max_liability_score: float | None = None,
) -> pd.DataFrame:
    """
    Rank binder designs using the minmax weighted rank algorithm.

    Parameters
    ----------
    df : DataFrame
        Must contain the 6 metric columns. May also contain
        liability_score and liability_num_violations.
    weights : dict
        Metric name -> weight. Defaults to PROTEIN_WEIGHTS.
    k : int
        Number of top designs to return.
    max_liability_violations : int or None
        If set, pre-filter designs exceeding this violation count.
    max_liability_score : float or None
        If set, pre-filter designs exceeding this liability score.

    Returns
    -------
    DataFrame sorted by algorithm score (best first), with rank columns.
    """
    if weights is None:
        weights = PROTEIN_WEIGHTS

    work = df.copy()

    # pre-filter on liability
    n_before = len(work)
    if max_liability_violations is not None and "liability_num_violations" in work.columns:
        work = work[work["liability_num_violations"] <= max_liability_violations]
    if max_liability_score is not None and "liability_score" in work.columns:
        work = work[work["liability_score"] <= max_liability_score]
    n_after = len(work)
    if n_after < n_before:
        print(f"Liability filter: {n_before} → {n_after} designs ({n_before - n_after} removed)")

    if n_after == 0:
        print("ERROR: No designs survived liability filtering.", file=sys.stderr)
        bt.logging.warning("No designs survived liability filtering.")
        return pd.DataFrame()

    # validate required columns
    missing = [m for m in weights if m not in work.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
        bt.logging.warning(f"Missing columns: {missing}")
        return pd.DataFrame()

    # compute ranks (1 = best)
    rank_cols = {}
    for metric, w in weights.items():
        ascending = METRIC_DEFS[metric]["direction"] == "min"
        raw_rank = work[metric].rank(ascending=ascending, method="average")
        weighted_rank = raw_rank / w
        col_name = f"wrank_{metric}"
        rank_cols[col_name] = weighted_rank

    rank_df = pd.DataFrame(rank_cols, index=work.index)

    # score = worst (max) weighted rank across metrics
    work["algo2_score"] = rank_df.max(axis=1)
    work["bottleneck_metric"] = rank_df.idxmax(axis=1).str.replace("wrank_", "", regex=False)

    # attach individual weighted ranks for inspection
    for col in rank_df.columns:
        work[col] = rank_df[col]

    # select top-k
    work = work.sort_values("algo2_score", ascending=True).reset_index(drop=True)
    work.index.name = "selection_rank"
    work.index += 1  # 1-based

    top_k = work.head(k)

    return top_k

