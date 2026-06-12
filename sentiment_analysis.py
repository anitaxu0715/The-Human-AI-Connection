"""
Step 3 — Sentiment Analysis & Topic Exploration
Research: "The Human-AI Connection: Sentiment and Ethics of AI Companions"

Pipeline:
  1. Stratified sampling  (all rows from small subreddits; 100k cap on large ones)
  2. VADER sentiment scoring
  3. Comparative analysis per subreddit
  4. TF-IDF key terms for Negative sentiment: r/characterai vs r/ethics
  5. Visualizations (bar chart + per-subreddit histogram grid)
  6. CSV export + top-10 emotionally extreme comments

Note on sampling: the reservoir fills from the earliest records in the CSV
(which is ordered by time). This means large-subreddit samples skew toward
older posts. Treat per-subreddit trends as exploratory, not longitudinally
representative.

Usage:
    python sentiment_analysis.py
    python sentiment_analysis.py --data-dir /path/to/csvs
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    CHUNKSIZE,
    FIGURES_DIR,
    FULL_SAMPLE_SUBREDDITS,
    LARGE_SUB_SAMPLE,
    NEG_THRESHOLD,
    POS_THRESHOLD,
    ensure_output_dirs,
    get_data_dir,
)

nltk.download("vader_lexicon", quiet=True)

DIVIDER  = "─" * 68
DIVIDER2 = "═" * 68


# ── Helpers ───────────────────────────────────────────────────────────────────

def categorize(score: float) -> str:
    if score >  POS_THRESHOLD:
        return "Positive"
    if score <  NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def print_section(title: str):
    print(f"\n{DIVIDER2}\n  {title}\n{DIVIDER2}")


def truncate(text: str, n: int = 220) -> str:
    text = str(text)
    return text[:n] + "…" if len(text) > n else text


# ── Step 1 — Stratified sampling ──────────────────────────────────────────────

def load_stratified_sample(coms_csv: str) -> pd.DataFrame:
    """
    Stream the comments CSV in chunks.
    - FULL_SAMPLE_SUBREDDITS: every row kept.
    - All others: first-arrival reservoir capped at LARGE_SUB_SAMPLE.

    Limitation: because the CSV is likely time-ordered, the reservoir captures
    the oldest posts first. Newer activity in large subreddits is undersampled.
    """
    print_section("STEP 1 — STRATIFIED SAMPLING")
    print(f"  Source: {coms_csv}")
    print(f"  Full-sample subreddits : {', '.join(sorted(FULL_SAMPLE_SUBREDDITS))}")
    print(f"  Large-sub cap          : {LARGE_SUB_SAMPLE:,} rows each")

    full_rows: list[pd.DataFrame] = []
    large_pools: dict[str, list]  = {}

    total_read = 0
    for chunk in pd.read_csv(
        coms_csv, encoding="utf-8-sig", chunksize=CHUNKSIZE, low_memory=False
    ):
        total_read += len(chunk)
        chunk["subreddit_lower"] = chunk["subreddit"].str.lower()

        mask_full = chunk["subreddit_lower"].isin(FULL_SAMPLE_SUBREDDITS)
        if mask_full.any():
            full_rows.append(chunk[mask_full].copy())

        for sub, grp in chunk[~mask_full].groupby("subreddit_lower"):
            pool = large_pools.setdefault(sub, [])
            if len(pool) < LARGE_SUB_SAMPLE:
                rows = grp.to_dict("records")
                pool.extend(rows[: LARGE_SUB_SAMPLE - len(pool)])

        if total_read % (CHUNKSIZE * 5) == 0:
            print(f"    … {total_read:,} rows scanned", flush=True)

    print(f"  Total rows scanned: {total_read:,}")

    parts = []
    if full_rows:
        full_df = pd.concat(full_rows, ignore_index=True)
        parts.append(full_df)
        for sub in FULL_SAMPLE_SUBREDDITS:
            n = (full_df["subreddit_lower"] == sub).sum()
            print(f"    r/{sub:<20}  {n:>8,}  rows  (ALL)")

    for sub, pool in sorted(large_pools.items()):
        df_sub = pd.DataFrame(pool)
        print(f"    r/{sub:<20}  {len(df_sub):>8,}  rows  (sampled)")
        parts.append(df_sub)

    df = pd.concat(parts, ignore_index=True)
    df.drop(columns=["subreddit_lower"], inplace=True, errors="ignore")
    print(f"\n  Final sample size: {len(df):,} rows")
    return df


# ── Step 2 — VADER scoring ────────────────────────────────────────────────────

def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 2 — VADER SENTIMENT SCORING")
    sia    = SentimentIntensityAnalyzer()
    scores = df["body"].fillna("").astype(str).apply(
        lambda t: sia.polarity_scores(t)["compound"]
    )
    df["vader_compound"]  = scores
    df["vader_sentiment"] = scores.apply(categorize)

    dist  = df["vader_sentiment"].value_counts()
    total = len(df)
    for label in ("Positive", "Neutral", "Negative"):
        n = dist.get(label, 0)
        print(f"  {label:<10}  {n:>8,}  ({n/total*100:.1f}%)")

    return df


# ── Step 3 — Comparative analysis ─────────────────────────────────────────────

def comparative_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 3 — COMPARATIVE ANALYSIS BY SUBREDDIT")

    agg = (
        df.groupby("subreddit")
        .agg(
            total_comments = ("vader_compound", "count"),
            avg_compound   = ("vader_compound", "mean"),
            pct_positive   = ("vader_sentiment", lambda s: (s == "Positive").mean() * 100),
            pct_neutral    = ("vader_sentiment", lambda s: (s == "Neutral").mean()  * 100),
            pct_negative   = ("vader_sentiment", lambda s: (s == "Negative").mean() * 100),
        )
        .reset_index()
        .sort_values("avg_compound", ascending=False)
    )
    for col in ("avg_compound", "pct_positive", "pct_neutral", "pct_negative"):
        decimals = 4 if col == "avg_compound" else 1
        agg[col] = agg[col].round(decimals)

    print(f"\n  {'Subreddit':<22} {'N':>8}  {'Avg':>7}  {'Pos%':>6}  {'Neu%':>6}  {'Neg%':>6}")
    print(f"  {DIVIDER}")
    for _, row in agg.iterrows():
        print(
            f"  r/{row['subreddit']:<20} {int(row['total_comments']):>8,}"
            f"  {row['avg_compound']:>7.4f}"
            f"  {row['pct_positive']:>6.1f}"
            f"  {row['pct_neutral']:>6.1f}"
            f"  {row['pct_negative']:>6.1f}"
        )

    print(f"\n  Most Positive: r/{agg.iloc[0]['subreddit']}  (avg={agg.iloc[0]['avg_compound']:.4f})")
    print(f"  Most Negative: r/{agg.iloc[-1]['subreddit']}  (avg={agg.iloc[-1]['avg_compound']:.4f})")
    return agg


# ── Step 4 — TF-IDF key terms ─────────────────────────────────────────────────

def tfidf_key_terms(df: pd.DataFrame, sub_a: str = "characterai", sub_b: str = "ethics"):
    print_section(f"STEP 4 — TF-IDF KEY TERMS  (Negative sentiment)")
    print(f"  Comparing: r/{sub_a}  vs  r/{sub_b}")

    results = {}
    for sub in (sub_a, sub_b):
        mask  = (df["subreddit"].str.lower() == sub.lower()) & (df["vader_sentiment"] == "Negative")
        texts = df.loc[mask, "body"].fillna("").astype(str).tolist()
        if len(texts) < 5:
            print(f"  r/{sub}: too few negative comments ({len(texts)}) — skipping.")
            results[sub] = []
            continue

        vec = TfidfVectorizer(
            max_features=5_000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        X          = vec.fit_transform(texts)
        mean_tfidf = X.mean(axis=0).A1
        top_idx    = mean_tfidf.argsort()[::-1][:15]
        terms      = [(vec.get_feature_names_out()[i], round(float(mean_tfidf[i]), 5))
                      for i in top_idx]
        results[sub] = terms

        print(f"\n  r/{sub}  ({len(texts):,} negative comments)")
        print(f"  {'Rank':<5}  {'Term':<25}  {'TF-IDF'}")
        print(f"  {DIVIDER}")
        for rank, (term, score) in enumerate(terms, 1):
            print(f"  {rank:<5}  {term:<25}  {score:.5f}")

    return results


# ── Step 5 — Visualizations ───────────────────────────────────────────────────

def make_bar_chart(agg: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#4CAF50" if v >= 0 else "#F44336" for v in agg["avg_compound"]]
    bars    = ax.barh(agg["subreddit"], agg["avg_compound"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average VADER Compound Score")
    ax.set_title("Average Sentiment Score by Subreddit\n(AI Companions Research Dataset)", fontsize=13)
    ax.invert_yaxis()
    for bar, val in zip(bars, agg["avg_compound"]):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def make_hist(df: pd.DataFrame, out_path: str):
    subreddits = sorted(df["subreddit"].unique())
    n_plots    = min(len(subreddits), 10)
    fig, axes  = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    axes       = axes.flatten()

    for i, sub in enumerate(subreddits[:n_plots]):
        ax     = axes[i]
        sub_df = df[df["subreddit"].str.lower() == sub.lower()]
        ax.hist(sub_df["vader_compound"], bins=40, color="#5C6BC0", edgecolor="white", alpha=0.85)
        ax.axvline(0,     color="black",   linewidth=0.7, linestyle="--")
        ax.axvline( 0.05, color="#4CAF50", linewidth=0.7, linestyle=":")
        ax.axvline(-0.05, color="#F44336", linewidth=0.7, linestyle=":")
        ax.set_title(f"r/{sub}", fontsize=9)
        ax.set_xlabel("Compound", fontsize=7)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("VADER Compound Score Distribution by Subreddit", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Step 6 — Export & extremes ────────────────────────────────────────────────

def export_and_extremes(df: pd.DataFrame, out_csv: str):
    print_section("STEP 6 — EXPORT & TOP-10 EMOTIONALLY EXTREME COMMENTS")

    export_cols = ["id", "subreddit", "author", "created_utc",
                   "score", "body", "vader_compound", "vader_sentiment"]
    df[export_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  Results saved → {out_csv}")

    top10 = (
        df.assign(abs_compound=df["vader_compound"].abs())
        .nlargest(10, "abs_compound")
        [["subreddit", "vader_compound", "vader_sentiment", "body"]]
        .reset_index(drop=True)
    )

    print(f"\n  Top 10 Most Emotionally Extreme Comments\n  {DIVIDER}")
    for i, row in top10.iterrows():
        print(f"\n  [{i+1:>2}] r/{row['subreddit']:<18}  score={row['vader_compound']:+.4f}  [{row['vader_sentiment']}]")
        print(f"       {truncate(row['body'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str):
    coms_csv = os.path.join(data_dir, "reddit_comments_filtered.csv")
    if not os.path.exists(coms_csv):
        print(f"ERROR: File not found: {coms_csv}")
        print("Run extract_reddit_research.py first.")
        return

    ensure_output_dirs()

    out_csv    = os.path.join(data_dir, "sentiment_summary_results.csv")
    chart_bar  = os.path.join(FIGURES_DIR, "sentiment_by_subreddit.png")
    chart_hist = os.path.join(FIGURES_DIR, "sentiment_distribution.png")

    print(DIVIDER2)
    print("  Reddit Sentiment Analysis — AI Companions Research")
    print(DIVIDER2)

    df  = load_stratified_sample(coms_csv)
    df  = run_vader(df)
    agg = comparative_analysis(df)
    tfidf_key_terms(df, sub_a="characterai", sub_b="ethics")

    print_section("STEP 5 — VISUALIZATIONS")
    make_bar_chart(agg, chart_bar)
    make_hist(df, chart_hist)

    export_and_extremes(df, out_csv)

    print(f"\n{DIVIDER2}\n  All done.\n{DIVIDER2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment analysis on filtered Reddit comments.")
    parser.add_argument(
        "--data-dir", default=get_data_dir(),
        help="Directory containing reddit_comments_filtered.csv",
    )
    args = parser.parse_args()
    main(args.data_dir)
