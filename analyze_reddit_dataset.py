"""
Step 2 — Reddit Filtered Dataset Analyzer
Research: "The Human-AI Connection: Sentiment and Ethics of AI Companions"

Reads reddit_submissions_filtered.csv and reddit_comments_filtered.csv,
prints a structured summary report, and saves 100-row samples.

Usage:
    python analyze_reddit_dataset.py
    python analyze_reddit_dataset.py --data-dir /path/to/csvs
"""

import argparse
import os

import pandas as pd

from config import get_data_dir, ensure_output_dirs, SAMPLES_DIR

CHUNK_THRESHOLD  = 500 * 1024 * 1024   # 500 MB — use chunked reading above this
ANALYSIS_CHUNKSIZE = 100_000           # smaller than config.CHUNKSIZE; conservative for in-memory concat

DIVIDER  = "─" * 64
DIVIDER2 = "═" * 64


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame:
    """Load a CSV; uses chunked reading for files over 500 MB."""
    size    = os.path.getsize(path)
    size_mb = size / (1024 ** 2)
    print(f"  Loading {label} ({size_mb:.1f} MB) …", end=" ", flush=True)

    if size > CHUNK_THRESHOLD:
        df = pd.concat(
            pd.read_csv(path, encoding="utf-8-sig", chunksize=ANALYSIS_CHUNKSIZE, low_memory=False),
            ignore_index=True,
        )
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    print(f"done — {len(df):,} rows loaded.")
    return df


def word_count(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).apply(lambda t: len(t.split()))


def missing_pct(series: pd.Series) -> str:
    n_miss = series.isna().sum() + (series.astype(str).str.strip() == "").sum()
    pct    = n_miss / len(series) * 100
    return f"{n_miss:,}  ({pct:.1f}%)"


def convert_utc(df: pd.DataFrame) -> pd.DataFrame:
    col = df["created_utc"]
    if pd.api.types.is_numeric_dtype(col):
        df["created_utc"] = pd.to_datetime(col, unit="s", utc=True)
    else:
        df["created_utc"] = pd.to_datetime(col, utc=True, errors="coerce")
    return df


def subreddit_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["subreddit"].str.lower().value_counts().reset_index()
    counts.columns = ["subreddit", "count"]
    counts["share_%"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def print_section(title: str):
    print(f"\n{DIVIDER2}\n  {title}\n{DIVIDER2}")


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, label: str, text_col: str):
    print_section(f"{label.upper()} ANALYSIS")

    print(f"\n{'Total rows':<30} {len(df):>12,}")
    print(f"{'Unique subreddits':<30} {df['subreddit'].str.lower().nunique():>12,}")
    print(f"{'Unique authors':<30} {df['author'].nunique():>12,}")

    print(f"\n{DIVIDER}\n  Distribution by Subreddit\n{DIVIDER}")
    print(subreddit_table(df).to_string(index=False))

    df = convert_utc(df)
    valid_dates = df["created_utc"].dropna()
    if not valid_dates.empty:
        earliest  = valid_dates.min().strftime("%Y-%m-%d %H:%M UTC")
        latest    = valid_dates.max().strftime("%Y-%m-%d %H:%M UTC")
        span_days = (valid_dates.max() - valid_dates.min()).days
    else:
        earliest = latest = "N/A"
        span_days = 0

    print(f"\n{DIVIDER}\n  Time Range\n{DIVIDER}")
    print(f"  {'Earliest post':<28} {earliest}")
    print(f"  {'Latest post':<28} {latest}")
    print(f"  {'Span':<28} {span_days:,} days")

    print(f"\n{DIVIDER}\n  Text Quality  [{text_col}]\n{DIVIDER}")
    wc = word_count(df[text_col])
    if not wc.empty:
        print(f"  {'Avg word count':<28} {wc.mean():.1f}")
        print(f"  {'Median word count':<28} {wc.median():.1f}")
        print(f"  {'Min word count':<28} {int(wc.min())}")
        print(f"  {'Max word count':<28} {int(wc.max())}")
    else:
        print("  No valid text found.")
    print(f"  {'Missing / blank values':<28} {missing_pct(df[text_col])}")

    if label == "Submissions" and "title" in df.columns:
        title_wc = word_count(df["title"])
        print(f"\n  [title]")
        print(f"  {'Avg word count':<28} {title_wc.mean():.1f}")
        print(f"  {'Missing / blank values':<28} {missing_pct(df['title'])}")

    return df


def main(data_dir: str):
    subs_csv = os.path.join(data_dir, "reddit_submissions_filtered.csv")
    coms_csv = os.path.join(data_dir, "reddit_comments_filtered.csv")

    for path in (subs_csv, coms_csv):
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            print("Run extract_reddit_research.py first.")
            return

    ensure_output_dirs()

    print(DIVIDER2)
    print("  Reddit Dataset Analyzer — AI Companions Research")
    print(DIVIDER2)

    print("\nLoading files …")
    subs = load_csv(subs_csv, "Submissions")
    coms = load_csv(coms_csv, "Comments")

    subs = analyze(subs, "Submissions", "selftext")
    coms = analyze(coms, "Comments",    "body")

    print_section("COMBINED SUMMARY")
    print(f"\n  {'Total records (combined)':<30} {len(subs) + len(coms):>12,}")
    print(f"  {'  · Submissions':<30} {len(subs):>12,}")
    print(f"  {'  · Comments':<30} {len(coms):>12,}")

    sample_subs = os.path.join(SAMPLES_DIR, "sample_submissions.csv")
    sample_coms = os.path.join(SAMPLES_DIR, "sample_comments.csv")

    print(f"\n{DIVIDER}\n  Saving 100-row samples …")
    subs.head(100).to_csv(sample_subs, index=False, encoding="utf-8-sig")
    coms.head(100).to_csv(sample_coms, index=False, encoding="utf-8-sig")
    print(f"  Submissions sample → {sample_subs}")
    print(f"  Comments sample    → {sample_coms}")

    print(f"\n{DIVIDER2}\n  Analysis complete.\n{DIVIDER2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the filtered Reddit CSVs.")
    parser.add_argument(
        "--data-dir", default=get_data_dir(),
        help="Directory containing the filtered CSV files",
    )
    args = parser.parse_args()
    main(args.data_dir)
