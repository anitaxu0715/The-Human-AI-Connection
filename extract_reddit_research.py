"""
Step 1 — Reddit .zst Archive Extractor
Research: "The Human-AI Connection: Sentiment and Ethics of AI Companions"

Streams and decompresses .zst files in-memory (no disk extraction).
Outputs two CSVs: reddit_submissions_filtered.csv, reddit_comments_filtered.csv

Usage:
    python extract_reddit_research.py
    python extract_reddit_research.py --data-dir /path/to/zst/files
"""

import argparse
import csv
import json
import os

import zstandard as zstd
from tqdm import tqdm

from config import (
    TARGET_SUBREDDITS,
    WRITE_BUFFER,
    get_data_dir,
)

SUBMISSION_FIELDS = ["id", "subreddit", "author", "created_utc", "score", "title", "selftext"]
COMMENT_FIELDS    = ["id", "subreddit", "author", "created_utc", "score", "body"]
DELETED           = {"[deleted]", "[removed]"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


def is_deleted(text: str) -> bool:
    return text.strip() in DELETED


def open_csv_writer(path: str, fieldnames: list):
    """Open a CSV in utf-8-sig mode (BOM for Excel compatibility)."""
    fh = open(path, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def stream_zst_lines(filepath: str):
    """
    Generator: yield decoded text lines from a .zst file.
    Uses a 16 MB read buffer — nothing is written to disk.
    """
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh, read_size=2**24) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**24)
                if not chunk:
                    if buffer:
                        yield buffer.decode("utf-8", errors="replace")
                    break
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    if line:
                        yield line.decode("utf-8", errors="replace")


def classify_file(filename: str):
    """Return 'submission', 'comment', or None based on Pushshift naming convention."""
    name = filename.upper()
    if name.startswith("RS_"):
        return "submission"
    if name.startswith("RC_"):
        return "comment"
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main(data_dir: str):
    submissions_csv = os.path.join(data_dir, "reddit_submissions_filtered.csv")
    comments_csv    = os.path.join(data_dir, "reddit_comments_filtered.csv")

    zst_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".zst"))
    if not zst_files:
        print(f"No .zst files found in {data_dir}")
        return

    print(f"Found {len(zst_files)} .zst file(s) in {data_dir}")
    print(f"Filtering for subreddits: {', '.join(sorted(TARGET_SUBREDDITS))}\n")

    sub_fh, sub_writer = open_csv_writer(submissions_csv, SUBMISSION_FIELDS)
    com_fh, com_writer = open_csv_writer(comments_csv,    COMMENT_FIELDS)

    sub_buf, com_buf = [], []
    total_subs = total_coms = total_skipped = 0

    def flush_buffer(writer, buf):
        writer.writerows(buf)
        buf.clear()

    try:
        for filename in zst_files:
            filepath  = os.path.join(data_dir, filename)
            file_size = os.path.getsize(filepath)
            file_type = classify_file(filename)

            pbar = tqdm(
                total=file_size,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"{filename[:40]:<40}", dynamic_ncols=True,
            )

            for raw_line in stream_zst_lines(filepath):
                pbar.update(len(raw_line.encode("utf-8")) + 1)

                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    total_skipped += 1
                    continue

                subreddit = obj.get("subreddit", "").lower()
                if subreddit not in TARGET_SUBREDDITS:
                    continue

                kind = file_type or ("submission" if "title" in obj else "comment")
                author      = obj.get("author", "")
                created_utc = obj.get("created_utc", "")
                score       = obj.get("score", "")
                rec_id      = obj.get("id", "")

                if kind == "submission":
                    title    = clean_text(obj.get("title",    ""))
                    selftext = clean_text(obj.get("selftext", ""))
                    if is_deleted(title) or is_deleted(selftext):
                        total_skipped += 1
                        continue
                    sub_buf.append({
                        "id": rec_id, "subreddit": obj.get("subreddit", ""),
                        "author": author, "created_utc": created_utc,
                        "score": score, "title": title, "selftext": selftext,
                    })
                    total_subs += 1
                    if len(sub_buf) >= WRITE_BUFFER:
                        flush_buffer(sub_writer, sub_buf)
                else:
                    body = clean_text(obj.get("body", ""))
                    if is_deleted(body):
                        total_skipped += 1
                        continue
                    com_buf.append({
                        "id": rec_id, "subreddit": obj.get("subreddit", ""),
                        "author": author, "created_utc": created_utc,
                        "score": score, "body": body,
                    })
                    total_coms += 1
                    if len(com_buf) >= WRITE_BUFFER:
                        flush_buffer(com_writer, com_buf)

            pbar.close()

        if sub_buf:
            flush_buffer(sub_writer, sub_buf)
        if com_buf:
            flush_buffer(com_writer, com_buf)

    finally:
        sub_fh.close()
        com_fh.close()

    print("\n── Extraction complete ──────────────────────────────────────────")
    print(f"  Submissions saved : {total_subs:>10,}  →  {submissions_csv}")
    print(f"  Comments saved    : {total_coms:>10,}  →  {comments_csv}")
    print(f"  Skipped (deleted/removed/bad JSON): {total_skipped:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Reddit .zst archives by subreddit.")
    parser.add_argument(
        "--data-dir", default=get_data_dir(),
        help="Directory containing .zst files (default: config.DEFAULT_DATA_DIR or DATA_DIR env var)",
    )
    args = parser.parse_args()
    main(args.data_dir)
