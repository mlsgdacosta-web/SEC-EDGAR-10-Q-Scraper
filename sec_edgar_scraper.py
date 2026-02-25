#!/usr/bin/env python3
"""
SEC EDGAR 10-Q Scraper
Author: Maria Luiza Sena Gomes da Costa
Date: December 29 2025

Scrape 10-Q filings from SEC EDGAR for the provided CIK list and extract:
- CIK
- Filing date of the form
- Word count of the form
- Sentence count of the form
- EPS diluted (current quarter and previous quarter comparator)

Assumptions:
1) If multiple 10-Q filings exist for a CIK in 2020 Q2, we keep the most recent by Date Filed.
2) Previous quarter EPS in a 10-Q is interpreted as the comparative quarter in the prior year
   Therefore, the two most recent periods found for us-gaap:EarningsPerShareDiluted are selected.

Run:
  python sec_edgar_scraper.py --cik-file CIK_list.txt --index-zip "EDGAR Index 2020 Q2.zip" --out edgar_scrape_output.csv

Notes:
- SEC requires a descriptive User-Agent. Use your email in --user-agent.


Setup 
- Use the SEC's Q2 2020 master form index (form.idx) located in the folder/zip "EDGAR Index 2020 Q2"
  to locate Form 10-Q filings and their archive paths (e.g., edgar/data/<CIK>/<accession>.txt).
- For the 50 CIKs provided in the text file CIK_list, scrape the corresponding Form 10-Qs directly
  from SEC EDGAR via:
    https://www.sec.gov/Archives/<path-from-index>
- From the scraped filings, extract and present:
  * CIK
  * Filing date of the form
  * Word count of the form
  * Sentence count of the form
  * Two quarterly EPS values (current quarter and previous quarter comparator), typically tagged as
    us-gaap:EarningsPerShareDiluted in the filing's embedded XBRL.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/"
DEFAULT_USER_AGENT = "Malu Sena Gomes da Costa (mlsgdacosta@gmail.com)"

# -----------------------------
# Parsing the index and choosing filing
# -----------------------------

@dataclass(frozen=True)
class IndexRow:
    form_type: str
    company_name: str
    cik: int
    date_filed: str  # YYYY-MM-DD
    filename: str    # edgar/data/.../*.txt

def read_cik_list(cik_file: Path) -> List[int]:
    """
    Reading the provided CIK_list.txt (one CIK per line).
    Returns list of ints.
    """
    ciks: List[int] = []
    for line in cik_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.lower() == "cik":
            continue
        ciks.append(int(line))
    return ciks

def parse_form_idx_from_zip(index_zip: Path, member_name: str = "form.idx") -> List[IndexRow]:
    """
    Parsing the SEC master index file (form.idx).
    File uses fixed-width columns; infer the slice points from the header line.
    """
    with zipfile.ZipFile(index_zip, "r") as zf:
        raw = zf.read(member_name).decode("latin1")

    lines = raw.splitlines()

    # Finding the header line that starts with "Form Type"
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Form Type"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header line starting with 'Form Type' in form.idx")

    header = lines[header_idx]

    # Column start positions from the header itself (prevents error from spacing changes)
    starts = {
        "form": header.index("Form Type"),
        "company": header.index("Company Name"),
        "cik": header.index("CIK"),
        "date": header.index("Date Filed"),
        "file": header.index("File Name"),
    }

    # Data begins after the dashed separator line 
    data_start = header_idx + 2

    rows: List[IndexRow] = []
    for line in lines[data_start:]:
        if not line.strip():
            continue

        form_type = line[starts["form"]:starts["company"]].strip()
        company_name = line[starts["company"]:starts["cik"]].strip()
        cik_str = line[starts["cik"]:starts["date"]].strip()
        date_filed = line[starts["date"]:starts["file"]].strip()
        filename = line[starts["file"]:].strip()

        # Skipping malformed rows
        if not cik_str.isdigit():
            continue

        rows.append(
            IndexRow(
                form_type=form_type,
                company_name=company_name,
                cik=int(cik_str),
                date_filed=date_filed,
                filename=filename,
            )
        )
    return rows

def pick_latest_10q_per_cik(index_rows: List[IndexRow], target_ciks: Iterable[int]) -> Dict[int, IndexRow]:
    """
    Filters to 10-Q rows for the target CIKs and picks the most recent by Date Filed.
    """
    target = set(target_ciks)
    per_cik: Dict[int, IndexRow] = {}

    for r in index_rows:
        if r.cik not in target:
            continue
        if r.form_type != "10-Q":
            continue

        if r.cik not in per_cik:
            per_cik[r.cik] = r
        else:
            # Keeping the latest by date_filed
            if r.date_filed > per_cik[r.cik].date_filed:
                per_cik[r.cik] = r

    return per_cik

# -----------------------------
# Scraping + cleaning for text counts
# -----------------------------

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[’'-][A-Za-z]+)*")
NUMERIC_TOKEN_RE = re.compile(r"^[\d\.,\-\(\)%$]+$")  # tokens that are numbers/symbols
SENTENCE_END_RE = re.compile(
    r"""
    (?<!\d)          # not preceded by a digit (avoids decimals)
    [.!?]+           # sentence-ending punctuation
    \s+              # followed by whitespace
    (?=[A-Z])        # next sentence starts with a capital letter
    """,
    re.VERBOSE,
)

def clean_text_for_counts(raw: str) -> str:
    """
    Converting a raw EDGAR filing text into mostly plain text for word/sentence counting.
    """
    # Removing embedded <SCRIPT>...</SCRIPT> and <STYLE>...</STYLE> blocks
    raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)

    # Stripping all tags
    no_tags = TAG_RE.sub(" ", raw)

    # Unescaping HTML entities
    try:
        import html as _html
        no_tags = _html.unescape(no_tags)
    except Exception:
        pass

    # Normalizing whitespace
    return WHITESPACE_RE.sub(" ", no_tags).strip()

TOKEN_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")  # strip leading/trailing punctuation

def word_count(text: str) -> int:
    if not text:
        return 0

    tokens = text.split()
    count = 0

    for tok in tokens:
        tok = TOKEN_STRIP_RE.sub("", tok)  # removing punctuation at ends
        if not tok:
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        # Ignore single-letter tokens (A, Q, etc.) which appear a lot in filings
        if len(tok) == 1:
            continue
        count += 1

    return count

def sentence_count(text: str) -> int:
    if not text:
        return 0

    sentences = SENTENCE_END_RE.split(text)

    # Filter out very short fragments that are unlikely to be real sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return len(sentences)

# -----------------------------
# EPS (XBRL) extraction
# -----------------------------

FACT_RE = re.compile(
    r"<us-gaap:EarningsPerShareDiluted\b[^>]*?contextRef=\"(?P<context>[^\"]+)\"[^>]*?>\s*(?P<val>-?\d+(\.\d+)?)\s*</us-gaap:EarningsPerShareDiluted>",
    re.IGNORECASE | re.DOTALL,
)

CONTEXT_RE = re.compile(
    r"<xbrli:context\b[^>]*?id=\"(?P<id>[^\"]+)\"[^>]*>(?P<body>.*?)</xbrli:context>",
    re.IGNORECASE | re.DOTALL,
)

PERIOD_RE = re.compile(
    r"<xbrli:period>(?P<body>.*?)</xbrli:period>",
    re.IGNORECASE | re.DOTALL,
)

START_RE = re.compile(r"<xbrli:startDate>\s*(?P<d>\d{4}-\d{2}-\d{2})\s*</xbrli:startDate>", re.IGNORECASE)
END_RE = re.compile(r"<xbrli:endDate>\s*(?P<d>\d{4}-\d{2}-\d{2})\s*</xbrli:endDate>", re.IGNORECASE)

def parse_context_periods(xbrl: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Returns mapping: context_id -> (startDate, endDate)
    for duration contexts found in the XBRL case.
    """
    ctx_periods: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    for m in CONTEXT_RE.finditer(xbrl):
        ctx_id = m.group("id")
        body = m.group("body")

        p = PERIOD_RE.search(body)
        if not p:
            continue
        pbody = p.group("body")

        start = START_RE.search(pbody)
        end = END_RE.search(pbody)
        start_d = start.group("d") if start else None
        end_d = end.group("d") if end else None

        ctx_periods[ctx_id] = (start_d, end_d)

    return ctx_periods

def is_quarter_length(start: str, end: str) -> bool:
    """
    Checking if a period is ~a quarter (75-110 days).
    """
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
        days = (e - s).days
        return 75 <= days <= 110
    except Exception:
        return False

def extract_eps_diluted_quarters(raw_filing: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts diluted EPS values for the two most recent quarter-length periods.
    Returns (current_quarter_eps, previous_comparator_eps).

    Implementation detail:
    - Finds XBRL facts for us-gaap:EarningsPerShareDiluted, keeps those that are quarter-length,
      then picks the two most recent by endDate. If multiple facts exist for the same endDate,
      uses the median value (prevents duplicates).
    """
    xbrl_text = raw_filing

    facts = [(m.group("context"), float(m.group("val"))) for m in FACT_RE.finditer(xbrl_text)]
    if not facts:
        return (None, None)

    ctx_periods = parse_context_periods(xbrl_text)

    # Grouping values by endDate for quarter-length periods
    enddate_to_vals: Dict[str, List[float]] = {}
    for ctx, val in facts:
        period = ctx_periods.get(ctx)
        if not period:
            continue
        start, end = period
        if not start or not end:
            continue
        if not is_quarter_length(start, end):
            continue

        enddate_to_vals.setdefault(end, []).append(val)

    if not enddate_to_vals:
        return (None, None)

    # Sorting endDates descending and take the top 2
    enddates_sorted = sorted(enddate_to_vals.keys(), reverse=True)
    top = enddates_sorted[:2]

    def median(xs: List[float]) -> float:
        xs2 = sorted(xs)
        n = len(xs2)
        if n % 2 == 1:
            return xs2[n // 2]
        return (xs2[n // 2 - 1] + xs2[n // 2]) / 2.0

    current = median(enddate_to_vals[top[0]]) if len(top) >= 1 else None
    prev = median(enddate_to_vals[top[1]]) if len(top) >= 2 else None
    return (current, prev)

# -----------------------------
# SEC download + producing CSV
# -----------------------------

def fetch_filing_text(url: str, user_agent: str, timeout: int = 60) -> str:
    """
    Downloading the filing text from SEC Archives.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def process_one_filing(cik: int, date_filed: str, filename: str, user_agent: str) -> dict:
    """
    Downloads + extracts required fields for one filing.
    """
    url = SEC_ARCHIVES_BASE + filename
    raw = fetch_filing_text(url, user_agent=user_agent)

    cleaned = clean_text_for_counts(raw)

    wc = word_count(cleaned)
    sc = sentence_count(cleaned)
    eps_curr, eps_prev = extract_eps_diluted_quarters(raw)

    return {
        "cik": cik,
        "filing_date": date_filed,
        "word_count": wc,
        "sentence_count": sc,
        "eps_diluted_current_q": eps_curr,
        "eps_diluted_prev_q": eps_prev,
        "filing_url": url,
    }

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cik-file", type=Path, required=True, help="Path to CIK_list.txt")
    p.add_argument("--index-zip", type=Path, required=True, help="Path to 'EDGAR Index 2020 Q2.zip'")
    p.add_argument("--out", type=Path, required=True, help="Output CSV file path")
    p.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="Descriptive User-Agent (include email)")
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between SEC requests")
    args = p.parse_args(argv)

    ciks = read_cik_list(args.cik_file)

    index_rows = parse_form_idx_from_zip(args.index_zip)
    latest = pick_latest_10q_per_cik(index_rows, ciks)

    missing = sorted(set(ciks) - set(latest.keys()))
    if missing:
        print(f"WARNING: No 10-Q found in Q2 2020 index for {len(missing)} CIKs: {missing[:10]}{'...' if len(missing)>10 else ''}", file=sys.stderr)

    out_rows: List[dict] = []

    # Deterministic order
    for cik in sorted(latest.keys()):
        row = latest[cik]
        try:
            out_rows.append(process_one_filing(cik=row.cik, date_filed=row.date_filed, filename=row.filename, user_agent=args.user_agent))
        except requests.HTTPError as e:
            print(f"ERROR: HTTP error for CIK {cik} ({row.filename}): {e}", file=sys.stderr)
            out_rows.append({
                "cik": row.cik,
                "filing_date": row.date_filed,
                "word_count": None,
                "sentence_count": None,
                "eps_diluted_current_q": None,
                "eps_diluted_prev_q": None,
                "filing_url": SEC_ARCHIVES_BASE + row.filename,
                "error": f"HTTPError: {e}",
            })
        except Exception as e:
            print(f"ERROR: Failed for CIK {cik} ({row.filename}): {e}", file=sys.stderr)
            out_rows.append({
                "cik": row.cik,
                "filing_date": row.date_filed,
                "word_count": None,
                "sentence_count": None,
                "eps_diluted_current_q": None,
                "eps_diluted_prev_q": None,
                "filing_url": SEC_ARCHIVES_BASE + row.filename,
                "error": f"Exception: {type(e).__name__}: {e}",
            })

        time.sleep(max(0.0, args.sleep))

    # Writing CSV
    fieldnames = ["cik", "filing_date", "word_count", "sentence_count",
                  "eps_diluted_current_q", "eps_diluted_prev_q", "filing_url", "error"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            if "error" not in r:
                r["error"] = ""
            w.writerow(r)

    print(f"Wrote {len(out_rows)} rows to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())