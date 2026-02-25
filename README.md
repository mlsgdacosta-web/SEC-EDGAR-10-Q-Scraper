# SEC-EDGAR-10-Q-Scraper
This repository contains a Python implementation for scraping and processing 10-Q filings from the SEC EDGAR database.  The script programmatically retrieves filings, cleans the raw text, computes linguistic metrics, and extracts financial data directly from embedded XBRL tags.

**What This Project Does**

For a provided list of CIKs and the Q2 2020 SEC master index:
- Identifies the most recent 10-Q filing per CIK
- Downloads the full filing from SEC Archives
- Cleans HTML/XML markup for text analysis

**Computes:**
- Word count (excluding numeric/symbol-only tokens)
- Sentence count (using structured punctuation rules)

**Extracts diluted EPS from XBRL:**
- Current quarter
- Previous comparable quarter
- Outputs results to a structured CSV file
- Technical Structure

**The script:**
- Parses the SEC form.idx master index (fixed-width format)
- Selects the latest 10-Q per firm
- Downloads filings using a compliant SEC User-Agent
- Cleans filing text (removes script/style blocks and tags)
- Implements custom token filtering for accurate word counts
- Extracts us-gaap:EarningsPerShareDiluted facts from XBRL
- Identifies quarter-length contexts (75–110 days)
- Selects the two most recent quarterly EPS values
- Writes structured output to CSV

**Design Choices**
- Uses regular expressions for robust HTML/XBRL parsing
- Filters out numeric-only tokens to avoid inflating word counts
- Uses median aggregation when duplicate EPS facts exist
- Includes request throttling to comply with SEC guidelines
- Implements structured error handling for HTTP and parsing failures

**Why This Matters**
- Programmatic interaction with SEC EDGAR
- Parsing fixed-width index files
- XBRL fact extraction
- Text preprocessing for linguistic metrics
- Reproducible data pipeline design
- Clean error handling and rate-limited web scraping

It combines financial data extraction and textual analysis in a fully automated workflow suitable for research applications.
