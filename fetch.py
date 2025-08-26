# -*- coding: utf-8 -*-
"""
Fetch pandas.concat source via DevDocs:
1) Load DevDocs HTML for pandas.concat (from a provided URL or a list of candidates)
2) Extract the [source] GitHub link
3) Convert it to a raw URL and download the Python file
4) Save the full file, and optionally extract only the function body (default: 'concat')

Usage examples:
    python fetch_concat.py
    python fetch_concat.py out_full.py out_func.txt
    python fetch_concat.py --devdocs https://documents.devdocs.io/pandas~1.5/reference/api/pandas.concat.html
    python fetch_concat.py --only-full
    python fetch_concat.py --func-name concat
    python fetch_concat.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional, List

import requests
from bs4 import BeautifulSoup


# ---------- Config ----------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEV_DOCS_CANDIDATES: List[str] = [
    # Newer first:
    "https://documents.devdocs.io/pandas~2.3/reference/api/pandas.concat.html",
    "https://documents.devdocs.io/pandas~2.2/reference/api/pandas.concat.html",
    "https://documents.devdocs.io/pandas~2.1/reference/api/pandas.concat.html",
    "https://documents.devdocs.io/pandas~2.0/reference/api/pandas.concat.html",
    "https://documents.devdocs.io/pandas~1.5/reference/api/pandas.concat.html",
    "https://documents.devdocs.io/pandas~1/reference/api/pandas.concat.html",
    # Less common fallbacks:
    "https://documents.devdocs.io/pandas/reference/api/pandas.concat.html",
    "https://documents.devdocs.io/pandas~1/reference/api/pandas.concat",
]


# ---------- HTTP helpers ----------

def make_session() -> requests.Session:
    """Create a requests session with basic retry on GET."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_with_retry(
    session: requests.Session,
    url: str,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.8,
    verbose: bool = False,
) -> Optional[requests.Response]:
    """GET with simple retry/backoff."""
    for i in range(retries):
        try:
            if verbose:
                print(f"  -> GET {url} (try {i+1}/{retries})")
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r
            if verbose:
                print(f"     status={r.status_code}, len={len(r.text)}")
        except requests.RequestException as e:
            if verbose:
                print(f"     error: {e}")
        time.sleep(backoff * (i + 1))
    return None


# ---------- Core steps ----------

def fetch_first_ok_html(session: requests.Session, urls: List[str], verbose: bool) -> Optional[str]:
    """Return first valid HTML that looks like the pandas.concat page."""
    for u in urls:
        r = get_with_retry(session, u, verbose=verbose)
        if r and "pandas.concat" in r.text:
            if verbose:
                print(f"     OK DevDocs page: {u}")
            return r.text
    return None


def extract_source_link_from_devdocs(html: str) -> str:
    """Find [source] link in DevDocs HTML; fallback to first GitHub link."""
    soup = BeautifulSoup(html, "lxml")

    a = soup.find("a", string=re.compile(r"\[?\s*source\s*\]?", re.I))
    if a and a.get("href"):
        return a["href"]

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "github.com" in href:
            return href

    raise RuntimeError("Could not find [source] or any GitHub link in DevDocs HTML.")


def github_blob_to_raw(github_url: str) -> str:
    """Convert GitHub blob URL to raw URL, strip #L... fragment."""
    github_url = github_url.split("#", 1)[0]  # remove #Lxxx
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", github_url)
    if not m:
        raise RuntimeError(f"Unsupported GitHub URL format: {github_url}")
    owner, repo, branch, path = m.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def extract_function_source(full_text: str, func_name: str = "concat") -> str:
    """Extract body of `def <func_name>(...)` from a Python module, until next top-level def/class."""
    start_pat = re.compile(rf"^def\s+{func_name}\s*\(", re.M)
    m = start_pat.search(full_text)
    if not m:
        raise RuntimeError(f"Cannot find function '{func_name}'.")

    start_idx = m.start()
    next_pat = re.compile(r"^(def|class)\s+\w+\s*\(", re.M)
    m_next = next_pat.search(full_text, pos=m.end())
    end_idx = m_next.start() if m_next else len(full_text)

    return full_text[start_idx:end_idx].rstrip() + "\n"


# ---------- CLI / Main ----------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch pandas.concat source via DevDocs.")
    p.add_argument("out_full", nargs="?", default="concat_full_source.py",
                   help="Path to save the full source file (default: concat_full_source.py)")
    p.add_argument("out_func", nargs="?", default="pandas_concat_source.txt",
                   help="Path to save only the function body (default: pandas_concat_source.txt)")
    p.add_argument("--devdocs", metavar="URL", help="Exact DevDocs 'documents.devdocs.io' URL for pandas.concat")
    p.add_argument("--only-full", action="store_true", help="Only save the full file (skip function extraction)")
    p.add_argument("--func-name", default="concat", help="Function name to extract (default: concat)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    session = make_session()

    print("[1/4] Fetch DevDocs page ...")
    if args.devdocs:
        html = fetch_first_ok_html(session, [args.devdocs], verbose=args.verbose)
        if not html:
            print("Provided DevDocs URL did not return a valid page. "
                  "Open devdocs.io, navigate to pandas.concat, copy its 'documents.devdocs.io' URL and try again.")
            return 1
    else:
        html = fetch_first_ok_html(session, DEV_DOCS_CANDIDATES, verbose=args.verbose)
        if not html:
            print("Could not find a working DevDocs URL.\n"
                  "   Tip: open devdocs.io → pandas → concat, copy the 'documents.devdocs.io' URL, "
                  "   and pass it via --devdocs.")
            return 1

    print("[2/4] Extract [source] GitHub link ...")
    gh_link = extract_source_link_from_devdocs(html)
    print(f"    -> {gh_link}")

    print("[3/4] Convert to raw & download ...")
    raw_url = github_blob_to_raw(gh_link)
    print(f"    -> {raw_url}")
    r = get_with_retry(session, raw_url, timeout=60, retries=3, backoff=0.8, verbose=args.verbose)
    if not r:
        print("Failed to download raw GitHub file.")
        return 1
    py_text = r.text

    Path(args.out_full).write_text(py_text, encoding="utf-8")
    print(f"Full file saved: {Path(args.out_full).resolve()}")

    if not args.only_full:
        print("[4/4] Extract function body ...")
        try:
            func_src = extract_function_source(py_text, args.func_name)
            Path(args.out_func).write_text(func_src, encoding="utf-8")
            print(f"Function body saved: {Path(args.out_func).resolve()}")
        except Exception as e:
            print(f"Could not extract only-function body: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())