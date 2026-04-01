#!/usr/bin/env python3
"""
Scheduled scan job — executed by GitHub Actions every 2 hours on weekdays
during US market hours.

Runs a full screen of all US-listed equities and writes results to a GitHub
Gist so the Streamlit app can load them instantly on startup.

Required environment variables
-------------------------------
  GIST_TOKEN  — GitHub PAT with 'gist' scope
  GIST_ID     — ID of the public Gist to write to

Optional
--------
  NEWS_API_KEY — passed through for future sentiment pre-caching
"""
import json
import os
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screener import screen_stocks

GIST_TOKEN = os.environ.get("GIST_TOKEN", "")
GIST_ID    = os.environ.get("GIST_ID", "")

# Default scan parameters — kept in sync with app.py defaults
SCAN_PARAMS = dict(
    min_market_cap=500e6,   # $500M — captures small/mid-cap opportunities
    max_pe=50,              # applies only when P/E data is available
    min_vol_ratio=2.0,      # 2x = meaningful spike (research: 2.5-3x is institutional signal)
    max_vol_ratio=20.0,     # no effective ceiling — keep high-spike events
    min_5yr_high_pct=20,    # sweet spot per research: 20-35% off 5yr high
    min_avg_volume=100_000, # liquidity gate
)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def run_scan() -> dict:
    _log(f"Starting full market scan with params: {SCAN_PARAMS}")

    def _progress(current: int, total: int, msg: str) -> None:
        if total:
            _log(f"  {current:>3}/{total} — {msg}")

    df = screen_stocks(**SCAN_PARAMS, progress_callback=_progress)
    _log(f"Scan complete — {len(df)} candidates found.")

    records = df.to_dict("records") if not df.empty else []
    return {
        "scanned_at":      datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(records),
        "scan_params":     {k: str(v) for k, v in SCAN_PARAMS.items()},
        "results":         records,
    }


def push_to_gist(payload: dict) -> str:
    if not GIST_TOKEN or not GIST_ID:
        raise EnvironmentError("GIST_TOKEN and GIST_ID environment variables must be set.")

    headers = {
        "Authorization": f"token {GIST_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    body = {
        "files": {
            "scan_cache.json": {
                "content": json.dumps(payload, indent=2, default=str)
            }
        }
    }
    resp = requests.patch(
        f"https://api.github.com/gists/{GIST_ID}",
        headers=headers,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    url = resp.json().get("html_url", "")
    _log(f"Gist updated: {url}")
    return url


if __name__ == "__main__":
    if not GIST_TOKEN or not GIST_ID:
        print("ERROR: Set GIST_TOKEN and GIST_ID environment variables.", file=sys.stderr)
        sys.exit(1)

    payload = run_scan()
    push_to_gist(payload)
    _log("Done.")
