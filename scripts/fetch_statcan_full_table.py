#!/usr/bin/env python3
"""
Fetch a full StatCan table (PID) via the Web Data Service (WDS) and extract the CSV.

Uses:
GET https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/<PID>/en
Returns a URL like:
https://www150.statcan.gc.ca/n1/tbl/csv/<PID>-eng.zip
(See StatCan WDS user guide.)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import requests


WDS_FULL_CSV = "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/{lang}"


def _die(msg: str, code: int = 2) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def get_zip_url(pid: str, lang: str) -> str:
    url = WDS_FULL_CSV.format(pid=pid, lang=lang)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and payload.get("status") == "SUCCESS" and isinstance(payload.get("object"), str):
        return payload["object"]
    _die(f"Unexpected WDS response for PID={pid}: {json.dumps(payload)[:4000]}")


def download_file(url: str) -> bytes:
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return r.content


def extract_first_csv(zip_bytes: bytes, out_csv_path: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            _die("ZIP had no CSV inside.")
        # Usually there is exactly one CSV; pick the first
        name = csv_names[0]
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, open(out_csv_path, "wb") as dst:
            dst.write(src.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="14100287", help="StatCan Product ID (PID), e.g. 14100287")
    ap.add_argument("--lang", default="en", choices=["en", "fr"])
    ap.add_argument("--out_dir", default="data/raw", help="Where to write the extracted CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching WDS CSV download URL for PID={args.pid} ({args.lang}) ...")
    zip_url = get_zip_url(args.pid, args.lang)
    print(f"ZIP URL: {zip_url}")

    zip_path = out_dir / f"{args.pid}-{args.lang}.zip"
    csv_path = out_dir / f"{args.pid}.csv"

    print(f"Downloading -> {zip_path} ...")
    zip_bytes = download_file(zip_url)
    zip_path.write_bytes(zip_bytes)

    print(f"Extracting first CSV -> {csv_path} ...")
    extract_first_csv(zip_bytes, csv_path)

    print("Done.")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
