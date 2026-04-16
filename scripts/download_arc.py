from __future__ import annotations

"""
download_arc.py — Download NeuroGolf 2026 tasks directly via Kaggle REST API.

No kaggle.json needed. Just set your token:
  export KAGGLE_API_TOKEN=KGAT_your_token_here

Then run:
  python3 scripts/download_arc.py
"""

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

COMPETITION = "neurogolf-2026"
TASKS_DIR   = Path("tasks")
API_BASE    = "https://www.kaggle.com/api/v1"


def get_token() -> str:
    # 1. Try environment variable (new token format)
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token

    # 2. Try legacy kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            creds = json.load(f)
        username = creds.get("username", "")
        key      = creds.get("key", "")
        if username and key:
            return (username, key)   # basic auth tuple

    print("❌  No Kaggle credentials found.\n")
    print("Run this in your terminal (with your real token):")
    print("  export KAGGLE_API_TOKEN=KGAT_your_token_here\n")
    print("Get your token: kaggle.com/settings → API → Generate New Token")
    sys.exit(1)


def download():
    TASKS_DIR.mkdir(exist_ok=True)
    token = get_token()

    # Build auth headers
    if isinstance(token, tuple):
        auth = token          # (username, key) → requests basic auth
        headers = {}
    else:
        auth = None
        headers = {"Authorization": f"Bearer {token}"}

    url = f"{API_BASE}/competitions/data/download-all/{COMPETITION}"
    print(f"Connecting to Kaggle API...")

    resp = requests.get(url, headers=headers, auth=auth,
                        stream=True, allow_redirects=True, timeout=120)

    if resp.status_code == 401:
        print("❌  Authentication failed. Check your token is correct and not expired.")
        sys.exit(1)
    if resp.status_code == 403:
        print("❌  Access denied. You need to accept the competition rules first:")
        print(f"   https://www.kaggle.com/competitions/{COMPETITION}/rules")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"❌  HTTP {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)

    # Download with progress
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunks = []
    print(f"Downloading... ({total // 1_000_000} MB)" if total else "Downloading...")
    for chunk in resp.iter_content(chunk_size=1024 * 256):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded * 100 // total
            print(f"\r  {pct}%  ({downloaded // 1_000_000} MB)", end="", flush=True)
    print()

    # Extract zip
    data = b"".join(chunks)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        task_files = [n for n in zf.namelist() if n.startswith("task") and n.endswith(".json")]
        print(f"Extracting {len(task_files)} task files...")
        for name in task_files:
            dest = TASKS_DIR / Path(name).name
            dest.write_bytes(zf.read(name))

    task_count = len(list(TASKS_DIR.glob("task*.json")))
    print(f"\n✅  Done — {task_count} task files saved to {TASKS_DIR}/")


if __name__ == "__main__":
    download()
