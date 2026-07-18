"""Download real vehicle-safety complaints from the public NHTSA API.

Data source: NHTSA Office of Defects Investigation (ODI) complaints API.
  - Models endpoint:    https://api.nhtsa.gov/products/vehicle/models
  - Complaints endpoint: https://api.nhtsa.gov/complaints/complaintsByVehicle
The API is public, requires no key, and its data is in the public domain
(U.S. Government work). See https://www.nhtsa.gov/nhtsa-datasets-and-apis.

This module is intentionally free of side effects at import time; all work
happens inside functions and under the ``__main__`` guard.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_PATH = RAW_DIR / "nhtsa_complaints.jsonl"

# A diverse slice of popular U.S.-market vehicles. We resolve the exact NHTSA
# model spellings per (make, year) at runtime, so we only need make + a model
# keyword here. Keeping many makes/models/years avoids any single recall event
# dominating the component distribution.
MAKES: dict[str, list[str]] = {
    "honda": ["ACCORD", "CIVIC", "CR-V", "PILOT", "ODYSSEY"],
    "toyota": ["CAMRY", "COROLLA", "RAV4", "HIGHLANDER", "TACOMA"],
    "ford": ["F-150", "ESCAPE", "EXPLORER", "FUSION", "FOCUS"],
    "chevrolet": ["SILVERADO", "EQUINOX", "MALIBU", "CRUZE", "TRAVERSE"],
    "nissan": ["ALTIMA", "ROGUE", "SENTRA", "PATHFINDER"],
    "jeep": ["GRAND CHEROKEE", "CHEROKEE", "WRANGLER"],
    "gmc": ["SIERRA", "ACADIA"],
    "ram": ["1500"],
    "subaru": ["OUTBACK", "FORESTER"],
    "hyundai": ["SONATA", "ELANTRA", "SANTA FE"],
    "kia": ["OPTIMA", "SORENTO", "SOUL"],
    "tesla": ["MODEL 3", "MODEL S", "MODEL Y"],
    "volkswagen": ["JETTA", "PASSAT"],
    "dodge": ["CHARGER", "DURANGO"],
}
YEARS = [2014, 2017, 2020]
API_BASE = "https://api.nhtsa.gov"


def _get_json(url: str, retries: int = 2, pause: float = 1.0, timeout: int = 25) -> dict:
    """GET a URL and parse JSON, with light retry on transient failures."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "module2-nlp-project/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as err:  # noqa: BLE001 - network errors are expected/skippable
            last_err = err
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed after {retries} retries: {url} ({last_err})")


def resolve_models(make: str, year: int) -> list[str]:
    """Return the exact NHTSA model spellings available for a make + year."""
    url = (
        f"{API_BASE}/products/vehicle/models?modelYear={year}"
        f"&make={urllib.parse.quote(make)}&issueType=c"
    )
    try:
        data = _get_json(url)
    except RuntimeError:
        return []
    return sorted({row["model"] for row in data.get("results", [])})


def match_models(available: Iterable[str], keywords: list[str]) -> list[str]:
    """Pick exact model names whose text contains one of our keywords."""
    picked: set[str] = set()
    for name in available:
        upper = name.upper()
        if any(kw.upper() in upper for kw in keywords):
            picked.add(name)
    return sorted(picked)


def fetch_complaints(make: str, model: str, year: int) -> list[dict]:
    """Fetch all complaints for one (make, model, year)."""
    url = (
        f"{API_BASE}/complaints/complaintsByVehicle?make={urllib.parse.quote(make)}"
        f"&model={urllib.parse.quote(model)}&modelYear={year}"
    )
    try:
        data = _get_json(url)
    except RuntimeError:
        return []
    return data.get("results", [])


def _slim(row: dict, make: str, model: str, year: int) -> dict:
    """Keep only the fields we need, normalising types."""
    return {
        "odiNumber": row.get("odiNumber"),
        "make": make,
        "model": model,
        "modelYear": year,
        "summary": (row.get("summary") or "").strip(),
        "components": (row.get("components") or "").strip(),
        "crash": bool(row.get("crash")),
        "fire": bool(row.get("fire")),
        "numberOfInjuries": int(row.get("numberOfInjuries") or 0),
        "numberOfDeaths": int(row.get("numberOfDeaths") or 0),
        "dateComplaintFiled": row.get("dateComplaintFiled"),
    }


def download(makes: dict[str, list[str]], years: list[int], pause: float = 0.2,
             out_path: Path | None = None) -> int:
    """Crawl the make/model/year grid, streaming deduped complaints to disk.

    Writing incrementally means partial progress survives an interruption and
    stays visible in the output log.
    """
    seen: set[int] = set()
    total = 0
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8") if out_path is not None else None
    try:
        for make, keywords in makes.items():
            for year in years:
                available = resolve_models(make, year)
                for model in match_models(available, keywords):
                    rows = fetch_complaints(make, model, year)
                    kept = 0
                    for row in rows:
                        odi = row.get("odiNumber")
                        if odi in seen or not (row.get("summary") or "").strip():
                            continue
                        if not (row.get("components") or "").strip():
                            continue
                        seen.add(odi)
                        rec = _slim(row, make, model, year)
                        if fh is not None:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total += 1
                        kept += 1
                    if fh is not None:
                        fh.flush()
                    print(f"  {make:12s} {model:22s} {year}  +{kept:4d}  (total {total})", flush=True)
                    time.sleep(pause)
    finally:
        if fh is not None:
            fh.close()
    return total


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NHTSA complaints corpus")
    parser.add_argument("--out", type=Path, default=RAW_PATH)
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between API calls")
    args = parser.parse_args()

    print(f"Downloading NHTSA complaints -> {args.out}", flush=True)
    total = download(MAKES, YEARS, pause=args.pause, out_path=args.out)
    print(f"\nSaved {total} unique complaints to {args.out}")


if __name__ == "__main__":
    main()
