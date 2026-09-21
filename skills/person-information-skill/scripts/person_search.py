"""
Person registry query tool for searching employee records in registry.csv.
"""
import csv
import json
import sys
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "registry.csv"

def query_person_registry(keyword: str = "", field: str = None) -> list:
    """
    Search registry.csv for matching persons across name, city, country, or job_title.
    """
    results = []
    if not REGISTRY_PATH.exists():
        return results

    keyword_clean = keyword.lower().strip()
    with open(REGISTRY_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not keyword_clean:
                results.append(row)
                continue
            if field and field in row:
                if keyword_clean in row[field].lower():
                    results.append(row)
            else:
                if any(keyword_clean in str(val).lower() for val in row.values()):
                    results.append(row)

    return results

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    field = sys.argv[2] if len(sys.argv) > 2 else None
    matches = query_person_registry(query, field)
    print(json.dumps(matches, indent=2))
