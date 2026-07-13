#!/usr/bin/env python3

import os
import csv
import shutil
from pathlib import Path
from datetime import datetime

OPR_DIR = Path("assets/charts/ОПР_статистика")
DEST_DIR = Path("data/raw/opr_stat")
CATALOG_FILE = Path("data/opr_data_catalog.csv")

DATA_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".csv", ".json"}

KEY_TERMS = {
    "budget": ["бюджет", "budget"],
    "hh": ["hh.ru", "hh_", "вакансии", "резюме"],
    "domclick": ["домклик", "domclick", "жилье", "жильё"],
    "consolidated": ["консолидированные"],
    "inflation": ["инфляц", "inflation"],
    "grp": ["врп", "grp"],
    "wages": ["зарп", "wage", "счр"],
    "prices": ["цен", "price"],
}


def categorize_file(filepath):
    filename = filepath.name.lower()
    category = "other"
    keywords = []

    for cat, terms in KEY_TERMS.items():
        for term in terms:
            if term in filename:
                if category == "other":
                    category = cat
                keywords.append(term)

    return category, list(set(keywords))


def get_file_metadata(filepath):
    stat = filepath.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {"size_bytes": stat.st_size, "size_mb": round(size_mb, 2), "modified": mtime}


def scan_opr_directory():
    catalog = []

    for filepath in OPR_DIR.rglob("*"):
        if filepath.is_file():
            metadata = get_file_metadata(filepath)
            category, keywords = categorize_file(filepath)

            catalog.append(
                {
                    "filename": filepath.name,
                    "relative_path": str(filepath.relative_to(OPR_DIR.parent)),
                    "absolute_path": str(filepath),
                    "extension": filepath.suffix,
                    "category": category,
                    "keywords": ";".join(keywords),
                    "size_bytes": metadata["size_bytes"],
                    "size_mb": metadata["size_mb"],
                    "modified": metadata["modified"],
                    "is_data_file": filepath.suffix in DATA_EXTENSIONS,
                }
            )

    return catalog


def copy_valid_data_files(catalog):
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copied_files = []

    for entry in catalog:
        if entry["is_data_file"]:
            src_path = Path(entry["absolute_path"])
            dest_path = DEST_DIR / entry["filename"]

            if not dest_path.exists():
                try:
                    shutil.copy2(src_path, dest_path)
                    copied_files.append(entry["filename"])
                except Exception as e:
                    print(f"Error copying {entry['filename']}: {e}")

    return copied_files


def write_catalog(catalog):
    fieldnames = [
        "filename",
        "relative_path",
        "category",
        "keywords",
        "extension",
        "size_bytes",
        "size_mb",
        "modified",
        "is_data_file",
    ]

    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CATALOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in catalog:
            writer.writerow({k: v for k, v in entry.items() if k in fieldnames})


def main():
    print("Scanning OPR directory...")
    catalog = scan_opr_directory()

    print(f"Found {len(catalog)} files")

    print(f"Copying valid data files to {DEST_DIR}...")
    copied = copy_valid_data_files(catalog)
    print(f"Copied {len(copied)} files")

    print(f"Writing catalog to {CATALOG_FILE}...")
    write_catalog(catalog)

    key_files = ["Консолидированные бюджеты", "hh.ru", "DomClick"]
    found_keys = []
    for key in key_files:
        key_lower = key.lower()
        found = any(
            key_lower in entry["filename"].lower()
            or key_lower in entry["keywords"].lower()
            for entry in catalog
        )
        if found:
            found_keys.append(key)

    print(f"\nCatalog Summary:")
    print(f"Total files: {len(catalog)}")
    print(f"Data files: {sum(1 for e in catalog if e['is_data_file'])}")
    print(f"Key files found: {', '.join(found_keys)}")
    print(f"Copied to {DEST_DIR}: {len(copied)} files")

    return catalog


if __name__ == "__main__":
    main()
