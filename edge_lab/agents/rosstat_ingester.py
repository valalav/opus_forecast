#!/usr/bin/env python3
"""
Rosstat Autonomous Ingestion System

This agent processes raw regional statistics from Rosstat (info-stat directory).
It extracts Kabardino-Balkarian Republic (KBR) data from Excel files.

Author: Auto-generated for Task 110
Date: 2025-01-22
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np


def fuzzy_partial_ratio(s1: str, s2: str) -> int:
    """
    Simple fuzzy matching implementation.
    Returns score 0-100 based on partial string matching.
    """
    s1_lower = s1.lower().replace(" ", "")
    s2_lower = s2.lower().replace(" ", "")

    # Direct substring match
    if s2_lower in s1_lower or s1_lower in s2_lower:
        return 100

    # Character overlap ratio
    set1 = set(s1_lower)
    set2 = set(s2_lower)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0

    return int(intersection / union * 100)


# Suppress openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


@dataclass
class FileSchema:
    """Schema information for a parsed Excel file."""

    filename: str
    header_row: int
    rf_row: int
    skfo_row: int
    kbr_row: int
    frequency: str  # 'monthly' or 'quarterly' or 'unknown'
    units: str
    total_rows: int
    data_columns: int
    rf_found: bool
    skfo_found: bool
    kbr_found: bool
    extraction_successful: bool


class RosstatIngester:
    """
    Intelligent data ingestion agent for Rosstat regional statistics.

    Uses 'Anchor Strategy':
    1. Find the row containing 'Северо-Кавказский федеральный округ' (SKFO)
    2. Scan subsequent 1-20 rows for 'Кабардино-Балкарская Республика' (KBR)
    3. Extract KBR data using identified header structure
    """

    # Fuzzy matching threshold for KBR name variations
    FUZZY_THRESHOLD = 70

    # KBR name variations to match
    KBR_VARIATIONS = [
        "Кабардино-Балкарская Республика",
        "Кабардино-Балкария",
        "Кабардино-Балкарская",
        "КБР",
        "Кабардино-Балкарская Респ.",
        "Каб.-Балкарская Республика",
    ]

    # RF (Russian Federation) variations
    RF_VARIATIONS = [
        "Российская Федерация",
        "Российская Федерация1",
        "РФ",
        "Россия",
    ]

    # SKFO variations
    SKFO_VARIATIONS = [
        "Северо-Кавказский федеральный округ",
        "Северо-Кавказский ФО",
        "СКФО",
        "Северо-Кавказский",
    ]

    def __init__(self, base_path: str = None):
        """
        Initialize the ingester.

        Args:
            base_path: Path to info-stat directory. If None, uses default location.
        """
        if base_path is None:
            base_path = "/home/valalav/_projects/sirena-kbr/data/raw/info-stat"

        self.base_path = Path(base_path)
        self.schema_registry: Dict[str, Dict] = {}
        self.extraction_results: Dict[str, pd.DataFrame] = {}
        self.quality_issues: List[Dict] = []

    def find_all_files(self) -> List[Path]:
        """
        Recursively find all Excel files in the directory tree.

        Returns:
            List of Path objects for .xlsx and .xls files
        """
        excel_files = []
        for pattern in ["*.xlsx", "*.xls"]:
            excel_files.extend(self.base_path.rglob(pattern))
        return sorted(excel_files)

    def detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Identify the header row containing date columns.

        Heuristics:
        - Headers typically in rows 0-10
        - Look for year patterns (20xx) or month names

        Args:
            df: DataFrame with no header specified

        Returns:
            Row index of the header
        """
        # Russian month names
        months = [
            "январь",
            "февраль",
            "март",
            "апрель",
            "май",
            "июнь",
            "июль",
            "август",
            "сентябрь",
            "октябрь",
            "ноябрь",
            "декабрь",
            "январь-февраль",
            "январь-март",
        ]  # Cumulative months

        for i in range(min(15, len(df))):
            row_str = " ".join([str(x) for x in df.iloc[i, :20] if pd.notna(x)])

            # Check for year pattern
            if re.search(r"20[0-9]{2}\s+год", row_str):
                return i

            # Check for month names
            month_count = sum(1 for m in months if m.lower() in row_str.lower())
            if month_count >= 2:
                return i

            # Check for "2016 год", "2017 год" pattern in columns
            year_count = 0
            for col in range(min(20, len(df.columns))):
                cell_val = str(df.iloc[i, col])
                if re.match(r"^20\d\d\s*год$", cell_val.strip()):
                    year_count += 1
            if year_count >= 2:
                return i

        # Default to row 3 if no pattern found
        return 3

    def detect_frequency(self, df: pd.DataFrame, header_row: int) -> str:
        """
        Detect if data is monthly or quarterly.

        Args:
            df: DataFrame
            header_row: Index of header row

        Returns:
            'monthly', 'quarterly', or 'unknown'
        """
        # Check row below header_row for month names
        month_header_row = header_row + 1
        if month_header_row >= len(df):
            month_header_row = header_row

        header_row_data = df.iloc[month_header_row, :].astype(str).tolist()
        header_str = " ".join(header_row_data).lower()

        # Check for cumulative month names (indicates quarterly/annual)
        cumulative_patterns = [
            "январь-февраль",
            "январь-март",
            "январь-апрель",
            "январь-декабрь",
            "квартал",
            "i квартал",
            "ii квартал",
            "iii квартал",
            "iv квартал",
        ]

        if any(p in header_str for p in cumulative_patterns):
            return "quarterly"

        # Check for regular monthly patterns
        monthly_months = [
            "январь",
            "февраль",
            "март",
            "апрель",
            "май",
            "июнь",
            "июль",
            "август",
            "сентябрь",
            "октябрь",
            "ноябрь",
            "декабрь",
        ]
        month_count = sum(1 for m in monthly_months if m in header_str)

        if month_count >= 4:
            return "monthly"

        return "unknown"

    def detect_units(self, df: pd.DataFrame, header_row: int) -> str:
        """
        Detect units of measurement from row 0-1.

        Args:
            df: DataFrame
            header_row: Index of header row

        Returns:
            Unit string (e.g., '%', 'млн. руб.', 'тыс. человек')
        """
        for i in range(min(3, len(df))):
            row_str = " ".join([str(x) for x in df.iloc[i, :5] if pd.notna(x)])

            unit_patterns = [
                (r"в\s*%", "%"),
                (r"%", "%"),
                (r"млрд\s*\.?\s*руб", "млрд. руб."),
                (r"млн\s*\.?\s*руб", "млн. руб."),
                (r"тыс\s*\.?\s*человек", "тыс. человек"),
                (r"тыс\s*\.?\s*чел", "тыс. чел."),
                (r"человек", "чел."),
                (r"тонн", "тонн"),
                (r"млн\s*\.?\s*кв\s*\.?\s*м", "млн. кв. м"),
            ]

            for pattern, unit in unit_patterns:
                if re.search(pattern, row_str, re.IGNORECASE):
                    return unit

        return "unknown"

    def find_rf_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the row containing Российская Федерация (RF).

        Args:
            df: DataFrame

        Returns:
            Row index or None if not found
        """
        for idx, row in df.iterrows():
            cell_val = str(row[0]) if pd.notna(row[0]) else ""
            for variation in self.RF_VARIATIONS:
                if variation.lower() in cell_val.lower():
                    return idx
        return None

    def find_skfo_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the row containing Северо-Кавказский федеральный округ.

        Args:
            df: DataFrame

        Returns:
            Row index or None if not found
        """
        for idx, row in df.iterrows():
            cell_val = str(row[0]) if pd.notna(row[0]) else ""
            for variation in self.SKFO_VARIATIONS:
                if variation.lower() in cell_val.lower():
                    return idx
        return None

    def find_kbr_row(self, df: pd.DataFrame, skfo_row: int) -> Optional[int]:
        """
        Find KBR row within 1-20 rows after SKFO.

        Args:
            df: DataFrame
            skfo_row: Row index of SKFO

        Returns:
            Row index or None if not found
        """
        search_range = range(skfo_row + 1, min(skfo_row + 21, len(df)))

        for idx in search_range:
            cell_val = str(df.iloc[idx, 0]) if pd.notna(df.iloc[idx, 0]) else ""

            # Direct match
            for variation in self.KBR_VARIATIONS:
                if variation.lower() in cell_val.lower():
                    return idx

            # Fuzzy match for typos/variations
            score = fuzzy_partial_ratio(cell_val, "Кабардино-Балкарская Республика")
            if score >= self.FUZZY_THRESHOLD:
                return idx

        return None

    def extract_region_data(
        self, df: pd.DataFrame, schema: FileSchema, region: str, row_idx: int
    ) -> Optional[pd.DataFrame]:
        """
        Extract data for a specific region based on schema.

        Args:
            df: DataFrame
            schema: FileSchema with header_row
            region: Region name (rf, skfo, kbr)
            row_idx: Row index of the region data

        Returns:
            DataFrame with extracted data or None
        """
        if row_idx is None or row_idx < 0:
            return None

        try:
            headers = df.iloc[schema.header_row, :].fillna("").astype(str).tolist()
            data_row = df.iloc[row_idx, :].values

            result_df = pd.DataFrame({"column": headers, "value": data_row})
            result_df["region"] = region

            result_df = result_df[
                (result_df["column"] != "")
                & (pd.notna(result_df["value"]))
                & (result_df["value"] != "")
            ]

            return result_df

        except Exception as e:
            self.quality_issues.append(
                {
                    "filename": schema.filename,
                    "region": region,
                    "issue": "extraction_error",
                    "message": str(e),
                }
            )
            return None

    def extract_kbr_data(
        self, df: pd.DataFrame, schema: FileSchema
    ) -> Optional[pd.DataFrame]:
        """
        Extract KBR data based on schema.

        Args:
            df: DataFrame
            schema: FileSchema with header_row and kbr_row

        Returns:
            DataFrame with extracted data or None
        """
        if not schema.kbr_found:
            return None

        try:
            # Get headers
            headers = df.iloc[schema.header_row, :].fillna("").astype(str).tolist()
            data_row = df.iloc[schema.kbr_row, :].values

            # Create DataFrame
            result_df = pd.DataFrame({"column": headers, "value": data_row})

            # Filter out empty headers and data
            result_df = result_df[
                (result_df["column"] != "")
                & (pd.notna(result_df["value"]))
                & (result_df["value"] != "")
            ]

            return result_df

        except Exception as e:
            self.quality_issues.append(
                {
                    "filename": schema.filename,
                    "issue": "extraction_error",
                    "message": str(e),
                }
            )
            return None

    def check_data_quality(self, df: pd.DataFrame, schema: FileSchema) -> List[Dict]:
        """
        Check extracted data for quality issues.

        Args:
            df: Extracted DataFrame
            schema: FileSchema

        Returns:
            List of quality issues
        """
        issues = []

        if df is None or len(df) == 0:
            return issues

        # Check for suspicious zeros
        zero_count = (df["value"] == 0).sum()
        if zero_count > len(df) * 0.5:
            issues.append(
                {
                    "filename": schema.filename,
                    "issue": "excessive_zeros",
                    "message": f"{zero_count}/{len(df)} values are zero",
                }
            )

        # Check for too many NaNs
        nan_count = df["value"].isna().sum()
        if nan_count > len(df) * 0.3:
            issues.append(
                {
                    "filename": schema.filename,
                    "issue": "excessive_nans",
                    "message": f"{nan_count}/{len(df)} values are NaN",
                }
            )

        # Check for extreme outliers
        if pd.api.types.is_numeric_dtype(df["value"]):
            q1 = df["value"].quantile(0.25)
            q3 = df["value"].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = (
                    (df["value"] < (q1 - 3 * iqr)) | (df["value"] > (q3 + 3 * iqr))
                ).sum()
                if outliers > len(df) * 0.1:
                    issues.append(
                        {
                            "filename": schema.filename,
                            "issue": "extreme_outliers",
                            "message": f"{outliers}/{len(df)} extreme values detected",
                        }
                    )

        return issues

    def parse_file(self, filepath: Path) -> FileSchema:
        """
        Parse a single Excel file and extract schema information.

        Args:
            filepath: Path to Excel file

        Returns:
            FileSchema object with parsed information
        """
        filename = str(filepath.relative_to(self.base_path))

        try:
            df = pd.read_excel(
                filepath,
                header=None,
                engine="openpyxl" if filepath.suffix == ".xlsx" else "xlrd",
            )

            header_row = self.detect_header_row(df)
            rf_row = self.find_rf_row(df)
            skfo_row = self.find_skfo_row(df)
            kbr_row = self.find_kbr_row(df, skfo_row) if skfo_row is not None else None

            rf_found = rf_row is not None
            skfo_found = skfo_row is not None
            kbr_found = kbr_row is not None

            frequency = self.detect_frequency(df, header_row)
            units = self.detect_units(df, header_row)

            schema = FileSchema(
                filename=filename,
                header_row=header_row,
                rf_row=rf_row if rf_row else -1,
                skfo_row=skfo_row if skfo_row else -1,
                kbr_row=kbr_row if kbr_row else -1,
                frequency=frequency,
                units=units,
                total_rows=len(df),
                data_columns=len(df.columns),
                rf_found=rf_found,
                skfo_found=skfo_found,
                kbr_found=kbr_found,
                extraction_successful=False,
            )

            # Extract data for all regions found
            all_extracted = []
            if rf_found:
                rf_data = self.extract_region_data(df, schema, "RF", rf_row)
                if rf_data is not None:
                    all_extracted.append(rf_data)
            if skfo_found:
                skfo_data = self.extract_region_data(df, schema, "SKFO", skfo_row)
                if skfo_data is not None:
                    all_extracted.append(skfo_data)
            if kbr_found:
                kbr_data = self.extract_kbr_data(df, schema)
                if kbr_data is not None:
                    kbr_data["region"] = "KBR"
                    all_extracted.append(kbr_data)

            if all_extracted:
                schema.extraction_successful = True
                combined = pd.concat(all_extracted, ignore_index=True)
                self.extraction_results[filename] = combined

                quality_issues = self.check_data_quality(combined, schema)
                self.quality_issues.extend(quality_issues)

            return schema

        except Exception as e:
            self.quality_issues.append(
                {"filename": filename, "issue": "parse_error", "message": str(e)}
            )

            return FileSchema(
                filename=filename,
                header_row=-1,
                rf_row=-1,
                skfo_row=-1,
                kbr_row=-1,
                frequency="unknown",
                units="unknown",
                total_rows=0,
                data_columns=0,
                rf_found=False,
                skfo_found=False,
                kbr_found=False,
                extraction_successful=False,
            )

    def run_ingestion(self) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.

        Returns:
            Dictionary with ingestion statistics
        """
        print("🏭 Starting Rosstat Autonomous Ingestion System...")
        print(f"📁 Base path: {self.base_path}")

        # Find all files
        files = self.find_all_files()
        print(f"📊 Found {len(files)} Excel files")

        # Parse each file
        schemas = []
        for filepath in files:
            print(f"  Parsing: {filepath.name}...", end=" ")
            schema = self.parse_file(filepath)
            schemas.append(schema)

            # Convert to dict for storage
            self.schema_registry[schema.filename] = {
                "header_row": schema.header_row,
                "rf_row": schema.rf_row,
                "skfo_row": schema.skfo_row,
                "kbr_row": schema.kbr_row,
                "frequency": schema.frequency,
                "units": schema.units,
                "total_rows": schema.total_rows,
                "data_columns": schema.data_columns,
                "rf_found": schema.rf_found,
                "skfo_found": schema.skfo_found,
                "kbr_found": schema.kbr_found,
                "extraction_successful": schema.extraction_successful,
            }

            status = (
                "✅"
                if schema.rf_found and schema.skfo_found and schema.kbr_found
                else "⚠️"
            )
            print(status)

        # Statistics
        stats = {
            "total_files": len(files),
            "rf_found": sum(s.rf_found for s in schemas),
            "skfo_found": sum(s.skfo_found for s in schemas),
            "kbr_found": sum(s.kbr_found for s in schemas),
            "rf_missing": sum(not s.rf_found for s in schemas),
            "skfo_missing": sum(not s.skfo_found for s in schemas),
            "kbr_missing": sum(not s.kbr_found for s in schemas),
            "extraction_successful": sum(s.extraction_successful for s in schemas),
            "monthly_files": sum(s.frequency == "monthly" for s in schemas),
            "quarterly_files": sum(s.frequency == "quarterly" for s in schemas),
            "quality_issues": len(self.quality_issues),
            "avg_rf_row": np.mean([s.rf_row for s in schemas if s.rf_row > 0]),
            "avg_skfo_row": np.mean([s.skfo_row for s in schemas if s.skfo_row > 0]),
            "avg_kbr_row": np.mean([s.kbr_row for s in schemas if s.kbr_row > 0]),
        }

        return stats

    def save_regional_hierarchy_data(self, output_path: str = None):
        """
        Save all extracted regional hierarchy data to a single CSV.

        Args:
            output_path: Path for output file
        """
        if output_path is None:
            output_path = (
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/regional_hierarchy_data.csv"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        all_data = []
        for filename, df in self.extraction_results.items():
            df_copy = df.copy()
            df_copy["filename"] = filename
            all_data.append(df_copy)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"💾 Regional hierarchy data saved: {output_path}")
        else:
            print("⚠️  No data extracted to save")

    def save_schema_registry(self, output_path: str = None):
        """
        Save schema registry to JSON file.

        Args:
            output_path: Path for output file
        """
        if output_path is None:
            output_path = (
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/schema_registry.json"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.schema_registry, f, indent=2, ensure_ascii=False)

        print(f"💾 Schema registry saved: {output_path}")

    def save_quality_report(self, output_path: str = None):
        """
        Save quality report to Markdown file.

        Args:
            output_path: Path for output file
        """
        if output_path is None:
            output_path = (
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/data_quality_report.md"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Rosstat Data Quality Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total files processed: {len(self.schema_registry)}\n")
            f.write(
                f"- KBR data found: {sum(1 for s in self.schema_registry.values() if s['kbr_found'])}\n"
            )
            f.write(
                f"- KBR data missing: {sum(1 for s in self.schema_registry.values() if not s['kbr_found'])}\n"
            )
            f.write(f"- Quality issues detected: {len(self.quality_issues)}\n\n")

            if self.quality_issues:
                f.write("## Quality Issues\n\n")
                f.write("| Filename | Issue | Message |\n")
                f.write("|----------|--------|---------|\n")
                for issue in self.quality_issues:
                    f.write(
                        f"| {issue['filename']} | {issue['issue']} | {issue['message']} |\n"
                    )
            else:
                f.write("## Quality Issues\n\n")
                f.write("✅ No quality issues detected!\n\n")

            f.write("## Files Without KBR Data\n\n")
            missing_kbr = [
                k for k, v in self.schema_registry.items() if not v["kbr_found"]
            ]
            if missing_kbr:
                for filename in missing_kbr:
                    f.write(f"- {filename}\n")
            else:
                f.write("✅ All files contain KBR data!\n\n")

        print(f"📝 Quality report saved: {output_path}")

    def save_extracted_data(self, output_dir: str = None):
        """
        Save all extracted KBR data to CSV files.

        Args:
            output_dir: Directory for output files
        """
        if output_dir is None:
            output_dir = "/home/valalav/_projects/sirena-kbr/edge_lab/data/extracted_kbr"

        os.makedirs(output_dir, exist_ok=True)

        for filename, df in self.extraction_results.items():
            # Create safe filename
            safe_name = (
                filename.replace("/", "_")
                .replace("\\", "_")
                .replace(".xlsx", "")
                .replace(".xls", "")
            )
            output_path = os.path.join(output_dir, f"{safe_name}.csv")
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"💾 Extracted data saved to: {output_dir}")


def main():
    """Main entry point."""
    ingester = RosstatIngester()
    stats = ingester.run_ingestion()

    print("\n" + "=" * 60)
    print("📊 Ingestion Statistics:")
    print("=" * 60)
    print(f"Total files:        {stats['total_files']}")
    print(
        f"RF found:           {stats['rf_found']} ({stats['rf_found'] / stats['total_files'] * 100:.1f}%)"
    )
    print(
        f"SKFO found:         {stats['skfo_found']} ({stats['skfo_found'] / stats['total_files'] * 100:.1f}%)"
    )
    print(
        f"KBR found:          {stats['kbr_found']} ({stats['kbr_found'] / stats['total_files'] * 100:.1f}%)"
    )
    print(f"RF missing:         {stats['rf_missing']}")
    print(f"SKFO missing:       {stats['skfo_missing']}")
    print(f"KBR missing:        {stats['kbr_missing']}")
    print(f"Extracted:          {stats['extraction_successful']}")
    print(f"Monthly files:      {stats['monthly_files']}")
    print(f"Quarterly files:    {stats['quarterly_files']}")
    print(f"Quality issues:      {stats['quality_issues']}")
    if stats["avg_rf_row"] > 0:
        print(f"Avg RF row:         {stats['avg_rf_row']:.1f}")
    if stats["avg_skfo_row"] > 0:
        print(f"Avg SKFO row:       {stats['avg_skfo_row']:.1f}")
    if stats["avg_kbr_row"] > 0:
        print(f"Avg KBR row:        {stats['avg_kbr_row']:.1f}")
    print("=" * 60)

    # Save outputs
    ingester.save_schema_registry()
    ingester.save_quality_report()
    ingester.save_regional_hierarchy_data()
    ingester.save_extracted_data()

    return stats


if __name__ == "__main__":
    main()
