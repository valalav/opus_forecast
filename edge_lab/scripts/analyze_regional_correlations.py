#!/usr/bin/env python3
"""
Regional Correlation Analysis Script

This script analyzes correlations between RF (Russian Federation), SKFO (District),
and KBR (Region) for key economic indicators. It also performs lead-lag detection
to identify if RF trends lead KBR trends.

Author: Auto-generated for Task 111
Date: 2025-01-22
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.signal import correlate
import warnings

warnings.filterwarnings("ignore")


class RegionalCorrelationAnalyzer:
    """
    Analyze correlations and lead-lag relationships between regional hierarchy levels.
    """

    # Top 5 key indicators to analyze (full path format)
    KEY_INDICATORS = [
        "01 промышленность/01-01 индекс промышленного производства.xlsx",
        "09 потребительские цены/09-01 цены на товары и услуги.xlsx",
        "05 торговля/05-01 оборот розничной торговли.xlsx",
        "11 доходы/11-01 среднедушевые денежные доходы.xlsx",
        "12 заработная плата/12-01 среднемесячная заработная плата.xlsx",
    ]

    def __init__(self, data_path: str = None):
        """
        Initialize the analyzer.

        Args:
            data_path: Path to regional_hierarchy_data.csv
        """
        if data_path is None:
            data_path = (
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/regional_hierarchy_data.csv"
            )

        self.data_path = Path(data_path)
        self.data = None
        self.correlation_results = {}
        self.lead_lag_results = {}

    def load_data(self) -> bool:
        """
        Load regional hierarchy data from CSV.

        Returns:
            True if successful, False otherwise
        """
        if not self.data_path.exists():
            print(f"❌ Data file not found: {self.data_path}")
            return False

        self.data = pd.read_csv(self.data_path, encoding="utf-8-sig")
        print(f"✅ Loaded {len(self.data)} data rows from {self.data_path}")
        return True

    def parse_year_column(self, column: str) -> int:
        """
        Parse year from column name.

        Args:
            column: Column name (e.g., "2016 год", "2017 год")

        Returns:
            Year as integer
        """
        try:
            year_str = str(column).split()[0]
            return int(year_str)
        except:
            return None

    def prepare_indicator_data(self, filename: str) -> pd.DataFrame:
        """
        Prepare data for a specific indicator, pivoting by region.

        Args:
            filename: Indicator file name

        Returns:
            DataFrame with years as index and regions as columns
        """
        indicator_data = self.data[self.data["filename"] == filename].copy()

        if len(indicator_data) == 0:
            return None

        # Convert values to numeric
        indicator_data["value"] = pd.to_numeric(
            indicator_data["value"], errors="coerce"
        )

        # Parse year from column
        indicator_data["year"] = indicator_data["column"].apply(
            lambda x: self.parse_year_column(x)
        )

        # Filter out rows where year parsing failed
        indicator_data = indicator_data[indicator_data["year"].notna()]

        # Pivot to get years as index and regions as columns
        pivoted = indicator_data.pivot(index="year", columns="region", values="value")

        # Ensure all regions exist
        for region in ["RF", "SKFO", "KBR"]:
            if region not in pivoted.columns:
                return None

        return pivoted

    def calculate_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Pearson correlation between regions.

        Args:
            df: DataFrame with regions as columns

        Returns:
            Dictionary with correlation coefficients
        """
        correlations = {}

        # RF vs SKFO
        if "RF" in df.columns and "SKFO" in df.columns:
            corr, pval = stats.pearsonr(df["RF"].dropna(), df["SKFO"].dropna())
            correlations["RF-SKFO"] = {"correlation": corr, "p_value": pval}

        # RF vs KBR
        if "RF" in df.columns and "KBR" in df.columns:
            corr, pval = stats.pearsonr(df["RF"].dropna(), df["KBR"].dropna())
            correlations["RF-KBR"] = {"correlation": corr, "p_value": pval}

        # SKFO vs KBR
        if "SKFO" in df.columns and "KBR" in df.columns:
            corr, pval = stats.pearsonr(df["SKFO"].dropna(), df["KBR"].dropna())
            correlations["SKFO-KBR"] = {"correlation": corr, "p_value": pval}

        return correlations

    def detect_lead_lag(self, df: pd.DataFrame, max_lag: int = 12) -> Dict:
        """
        Detect lead-lag relationships using cross-correlation.

        Args:
            df: DataFrame with regions as columns
            max_lag: Maximum number of periods to check (in months/years)

        Returns:
            Dictionary with lead-lag information
        """
        results = {}

        for pair in [("RF", "KBR"), ("SKFO", "KBR")]:
            region1, region2 = pair

            if region1 not in df.columns or region2 not in df.columns:
                continue

            series1 = df[region1].dropna().values
            series2 = df[region2].dropna().values

            if len(series1) < 4 or len(series2) < 4:
                continue

            # Normalize both series
            series1_norm = (series1 - np.mean(series1)) / np.std(series1)
            series2_norm = (series2 - np.mean(series2)) / np.std(series2)

            # Calculate cross-correlation
            cross_corr = correlate(series1_norm, series2_norm, mode="full")
            lags = np.arange(-len(series2) + 1, len(series1))

            # Find peak correlation and its lag
            peak_idx = np.argmax(np.abs(cross_corr))
            peak_lag = lags[peak_idx]
            peak_corr = cross_corr[peak_idx] / len(series1)

            results[f"{region1}-{region2}"] = {
                "optimal_lag": int(peak_lag),
                "max_correlation": float(peak_corr),
                "interpretation": self._interpret_lag(peak_lag, region1, region2),
            }

        return results

    def _interpret_lag(self, lag: int, region1: str, region2: str) -> str:
        """
        Interpret the lag value.

        Args:
            lag: Lag value (positive = region1 leads, negative = region2 leads)
            region1: First region name
            region2: Second region name

        Returns:
            Interpretation string
        """
        if lag == 0:
            return "No lead-lag relationship (synchronous)"
        elif lag > 0:
            return f"{region1} leads {region2} by {lag} period(s)"
        else:
            return f"{region2} leads {region1} by {abs(lag)} period(s)"

    def analyze_indicator(self, filename: str) -> Dict:
        """
        Perform complete analysis for a single indicator.

        Args:
            filename: Indicator file name

        Returns:
            Dictionary with analysis results
        """
        df = self.prepare_indicator_data(filename)

        if df is None:
            return None

        correlations = self.calculate_correlation(df)
        lead_lag = self.detect_lead_lag(df)

        return {"correlations": correlations, "lead_lag": lead_lag, "data": df}

    def run_analysis(self) -> Dict:
        """
        Run correlation analysis on all key indicators.

        Returns:
            Dictionary with all analysis results
        """
        print("🔬 Starting Regional Correlation Analysis...")

        results = {}
        analyzed_count = 0

        for indicator in self.KEY_INDICATORS:
            indicator_name = (
                indicator.split("/")[-1].replace(".xlsx", "").replace(".xls", "")
            )
            print(f"\n📊 Analyzing: {indicator_name}")

            result = self.analyze_indicator(indicator)

            if result:
                results[indicator] = result
                analyzed_count += 1

                # Print correlation matrix
                corr = result["correlations"]
                print("  Correlations:")
                for pair, data in corr.items():
                    print(
                        f"    {pair}: {data['correlation']:.3f} (p={data['p_value']:.3f})"
                    )

                # Print lead-lag results
                ll = result["lead_lag"]
                print("  Lead-Lag:")
                for pair, data in ll.items():
                    print(
                        f"    {pair}: {data['interpretation']} (corr={data['max_correlation']:.3f})"
                    )
            else:
                print(f"  ⚠️  No data available")

        print(f"\n✅ Analyzed {analyzed_count}/{len(self.KEY_INDICATORS)} indicators")

        return results

    def generate_correlation_matrix(self, results: Dict) -> pd.DataFrame:
        """
        Generate a summary correlation matrix for all indicators.

        Args:
            results: Analysis results dictionary

        Returns:
            DataFrame with correlation matrix
        """
        matrix_data = []

        for indicator, result in results.items():
            indicator_name = (
                indicator.split("/")[-1].replace(".xlsx", "").replace(".xls", "")
            )

            for pair, data in result["correlations"].items():
                matrix_data.append(
                    {
                        "Indicator": indicator_name,
                        "Pair": pair,
                        "Correlation": data["correlation"],
                        "P-Value": data["p_value"],
                    }
                )

        df = pd.DataFrame(matrix_data)
        return df

    def generate_lead_lag_summary(self, results: Dict) -> pd.DataFrame:
        """
        Generate a summary of lead-lag relationships.

        Args:
            results: Analysis results dictionary

        Returns:
            DataFrame with lead-lag summary
        """
        summary_data = []

        for indicator, result in results.items():
            indicator_name = (
                indicator.split("/")[-1].replace(".xlsx", "").replace(".xls", "")
            )

            for pair, data in result["lead_lag"].items():
                summary_data.append(
                    {
                        "Indicator": indicator_name,
                        "Pair": pair,
                        "Optimal_Lag": data["optimal_lag"],
                        "Max_Correlation": data["max_correlation"],
                        "Interpretation": data["interpretation"],
                    }
                )

        df = pd.DataFrame(summary_data)
        return df

    def save_correlation_report(self, results: Dict, output_path: str = None):
        """
        Save correlation analysis report to Markdown file.

        Args:
            results: Analysis results dictionary
            output_path: Path for output file
        """
        if output_path is None:
            output_path = (
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/correlation_report.md"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Regional Correlation Analysis Report\n\n")
            f.write("Generated: 2025-01-22\n\n")
            f.write(
                "This report analyzes correlations between RF (Russian Federation), SKFO (District),\n"
            )
            f.write("and KBR (Region) for key economic indicators.\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Indicators analyzed: {len(results)}\n")
            f.write("- Regional hierarchy levels: RF, SKFO, KBR\n\n")

            # Correlation Matrix
            f.write("## Correlation Matrix (Top 5 Indicators)\n\n")
            corr_matrix = self.generate_correlation_matrix(results)

            f.write("| Indicator | Pair | Correlation | P-Value |\n")
            f.write("|-----------|------|-------------|---------|\n")
            for _, row in corr_matrix.iterrows():
                f.write(
                    f"| {row['Indicator']} | {row['Pair']} | {row['Correlation']:.3f} | {row['P-Value']:.3f} |\n"
                )

            # Lead-Lag Analysis
            f.write("\n## Lead-Lag Analysis\n\n")
            ll_summary = self.generate_lead_lag_summary(results)

            f.write(
                "| Indicator | Pair | Lag Periods | Correlation | Interpretation |\n"
            )
            f.write(
                "|-----------|------|-------------|-------------|----------------|\n"
            )
            for _, row in ll_summary.iterrows():
                f.write(
                    f"| {row['Indicator']} | {row['Pair']} | {row['Optimal_Lag']} | {row['Max_Correlation']:.3f} | {row['Interpretation']} |\n"
                )

            # Key Findings
            f.write("\n## Key Findings\n\n")

            # Find strongest correlations
            strongest = corr_matrix.loc[corr_matrix["Correlation"].abs().idxmax()]
            f.write(
                f"1. **Strongest Correlation**: {strongest['Indicator']} - {strongest['Pair']} ({strongest['Correlation']:.3f})\n"
            )

            # RF leading KBR indicators
            rf_leads = ll_summary[
                (ll_summary["Pair"] == "RF-KBR") & (ll_summary["Optimal_Lag"] > 0)
            ]
            if len(rf_leads) > 0:
                f.write(
                    f"\n2. **RF Leading KBR**: {len(rf_leads)} indicators show RF trends leading KBR\n"
                )
                for _, row in rf_leads.iterrows():
                    f.write(f"   - {row['Indicator']}: {row['Interpretation']}\n")

            # KBR leading RF indicators
            kbr_leads = ll_summary[
                (ll_summary["Pair"] == "RF-KBR") & (ll_summary["Optimal_Lag"] < 0)
            ]
            if len(kbr_leads) > 0:
                f.write(
                    f"\n3. **KBR Leading RF**: {len(kbr_leads)} indicators show KBR trends leading RF\n"
                )
                for _, row in kbr_leads.iterrows():
                    f.write(f"   - {row['Indicator']}: {row['Interpretation']}\n")

            # Synchronous relationships
            sync = ll_summary[ll_summary["Optimal_Lag"] == 0]
            if len(sync) > 0:
                f.write(
                    f"\n4. **Synchronous Relationships**: {len(sync)} indicators move simultaneously\n"
                )

            f.write("\n---\n")
            f.write(
                "\n*Note: Correlation coefficients range from -1 (perfect negative) to +1 (perfect positive).\n"
            )
            f.write(
                "P-values < 0.05 indicate statistically significant correlations.*\n"
            )

        print(f"📝 Correlation report saved: {output_path}")

    def save_detailed_results(self, results: Dict, output_dir: str = None):
        """
        Save detailed correlation and lead-lag results to CSV files.

        Args:
            results: Analysis results dictionary
            output_dir: Directory for output files
        """
        if output_dir is None:
            output_dir = (
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/correlation_analysis"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Save correlation matrix
        corr_matrix = self.generate_correlation_matrix(results)
        corr_path = os.path.join(output_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(corr_path, index=False, encoding="utf-8-sig")
        print(f"💾 Correlation matrix saved: {corr_path}")

        # Save lead-lag summary
        ll_summary = self.generate_lead_lag_summary(results)
        ll_path = os.path.join(output_dir, "lead_lag_summary.csv")
        ll_summary.to_csv(ll_path, index=False, encoding="utf-8-sig")
        print(f"💾 Lead-lag summary saved: {ll_path}")


def main():
    """Main entry point."""
    analyzer = RegionalCorrelationAnalyzer()

    if not analyzer.load_data():
        return

    # Run analysis
    results = analyzer.run_analysis()

    if len(results) == 0:
        print("❌ No indicators could be analyzed")
        return

    # Save outputs
    analyzer.save_correlation_report(results)
    analyzer.save_detailed_results(results)

    print("\n✅ Regional Correlation Analysis Complete!")

    return results


if __name__ == "__main__":
    main()
