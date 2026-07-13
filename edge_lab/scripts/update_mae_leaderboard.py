#!/usr/bin/env python3
"""
MAE Leaderboard Updater
=======================
Automatically updates CLAUDE.md with current MAE values from backtest results.
"""

import re
from pathlib import Path
import pandas as pd
from collections import defaultdict


def load_backtest_results(data_dir: Path, archive_dir: Path) -> dict:
    """Load all available backtest results from CSV files."""
    mae_values = defaultdict(dict)

    # Load consolidated metrics
    consolidated = data_dir / "consolidated_metrics.csv"
    if consolidated.exists():
        try:
            df = pd.read_csv(consolidated)
            for _, row in df.iterrows():
                model_name = row["Model"]
                # Prefer h=1 for single MAE value
                if pd.notna(row.get("MAE_h1")):
                    mae_values[model_name] = float(row["MAE_h1"])
                elif pd.notna(row.get("Weighted_Score")):
                    mae_values[model_name] = float(row["Weighted_Score"])
        except Exception as e:
            print(f"Warning: Could not load consolidated metrics: {e}")

    # Load individual horizon backtests
    for h in [1, 2, 12]:
        file_path = archive_dir / f"backtest_h{h}_metrics.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    model_name = row["model"]
                    if pd.notna(row.get("MAE")):
                        mae_val = float(row["MAE"])
                        # Prefer h=1, store others for reference
                        if h == 1 or model_name not in mae_values:
                            mae_values[model_name] = mae_val
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

    # Load all metrics file
    all_metrics = archive_dir / "backtest_all_metrics.csv"
    if all_metrics.exists():
        try:
            df = pd.read_csv(all_metrics)
            for _, row in df.iterrows():
                model_name = row["model"]
                horizon = int(row.get("horizon", 1))
                if pd.notna(row.get("MAE")):
                    mae_val = float(row["MAE"])
                    # Prefer h=1
                    if horizon == 1 or model_name not in mae_values:
                        mae_values[model_name] = mae_val
        except Exception as e:
            print(f"Warning: Could not load all metrics: {e}")

    return mae_values


def parse_claude_md(claude_path: Path) -> tuple:
    """Parse CLAUDE.md to extract MAE patterns and sections."""
    content = claude_path.read_text(encoding="utf-8")

    # Find model entries with MAE values
    # Pattern: - **MAE**: <value>
    mae_pattern = r"- \*\*MAE\*\*:\s*([^\n]+)"
    matches = list(re.finditer(mae_pattern, content))

    model_mae_positions = []
    for match in matches:
        current_mae = match.group(1).strip()
        start_pos = match.start(1)
        end_pos = match.end(1)

        # Extract model name from nearest ### header before this match
        text_before = content[: match.start()]
        header_matches = list(
            re.finditer(r"^###\s+([\w\s().,]+?)\s*$", text_before, re.MULTILINE)
        )
        if header_matches:
            model_name = header_matches[-1].group(1).strip()
            model_mae_positions.append((model_name, start_pos, end_pos, current_mae))

    return content, model_mae_positions


def build_model_name_map() -> dict:
    """Map model names in backtest results to CLAUDE.md section names."""
    return {
        "opr_ridge": "OPREnhancedRidgeForecaster",
        "ridge_extended": "RidgeExtendedForecaster",
        "ridge_shock": "RidgeShockDummiesForecaster",
        "ridge_macro": "RidgeMacroForecaster",
        "elasticnet": "ElasticNetForecaster",
        "huber": "HuberForecaster",
        "ngboost": "NGBoostForecaster",
        "ngboost_shock": "NGBoostShockForecaster",
        "ebm": "EBMForecaster",
        "bayesian_ridge": "BayesianRidgeForecaster",
        "subcomponent": "SubcomponentForecaster",
        "subcomponent_multi": "SubcomponentMultiForecaster",
        "microcomponent": "MicrocomponentForecaster",
        "hier_micro": "HierarchicalMicroForecaster",
        "horizon_ensemble": "HorizonEnsembleForecaster",
        "weekly": "WeeklySignalForecaster",
        "midas": "MIDASForecaster",
        "exog_prophet": "ExogProphetForecaster",
    }


def update_mae_in_content(
    content: str, model_mae_positions: list, backtest_maes: dict, name_map: dict
) -> str:
    """Update MAE values in CLAUDE.md content."""
    content_list = list(content)
    updates_made = []

    for model_name, start_pos, end_pos, current_mae in model_mae_positions:
        # Try to find matching backtest data
        mae_value = None

        # Direct name match
        for bt_name, bt_mae in backtest_maes.items():
            if (
                bt_name.lower() in model_name.lower()
                or model_name.lower() in bt_name.lower()
            ):
                mae_value = bt_mae
                break

        # Try name map
        if mae_value is None:
            for bt_name, mapped_name in name_map.items():
                if mapped_name.lower() in model_name.lower():
                    if bt_name in backtest_maes:
                        mae_value = backtest_maes[bt_name]
                        break

        if mae_value is not None:
            # Check if value is different
            new_mae_str = f"{mae_value:.4f}"
            if current_mae != new_mae_str:
                # Replace the MAE value
                old_text = content[start_pos:end_pos]
                content[start_pos:end_pos] = new_mae_str
                updates_made.append((model_name, current_mae, new_mae_str))

    return "".join(content_list), updates_made


def generate_leaderboard_section(backtest_maes: dict) -> str:
    """Generate markdown leaderboard section from backtest MAE values."""
    if not backtest_maes:
        return ""

    # Sort by MAE (lower is better)
    sorted_maes = sorted(backtest_maes.items(), key=lambda x: x[1])

    lines = [
        "",
        "---",
        "",
        "## 🏆 Performance Leaderboard",
        "",
        "*(Auto-generated from backtest results)*",
        "",
        "| Rank | Model | MAE (h=1) | Status |",
        "|------|-------|-----------|--------|",
    ]

    for rank, (model_name, mae) in enumerate(sorted_maes, 1):
        status = "✅ Verified" if mae < 0.4 else "⚠️ Needs Improvement"
        lines.append(f"| {rank} | {model_name} | {mae:.4f} | {status} |")

    lines.append("")
    lines.append("*Last updated: Automatically generated*")
    lines.append("")

    return "\n".join(lines)


def find_or_replace_leaderboard(content: str, new_leaderboard: str) -> str:
    """Find existing leaderboard section or append new one."""
    leaderboard_pattern = r"## 🏆 Performance Leaderboard.*?(?=\n## |\Z)"

    if re.search(leaderboard_pattern, content, re.DOTALL):
        # Replace existing leaderboard
        content = re.sub(
            leaderboard_pattern, new_leaderboard.strip(), content, flags=re.DOTALL
        )
    else:
        # Append new leaderboard before the end
        content = content.rstrip() + "\n" + new_leaderboard

    return content


def main():
    """Main function."""
    # Paths
    base_dir = Path("/home/valalav/_projects/sirena-kbr/edge_lab")
    claude_path = base_dir / "CLAUDE.md"
    data_dir = base_dir / "data"
    archive_dir = base_dir / "archive" / "results"

    if not claude_path.exists():
        print(f"Error: CLAUDE.md not found at {claude_path}")
        return 1

    print("Loading backtest results...")
    backtest_maes = load_backtest_results(data_dir, archive_dir)
    print(f"Found MAE values for {len(backtest_maes)} models:")
    for model_name, mae in sorted(backtest_maes.items()):
        print(f"  - {model_name}: {mae:.4f}")

    print("\nParsing CLAUDE.md...")
    content, model_mae_positions = parse_claude_md(claude_path)
    print(f"Found {len(model_mae_positions)} MAE entries in documentation")

    print("\nUpdating MAE values...")
    name_map = build_model_name_map()
    updated_content, updates = update_mae_in_content(
        content, model_mae_positions, backtest_maes, name_map
    )

    if updates:
        print(f"Updated {len(updates)} MAE values:")
        for model, old_val, new_val in updates:
            print(f"  - {model}: {old_val} → {new_val}")
    else:
        print("No MAE values needed updating")

    print("\nGenerating leaderboard section...")
    leaderboard = generate_leaderboard_section(backtest_maes)
    final_content = find_or_replace_leaderboard(updated_content, leaderboard)

    # Write updated content
    claude_path.write_text(final_content, encoding="utf-8")
    print(f"\nUpdated CLAUDE.md saved")

    return 0


if __name__ == "__main__":
    exit(main())
