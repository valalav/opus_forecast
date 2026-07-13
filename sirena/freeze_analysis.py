from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_RESULTS_DIR = PROJECT_ROOT / "archive" / "results"
ASSETS_CHARTS_DIR = PROJECT_ROOT / "assets" / "charts"
LEGACY_PLOTS_DIR = DATA_DIR / "plots"

EXCLUSION_RULES = {
    162: "ЖКХ Level-5",
    400: "ЖКХ Level-5",
    416: "ЖКХ Level-5",
    444: "ЖКХ Level-5",
    445: "ЖКХ Level-5",
    446: "ЖКХ Level-5",
    449: "Нотариальная Level-5 позиция",
    545: "ЖКХ Level-5",
    589: "ЖКХ Level-5",
    590: "ЖКХ Level-5",
    664: "Нотариальная Level-5 позиция",
    676: "ЖКХ Level-5",
    679: "ЖКХ Level-5",
    681: "ЖКХ Level-5",
    969: "ЖКХ Level-5",
}

AGGREGATE_BUCKET_CODES = [1, 2, 4, 6, 7, 8, 9, 14, 53, 54, 55, 436, 510]
EDUCATION_SERVICE_PATTERN = (
    r"дополнительные занятия|"
    r"занятия на курсах|"
    r"начальный курс обучения вождению|"
    r"обучение в |"
    r"услуги (?:в системе )?образования|"
    r"услуги высшего образования|"
    r"услуги среднего образования|"
    r"услуги профессионального обучения|"
    r"услуги дошкольного воспитания"
)


_PLOT_BG = "#f2f2f2"
_PLOT_GRID = "#d6d6d6"
_PLOT_DARK = "#333333"
_PLOT_ACCENT = "#e78f5a"
_PLOT_FROZEN = "#f0a06c"
_PLOT_SIMPLE = "#4f86c6"
_PLOT_ACTIVE = "#4dba84"
_PLOT_COMPONENT_PALETTE = [
    "#80c4df",
    "#b0e0d0",
    "#f0c0c0",
    "#d7d7a6",
    "#b9a6d7",
    "#c9d9f0",
    "#d4a57f",
    "#7bc47f",
    "#f3a7c7",
    "#9fd0b3",
]
_MONTH_NAMES = [
    "Янв",
    "Фев",
    "Мар",
    "Апр",
    "Май",
    "Июн",
    "Июл",
    "Авг",
    "Сен",
    "Окт",
    "Ноя",
    "Дек",
]


def _format_pct(value: float | int) -> str:
    return f"{float(value):.1f}%"


def _normalize_component_name(value: object) -> str:
    is_missing = pd.isna(value)
    if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
        return "Без категории"
    text = str(value).strip()
    return text if text else "Без категории"


def _build_component_palette(values: pd.Series) -> dict[str, str]:
    unique_values = sorted({_normalize_component_name(value) for value in values})
    palette = {}
    for index, value in enumerate(unique_values):
        palette[value] = _PLOT_COMPONENT_PALETTE[index % len(_PLOT_COMPONENT_PALETTE)]
    return palette


def _decorate_axis(ax: Axes, title: str | None = None) -> None:
    ax.set_facecolor(_PLOT_BG)
    ax.tick_params(axis="both", colors=_PLOT_DARK, labelsize=9)
    ax.grid(color=_PLOT_GRID, alpha=0.6, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(_PLOT_GRID)
    if title is not None:
        ax.set_title(title, color=_PLOT_DARK, fontsize=12, pad=12, fontweight="bold")


def _annotate_point(ax: Axes, x: float, y: float, label: str, *, color: str) -> None:
    ax.annotate(
        label,
        (x, y),
        xytext=(8, 8),
        textcoords="offset points",
        color=color,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=color,
            alpha=0.85,
            linewidth=0.8,
        ),
    )


def _short_label(name: str, max_len: int = 46) -> str:
    text = str(name).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass(frozen=True)
class AnalysisOutputs:
    item_summary: pd.DataFrame
    monthly_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    exclusions_summary: pd.DataFrame
    education_summary: pd.DataFrame
    report_path: Path


def load_freeze_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    monthly = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "kbr_full_monthly.csv"))
    monthly = cast(pd.DataFrame, monthly[monthly["Region_code"] == 7].copy())
    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly["MoM"] = pd.to_numeric(monthly["MoM"], errors="coerce")
    monthly = cast(pd.DataFrame, monthly.dropna(subset=["MoM"]))
    monthly["Year"] = cast(pd.Series, monthly["Date"]).dt.year

    weights = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "access_weights.csv"))
    weights = cast(pd.DataFrame, weights[weights["Region_code"] == 7].copy())
    weights["Day"] = pd.to_datetime(weights["Day"], format="%d/%m/%y %H:%M:%S")
    weights["Year"] = cast(pd.Series, weights["Day"]).dt.year
    weights["Weight_vertical"] = pd.to_numeric(
        weights["Weight_vertical"], errors="coerce"
    )
    weights = cast(pd.DataFrame, weights.dropna(subset=["Weight_vertical"]))
    weights = cast(
        pd.DataFrame,
        weights[["Year", "Item_code", "Weight_vertical"]].drop_duplicates(),
    )

    item_names = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "items_names.csv"))
    item_names = cast(pd.DataFrame, item_names.drop_duplicates(subset=["Item_code"]))

    micro_sprav = cast(
        pd.DataFrame,
        pd.read_csv(DATA_DIR / "raw" / "micro_sprav.csv", sep=";", decimal=","),
    )
    micro_sprav.columns = [
        column.replace("\ufeff", "") for column in micro_sprav.columns
    ]
    micro_sprav = micro_sprav.rename(
        columns={
            "Товар": "Micro_name",
            "Компонент": "Component",
            "Субкомпонент": "Subcomponent",
        }
    )
    micro_sprav["Item_code"] = pd.to_numeric(micro_sprav["Item_code"], errors="coerce")
    micro_sprav = cast(pd.DataFrame, micro_sprav.dropna(subset=["Item_code"]).copy())
    micro_sprav["Item_code"] = cast(pd.Series, micro_sprav["Item_code"]).astype(int)
    micro_sprav = cast(pd.DataFrame, micro_sprav.drop_duplicates(subset=["Item_code"]))

    return monthly, weights, item_names, micro_sprav


def build_level5_scope(
    monthly: pd.DataFrame,
    weights: pd.DataFrame,
    item_names: pd.DataFrame,
    micro_sprav: pd.DataFrame,
) -> pd.DataFrame:
    level5 = monthly.merge(weights, on=["Year", "Item_code"], how="inner")
    level5 = level5.merge(
        micro_sprav[["Item_code", "Micro_name", "Component", "Subcomponent"]],
        on="Item_code",
        how="inner",
    )
    level5 = level5.merge(
        item_names[["Item_code", "Item_name"]], on="Item_code", how="left"
    )
    level5["Item_name"] = level5["Item_name"].fillna(level5["Micro_name"])
    level5["Item_label"] = level5["Item_name"]
    level5 = cast(
        pd.DataFrame,
        level5[~level5["Item_code"].isin(AGGREGATE_BUCKET_CODES)].copy(),
    )
    level5 = cast(
        pd.DataFrame,
        level5.sort_values(by=["Date", "Item_code"]).reset_index(drop=True),
    )
    return level5


def is_education_item(df: pd.DataFrame) -> pd.Series:
    item_name = df["Item_name"].fillna("")
    micro_name = df["Micro_name"].fillna("")
    combined = item_name + " " + micro_name
    component = df["Component"].fillna("")
    return component.str.fullmatch(r"Услуги", case=False) & combined.str.contains(
        EDUCATION_SERVICE_PATTERN,
        case=False,
        regex=True,
    )


def apply_exclusions(level5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclusion_codes = list(EXCLUSION_RULES)
    exclusions = cast(
        pd.DataFrame, level5[level5["Item_code"].isin(exclusion_codes)].copy()
    )
    included = cast(
        pd.DataFrame, level5[~level5["Item_code"].isin(exclusion_codes)].copy()
    )
    return included, exclusions


def add_freeze_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = cast(pd.DataFrame, df.copy())
    result["Freeze"] = np.isclose(result["MoM"], 100.0, atol=1e-9)
    result["MoM_pct"] = result["MoM"] - 100.0
    result["Abs_MoM_pct"] = np.abs(result["MoM_pct"])
    result["Weighted_MoM_pct"] = result["MoM_pct"] * result["Weight_vertical"]
    result["Weighted_abs_MoM_pct"] = result["Abs_MoM_pct"] * result["Weight_vertical"]
    result["Freeze_weight"] = result["Freeze"].astype(float) * result["Weight_vertical"]
    return result


def compute_monthly_summary(
    included: pd.DataFrame, monthly: pd.DataFrame
) -> pd.DataFrame:
    grouped = cast(
        pd.DataFrame,
        included.groupby("Date", as_index=False).agg(
            basket_weight=("Weight_vertical", "sum"),
            freeze_weight=("Freeze_weight", "sum"),
            freeze_items=("Freeze", "sum"),
            total_items=("Item_code", "nunique"),
            included_mom_pct=("Weighted_MoM_pct", "sum"),
            included_abs_mom_pct=("Weighted_abs_MoM_pct", "sum"),
        ),
    )
    grouped["FDI_weighted_pct"] = (
        grouped["freeze_weight"] / grouped["basket_weight"] * 100.0
    )
    grouped["FDI_simple_pct"] = grouped["freeze_items"] / grouped["total_items"] * 100.0
    grouped["Active_base_pct"] = 100.0 - grouped["FDI_weighted_pct"]
    grouped["Avg_abs_MoM_pct"] = np.where(
        grouped["basket_weight"] > 0,
        grouped["included_abs_mom_pct"] / grouped["basket_weight"],
        np.nan,
    )
    grouped["Included_MoM"] = (
        100.0 + grouped["included_mom_pct"] / grouped["basket_weight"]
    )

    headline = cast(
        pd.DataFrame,
        monthly.loc[monthly["Item_code"] == 1, ["Date", "MoM"]].drop_duplicates(
            subset=["Date"]
        ),
    )
    grouped = grouped.merge(
        headline.rename(columns={"MoM": "Headline_MoM"}), on="Date", how="left"
    )
    active_share = grouped["Active_base_pct"] / 100.0
    grouped["Implied_active_MoM"] = np.where(
        active_share > 0,
        100.0 + ((grouped["Headline_MoM"] - 100.0) / active_share),
        np.nan,
    )
    grouped["Year"] = cast(pd.Series, grouped["Date"]).dt.year
    grouped["Month"] = cast(pd.Series, grouped["Date"]).dt.month
    return cast(
        pd.DataFrame,
        grouped[
            [
                "Date",
                "Year",
                "Month",
                "basket_weight",
                "freeze_weight",
                "freeze_items",
                "total_items",
                "FDI_weighted_pct",
                "FDI_simple_pct",
                "Active_base_pct",
                "Avg_abs_MoM_pct",
                "Headline_MoM",
                "Included_MoM",
                "Implied_active_MoM",
            ]
        ],
    )


def compute_component_monthly_summary(included: pd.DataFrame) -> pd.DataFrame:
    component_input = included.copy()
    component_input["Date"] = pd.to_datetime(component_input["Date"])
    component_input["Year"] = cast(pd.Series, component_input["Date"]).dt.year
    component_input["Month"] = cast(pd.Series, component_input["Date"]).dt.month

    component_input = cast(
        pd.DataFrame,
        component_input[
            [
                "Date",
                "Year",
                "Month",
                "Component",
                "Weight_vertical",
                "Freeze_weight",
                "Freeze",
                "Weighted_MoM_pct",
            ]
        ].copy(),
    )
    component_input["Component"] = component_input["Component"].map(
        _normalize_component_name
    )

    component_monthly = cast(
        pd.DataFrame,
        component_input.groupby(
            ["Date", "Year", "Month", "Component"], as_index=False
        ).agg(
            basket_weight=("Weight_vertical", "sum"),
            freeze_weight=("Freeze_weight", "sum"),
            freeze_items=("Freeze", "sum"),
            total_items=("Weight_vertical", "count"),
            included_mom_pct=("Weighted_MoM_pct", "sum"),
        ),
    )
    component_monthly["FDI_weighted_pct"] = np.where(
        component_monthly["basket_weight"] > 0,
        component_monthly["freeze_weight"] / component_monthly["basket_weight"] * 100.0,
        np.nan,
    )
    component_monthly["FDI_simple_pct"] = np.where(
        component_monthly["total_items"] > 0,
        component_monthly["freeze_items"] / component_monthly["total_items"] * 100.0,
        np.nan,
    )
    component_monthly["Active_base_pct"] = 100.0 - component_monthly["FDI_weighted_pct"]
    return component_monthly


def _freeze_spells(values: pd.Series) -> tuple[int, int, float]:
    spells: list[int] = []
    current = 0
    for is_freeze in values.fillna(False).astype(bool):
        if is_freeze:
            current += 1
        elif current:
            spells.append(current)
            current = 0
    if current:
        spells.append(current)
    if not spells:
        return 0, 0, 0.0
    return len(spells), max(spells), float(np.mean(spells))


def compute_item_summary(included: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item_code, item_df in included.groupby("Item_code"):
        item_df = item_df.sort_values("Date")
        item_weight_sum = float(item_df["Weight_vertical"].sum())
        frozen_weight_sum = float(item_df["Freeze_weight"].sum())
        weighted_psi_pct = (
            (frozen_weight_sum / item_weight_sum * 100.0) if item_weight_sum else 0.0
        )
        spell_count, max_spell, avg_spell = _freeze_spells(
            cast(pd.Series, item_df["Freeze"])
        )
        thaw_moves = item_df.loc[
            item_df["Freeze"].shift(fill_value=False) & ~item_df["Freeze"], "MoM_pct"
        ].abs()
        rows.append(
            {
                "Item_code": item_code,
                "Item_name": item_df["Item_name"].iloc[0],
                "Component": item_df["Component"].iloc[0],
                "Subcomponent": item_df["Subcomponent"].iloc[0],
                "Avg_weight": item_df["Weight_vertical"].mean(),
                "Observations": len(item_df),
                "Freeze_months": int(item_df["Freeze"].sum()),
                "PSI_pct": item_df["Freeze"].mean() * 100.0,
                "Weighted_PSI_pct": weighted_psi_pct,
                "Mean_MoM": item_df["MoM"].mean(),
                "Mean_abs_MoM_pct": item_df["MoM_pct"].abs().mean(),
                "Freeze_spell_count": spell_count,
                "Max_freeze_spell": max_spell,
                "Avg_freeze_spell": avg_spell,
                "Weighted_abs_MoM_pct": item_df["Weighted_abs_MoM_pct"].mean(),
                "Avg_thaw_abs_move_pct": thaw_moves.mean()
                if not thaw_moves.empty
                else np.nan,
                "Is_education": bool(item_df["Is_education"].iloc[0]),
            }
        )

    result = cast(
        pd.DataFrame,
        pd.DataFrame(rows).sort_values(
            by=["PSI_pct", "Avg_weight"], ascending=[False, False]
        ),
    )
    return cast(pd.DataFrame, result.reset_index(drop=True))


def compute_yearly_summary(monthly_summary: pd.DataFrame) -> pd.DataFrame:
    yearly = monthly_summary.groupby("Year", as_index=False).agg(
        Avg_monthly_FDI_weighted_pct=("FDI_weighted_pct", "mean"),
        Avg_monthly_FDI_simple_pct=("FDI_simple_pct", "mean"),
        Active_base_pct=("Active_base_pct", "mean"),
        Headline_MoM=("Headline_MoM", "mean"),
        Included_MoM=("Included_MoM", "mean"),
    )
    return cast(pd.DataFrame, yearly)


def compute_exclusions_summary(
    level5: pd.DataFrame, micro_sprav: pd.DataFrame, item_names: pd.DataFrame
) -> pd.DataFrame:
    names_map = item_names.drop_duplicates(subset=["Item_code"]).set_index("Item_code")[
        "Item_name"
    ]
    micro_codes = set(micro_sprav["Item_code"])
    level5_codes = set(level5["Item_code"])

    rows = []
    for item_code, reason in EXCLUSION_RULES.items():
        rows.append(
            {
                "Item_code": item_code,
                "Item_name": names_map.get(item_code, ""),
                "Reason": reason,
                "In_micro_sprav": item_code in micro_codes,
                "In_level5_scope": item_code in level5_codes,
            }
        )

    return cast(
        pd.DataFrame,
        pd.DataFrame(rows).sort_values("Item_code").reset_index(drop=True),
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: Figure, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def chart_yearly_psi(
    yearly_summary: pd.DataFrame,
    item_summary: pd.DataFrame,
    component_monthly: pd.DataFrame,
) -> None:
    if yearly_summary.empty:
        return

    years = yearly_summary["Year"].to_numpy()
    weighted = yearly_summary["Avg_monthly_FDI_weighted_pct"].to_numpy()
    simple = yearly_summary["Avg_monthly_FDI_simple_pct"].to_numpy()
    active = 100.0 - weighted

    component_monthly = component_monthly.copy()
    component_colors = _build_component_palette(
        cast(pd.Series, component_monthly["Component"])
    )
    component_yearly = component_monthly.groupby(
        ["Year", "Component"], as_index=False
    ).agg(Avg_FDI_weighted_pct=("FDI_weighted_pct", "mean"))
    component_yearly_values = cast(pd.Series, component_yearly["Avg_FDI_weighted_pct"])
    component_yearly = cast(
        pd.DataFrame,
        component_yearly.loc[pd.notna(component_yearly_values)],
    )
    component_pivot = component_yearly.pivot(
        index="Year",
        columns="Component",
        values="Avg_FDI_weighted_pct",
    )
    component_mean = cast(pd.Series, component_pivot.mean(axis=0))
    top_components = component_mean.sort_values(ascending=False).head(4).index.tolist()

    item_score = cast(
        pd.DataFrame,
        item_summary.sort_values("Weighted_PSI_pct", ascending=False).head(12)[
            ["Item_name", "Weighted_PSI_pct", "Component", "Avg_weight", "PSI_pct"]
        ],
    )
    component_payload = ""
    if not item_score.empty:
        head_names = [_short_label(name, 36) for name in item_score["Item_name"][:6]]
        head_vals = [f"{value:.1f}%" for value in item_score["Weighted_PSI_pct"][:6]]
        component_payload = " | ".join(
            f"{name}: {val}" for name, val in zip(head_names, head_vals)
        )

    fig = plt.figure(figsize=(18, 10), facecolor=_PLOT_BG)
    layout = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.5, 1.0],
        width_ratios=[2.0, 1.0],
        hspace=0.35,
        wspace=0.30,
    )

    ax_main = fig.add_subplot(layout[0, :])
    _decorate_axis(ax_main)
    ax_main.plot(
        years,
        weighted,
        marker="o",
        linewidth=2.6,
        color=_PLOT_FROZEN,
        label="Взвешенная доля заморозки",
    )
    ax_main.plot(
        years,
        simple,
        marker="o",
        linewidth=1.9,
        linestyle="--",
        color=_PLOT_SIMPLE,
        alpha=0.95,
        label="Простая доля заморозки",
    )
    ax_main.plot(
        years,
        active,
        marker="o",
        linewidth=1.8,
        color=_PLOT_ACTIVE,
        label="Свободная база (100 − взвешенная доля)",
    )
    ax_main.fill_between(
        years,
        0,
        active,
        color=_PLOT_ACTIVE,
        alpha=0.16,
        label="Активная база",
    )
    ax_main.set_xlabel("Год")
    ax_main.set_ylabel("Доля, %")
    ax_main.set_xticks(years)
    ax_main.set_ylim(0, max(active.max(), weighted.max()) + 8)
    ax_main.legend(loc="upper left", frameon=True)

    if weighted.size > 0:
        _annotate_point(
            ax_main,
            years[-1],
            weighted[-1],
            f"t={years[-1]}: {weighted[-1]:.1f}%",
            color=_PLOT_FROZEN,
        )
    if simple.size > 0:
        _annotate_point(
            ax_main,
            years[0],
            simple[0],
            f"{int(years[0])}: {simple[0]:.1f}%",
            color=_PLOT_SIMPLE,
        )

    if len(component_pivot) > 0 and top_components:
        ax_component = fig.add_subplot(layout[1, 0])
        _decorate_axis(ax_component)
        for component in top_components:
            component_name = str(component)
            y = cast(pd.Series, component_pivot[component_name].reindex(years))
            color = component_colors.get(component_name, _PLOT_ACCENT)
            ax_component.plot(
                years,
                y,
                marker="o",
                linewidth=1.9,
                label=component_name,
                color=color,
            )
            if not y.dropna().empty:
                y_non_null = y.dropna()
                last_x = float(np.asarray(y_non_null.index.to_numpy())[-1])
                last_y = float(y_non_null.iloc[-1])
                _annotate_point(
                    ax_component,
                    last_x,
                    last_y,
                    f"{last_y:.1f}%",
                    color=color,
                )
        ax_component.set_title("Компонентная декомпозиция, среднее за год", fontsize=11)
        ax_component.set_xlabel("Год")
        ax_component.set_ylabel("Доля, %")
        ax_component.set_xticks(years)
        ax_component.legend(loc="upper left", frameon=True)
    else:
        ax_component = fig.add_subplot(layout[1, 0])
        _decorate_axis(ax_component)
        ax_component.axis("off")

    ax_stats = fig.add_subplot(layout[1, 1])
    _decorate_axis(ax_stats)
    ax_stats.axis("off")

    if not yearly_summary.empty:
        start_weighted = float(yearly_summary["Avg_monthly_FDI_weighted_pct"].iloc[0])
        end_weighted = float(yearly_summary["Avg_monthly_FDI_weighted_pct"].iloc[-1])
        start_simple = float(yearly_summary["Avg_monthly_FDI_simple_pct"].iloc[0])
        end_simple = float(yearly_summary["Avg_monthly_FDI_simple_pct"].iloc[-1])
        start_active = 100.0 - start_weighted
        end_active = 100.0 - end_weighted
        delta_weighted = end_weighted - start_weighted
        delta_simple = end_simple - start_simple
        delta_active = end_active - start_active

        stats_lines = [
            f"Окно наблюдений: {yearly_summary['Year'].iloc[0]}–{yearly_summary['Year'].iloc[-1]}",
            "",
            f"Взвешенная доля заморозки: {start_weighted:.1f}% → {end_weighted:.1f}% ({delta_weighted:+.1f} п.п.)",
            f"Простая доля заморозки: {start_simple:.1f}% → {end_simple:.1f}% ({delta_simple:+.1f} п.п.)",
            f"Свободная база: {start_active:.1f}% → {end_active:.1f}% ({delta_active:+.1f} п.п.)",
            "",
            "Топ-6 по взвешенной доле заморозки:",
            component_payload,
        ]
        for index, line in enumerate(stats_lines):
            ax_stats.text(
                0.02,
                0.95 - index * 0.13,
                line,
                transform=ax_stats.transAxes,
                fontsize=10,
                color=_PLOT_DARK,
                fontweight="bold" if index in [0, 2, 3, 4] else "normal",
            )
    ax_main.set_title(
        "Уровень замороженных цен (Уровень-5): динамика и компонентная декомпозиция"
    )
    ax_main.set_xlabel("Год")

    save_figure(
        fig,
        ASSETS_CHARTS_DIR / "final_freeze_level5_psi_yearly.png",
        LEGACY_PLOTS_DIR / "psi_yearly.png",
    )


def chart_top_bottom_items(item_summary: pd.DataFrame) -> None:
    top_count = 14
    bottom_count = 14
    component_colors = _build_component_palette(
        cast(pd.Series, item_summary["Component"])
    )
    weighted_sorted = item_summary.sort_values("Weighted_PSI_pct", ascending=False)
    top = weighted_sorted.head(top_count).copy().iloc[::-1]
    bottom = weighted_sorted.tail(bottom_count).copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 10),
        sharey=False,
        gridspec_kw={"width_ratios": [1.5, 1.5, 1.0]},
    )

    for ax in axes:
        _decorate_axis(ax)

    weighted_median = float(item_summary["Weighted_PSI_pct"].median())
    weighted_mean = float(item_summary["Weighted_PSI_pct"].mean())
    top_xlim = float(top["Weighted_PSI_pct"].max()) if not top.empty else 10.0
    top_xlim = max(8.0, top_xlim * 1.08)
    bottom_xlim = float(bottom["Weighted_PSI_pct"].max()) if not bottom.empty else 2.0
    bottom_xlim = max(1.0, bottom_xlim * 1.15)

    axes[0].set_xlim(0, min(100.0, top_xlim))
    axes[0].set_title("Товары с максимальной долей заморозки")
    axes[0].set_xlabel("Взвешенная доля заморозки, %")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].text(
        0.02,
        0.97,
        f"Медиана по выборке: {weighted_median:.1f}%",
        transform=axes[0].transAxes,
        fontsize=8,
        color=_PLOT_DARK,
    )
    axes[0].axvline(
        weighted_median,
        linestyle="--",
        color="#888",
        label="Медиана",
    )
    axes[0].legend(loc="lower right", fontsize=8)

    for y, (_, row) in enumerate(top.reset_index(drop=True).iterrows()):
        comp = _normalize_component_name(row["Component"])
        color = component_colors.get(comp, _PLOT_FROZEN)
        value = float(row["Weighted_PSI_pct"])
        axes[0].barh(y, value, color=color, alpha=0.9)
        axes[0].text(
            value + max(0.4, top_xlim * 0.015),
            y,
            f"{value:.1f}% (простая: {row['PSI_pct']:.1f}%)",
            va="center",
            fontsize=8,
            color=_PLOT_DARK,
        )
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(
        [_short_label(name, max_len=44) for name in top["Item_name"]]
    )

    for i, row in bottom.reset_index(drop=True).iterrows():
        comp = _normalize_component_name(row["Component"])
        color = component_colors.get(comp, _PLOT_SIMPLE)
        value = float(row["Weighted_PSI_pct"])
        axes[1].barh(i, value, color=color, alpha=0.9)
        axes[1].text(
            value + max(0.1, bottom_xlim * 0.05),
            i,
            f"{value:.1f}% (простая: {row['PSI_pct']:.1f}%)",
            va="center",
            fontsize=8,
            color=_PLOT_DARK,
        )
    axes[1].set_yticks(range(len(bottom)))
    axes[1].set_yticklabels(
        [_short_label(name, max_len=44) for name in bottom["Item_name"]]
    )
    axes[1].set_title("Товары с минимальной долей заморозки")
    axes[1].set_xlabel("Взвешенная доля заморозки, %")
    axes[1].set_xlim(0, max(2.0, bottom_xlim))
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].text(
        0.02,
        0.97,
        f"Среднее по выборке: {weighted_mean:.1f}%",
        transform=axes[1].transAxes,
        fontsize=8,
        color=_PLOT_DARK,
    )

    comp_weights = weighted_sorted.head(30).copy()
    comp_weights["Component"] = comp_weights["Component"].map(_normalize_component_name)
    comp_weights = cast(
        pd.Series, comp_weights.groupby("Component")["Avg_weight"].sum()
    )
    comp_weights = comp_weights.sort_values(ascending=True)
    top_comp = comp_weights.tail(8)
    axes[2].barh(
        [_short_label(str(name), max_len=16) for name in top_comp.index],
        top_comp.values,
        color=[
            component_colors.get(str(name), _PLOT_ACCENT) for name in top_comp.index
        ],
    )
    for index, (name, value) in enumerate(top_comp.items()):
        axes[2].text(
            value + value * 0.02,
            index,
            f"{value:.4f}",
            va="center",
            fontsize=8,
        )
    axes[2].set_title("Весовая структура (топ-30 по взвешенной доле заморозки)")
    axes[2].set_xlabel("Суммарный вес в корзине")
    axes[2].set_ylabel("Компонент")
    if not top_comp.empty:
        axes[2].set_xlim(0.0, float(top_comp.max()) * 1.18)

    legend_handles = [
        Line2D([0], [0], color=color, lw=8) for color in component_colors.values()
    ]
    legend_labels = [str(label) for label in component_colors.keys()]
    axes[2].legend(
        legend_handles,
        legend_labels,
        title="Компоненты",
        loc="upper right",
        fontsize=8,
    )

    axes[2].text(
        0.02,
        0.05,
        "Размер столбца — доля веса выбранных статей в корзине.",
        fontsize=8,
        color="#555555",
        transform=axes[2].transAxes,
    )

    fig.tight_layout()

    save_figure(
        fig,
        ASSETS_CHARTS_DIR / "final_freeze_level5_top_bottom.png",
        LEGACY_PLOTS_DIR / "psi_top_bottom.png",
    )


def chart_freeze_diffusion(monthly_summary: pd.DataFrame) -> None:
    if monthly_summary.empty:
        return
    monthly = monthly_summary.sort_values("Date").copy()
    monthly["Date_ord"] = pd.to_datetime(monthly["Date"])
    monthly["Diff_to_active"] = monthly["Implied_active_MoM"] - monthly["Headline_MoM"]
    monthly["Implied_boost"] = np.where(
        monthly["Diff_to_active"].isna(), np.nan, monthly["Diff_to_active"]
    )
    monthly["year"] = monthly["Date_ord"].dt.year

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex="col")
    for ax in axes.flat:
        _decorate_axis(ax)

    axes[0, 0].plot(
        monthly["Date_ord"],
        monthly["FDI_weighted_pct"],
        color=_PLOT_FROZEN,
        linewidth=2,
        label="Взвешенная доля заморозки",
    )
    axes[0, 0].plot(
        monthly["Date_ord"],
        monthly["FDI_simple_pct"],
        color=_PLOT_SIMPLE,
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label="Простая доля заморозки",
    )
    axes[0, 0].plot(
        monthly["Date_ord"],
        monthly["Active_base_pct"],
        color=_PLOT_ACTIVE,
        linewidth=2,
        label="Активная база",
    )
    axes[0, 0].set_title("Структура замороженных цен и свободной базы")
    axes[0, 0].set_ylabel("Доля, %")
    axes[0, 0].text(
        0.02,
        0.96,
        "Слева: взвешенная доля заморозки (по весам), простая доля замороженности (по числу позиций), свободная база = 100 − взвешенная доля.",
        fontsize=8,
        color=_PLOT_DARK,
        transform=axes[0, 0].transAxes,
    )
    axes[0, 0].legend(loc="upper right", frameon=True)

    if not monthly.empty:
        idx_min = monthly["FDI_weighted_pct"].idxmin()
        idx_max = monthly["FDI_weighted_pct"].idxmax()
        axes[0, 0].scatter(
            monthly.loc[[idx_min, idx_max], "Date_ord"],
            monthly.loc[[idx_min, idx_max], "FDI_weighted_pct"],
            s=60,
            color=_PLOT_DARK,
            zorder=5,
        )
        _annotate_point(
            axes[0, 0],
            monthly.loc[idx_min, "Date_ord"],
            monthly.loc[idx_min, "FDI_weighted_pct"],
            f"минимум {monthly.loc[idx_min, 'FDI_weighted_pct']:.1f}%",
            color=_PLOT_DARK,
        )
        _annotate_point(
            axes[0, 0],
            monthly.loc[idx_max, "Date_ord"],
            monthly.loc[idx_max, "FDI_weighted_pct"],
            f"максимум {monthly.loc[idx_max, 'FDI_weighted_pct']:.1f}%",
            color=_PLOT_DARK,
        )

    axes[0, 1].plot(
        monthly["Date_ord"],
        monthly["Headline_MoM"],
        color=_PLOT_DARK,
        linewidth=1.6,
        label="ИПЦ (включенный, месяц к месяцу)",
    )
    axes[0, 1].plot(
        monthly["Date_ord"],
        monthly["Implied_active_MoM"],
        color=_PLOT_ACCENT,
        linewidth=2,
        label="Индикатор после очистки",
    )
    axes[0, 1].set_title("Индекс в активной части корзины")
    axes[0, 1].set_ylabel("Индекс, месяц к месяцу")
    axes[0, 1].text(
        0.02,
        0.96,
        "Индикатор свободной базы = 100 + (ИПЦ к прошлому месяцу − 100) / (свободная база/100)",
        fontsize=8,
        color=_PLOT_DARK,
        transform=axes[0, 1].transAxes,
    )
    axes[0, 1].legend(loc="upper right", frameon=True)

    yearly = (
        monthly.groupby("year")
        .agg(
            Avg_FDI=("FDI_weighted_pct", "mean"),
            Avg_Active=("Active_base_pct", "mean"),
        )
        .reset_index()
    )
    x = np.arange(len(yearly["year"]))
    bars = 0.34
    axes[1, 0].bar(
        x - bars,
        yearly["Avg_FDI"],
        width=bars,
        color=_PLOT_FROZEN,
        alpha=0.9,
        label="Взвешенная доля заморозки",
    )
    axes[1, 0].bar(
        x,
        yearly["Avg_Active"],
        width=bars,
        color=_PLOT_ACTIVE,
        alpha=0.8,
        label="Активная база",
    )
    axes[1, 0].set_title("Агрегация по годам")
    axes[1, 0].set_ylabel("Средняя доля, %")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(yearly["year"])
    axes[1, 0].text(
        0.02,
        0.96,
        "Эта колонка показывает средние значения: чем выше взвешенная доля заморозки, тем меньше свободная база (100 − взвешенная доля).",
        fontsize=8,
        color="#666",
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].legend(loc="upper right", frameon=True)

    valid = monthly.dropna(subset=["Implied_active_MoM", "FDI_weighted_pct"]).copy()
    if valid.empty:
        return
    scatter = axes[1, 1].scatter(
        valid["FDI_weighted_pct"],
        valid["Implied_boost"],
        c=valid["year"],
        cmap="tab20",
        s=26,
        alpha=0.85,
    )
    cbar = plt.colorbar(scatter, ax=axes[1, 1], label="Год")
    cbar.ax.tick_params(labelsize=8)
    axes[1, 1].axhline(0.0, color=_PLOT_DARK, linewidth=0.8)
    axes[1, 1].set_title("Смещение между наблюдаемым и очищенным индексом")
    axes[1, 1].set_xlabel("Взвешенная доля заморозки, %")
    axes[1, 1].set_ylabel("Сдвиг индекса активной базы, п.п.")

    if len(valid) > 1:
        idx_strong = valid["Implied_boost"].abs().idxmax()
        axes[1, 1].scatter(
            valid.loc[idx_strong, "FDI_weighted_pct"],
            valid.loc[idx_strong, "Implied_boost"],
            s=85,
            color=_PLOT_DARK,
            zorder=6,
        )
        _annotate_point(
            axes[1, 1],
            valid.loc[idx_strong, "FDI_weighted_pct"],
            valid.loc[idx_strong, "Implied_boost"],
            f"{valid.loc[idx_strong, 'Implied_boost']:.2f}",
            color=_PLOT_DARK,
        )

    fig.suptitle(
        "Диффузионный индекс (Уровень-5): скрытая инфляция через свободную базу"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(
        fig,
        ASSETS_CHARTS_DIR / "final_freeze_level5_diffusion.png",
        LEGACY_PLOTS_DIR / "freeze_diffusion_index.png",
    )


def chart_freeze_seasonality(
    monthly_summary: pd.DataFrame, component_monthly: pd.DataFrame
) -> None:
    if monthly_summary.empty:
        return

    seasonality = monthly_summary.groupby("Month", as_index=False).agg(
        FDI_weighted_pct=("FDI_weighted_pct", "mean"),
        FDI_simple_pct=("FDI_simple_pct", "mean"),
        Avg_abs_MoM_pct=("Avg_abs_MoM_pct", "mean"),
    )
    seasonality = seasonality.set_index("Month").reindex(range(1, 13)).reset_index()
    months = [_MONTH_NAMES[m - 1] for m in seasonality["Month"]]

    component_input = cast(
        pd.DataFrame,
        component_monthly.groupby(["Month", "Component"], as_index=False).agg(
            freeze_weight=("freeze_weight", "sum"),
        ),
    )
    component_input["month_total"] = component_input.groupby("Month")[
        "freeze_weight"
    ].transform("sum")
    component_input["Share_of_freeze"] = np.where(
        component_input["month_total"] > 0,
        component_input["freeze_weight"] / component_input["month_total"] * 100.0,
        0.0,
    )
    component_monthly_pivot = (
        component_input.pivot(
            index="Month",
            columns="Component",
            values="Share_of_freeze",
        )
        .reindex(range(1, 13))
        .fillna(0)
    )
    component_colors = _build_component_palette(
        pd.Series(component_monthly_pivot.columns.astype(str).tolist())
    )
    component_share = cast(pd.Series, component_monthly_pivot.mean(axis=0)).sort_values(
        ascending=False
    )
    top_components = component_share.head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for ax in axes.flat:
        _decorate_axis(ax)

    x = np.arange(1, 13)
    width = 0.37
    axes[0, 0].bar(
        x - width / 2,
        seasonality["FDI_weighted_pct"],
        width=width,
        color=_PLOT_FROZEN,
        alpha=0.95,
        label="Взвешенная доля заморозки",
    )
    axes[0, 0].bar(
        x + width / 2,
        seasonality["FDI_simple_pct"],
        width=width,
        color=_PLOT_SIMPLE,
        alpha=0.85,
        label="Простая доля заморозки",
    )
    axes[0, 0].set_title("Сезонная структура замороженности")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(months)
    axes[0, 0].set_ylabel("Доля, %")
    axes[0, 0].legend(loc="upper right", frameon=True)
    for i, value in enumerate(seasonality["FDI_weighted_pct"]):
        axes[0, 0].text(x[i], value + 0.4, f"{value:.1f}", ha="center", fontsize=8)

    axes[0, 1].plot(
        x,
        seasonality["Avg_abs_MoM_pct"],
        color=_PLOT_ACCENT,
        marker="o",
        linewidth=2,
    )
    axes[0, 1].set_title("Сезонная амплитуда по уровню цен (взвешенная)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(months)
    axes[0, 1].set_ylabel("|ИПЦ к прошлому месяцу − 100|, %")

    if top_components:
        bottom = np.zeros(len(component_monthly_pivot))
        for comp in top_components:
            values = component_monthly_pivot[comp].to_numpy()
            axes[1, 0].bar(
                x,
                values,
                width=0.75,
                bottom=bottom,
                color=component_colors.get(str(comp), _PLOT_ACCENT),
                alpha=0.88,
                label=str(comp),
            )
            bottom = bottom + values
        axes[1, 0].set_title("Компонентная структура замороженности по месяцам (топ-4)")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(months)
        axes[1, 0].set_ylabel("Доля замороженного веса, %")
        axes[1, 0].legend(loc="upper left", frameon=True, fontsize=8)
    else:
        axes[1, 0].axis("off")

    weighted_series = seasonality["FDI_weighted_pct"]
    peak_idx = cast(int, weighted_series.idxmax())
    trough_idx = cast(int, weighted_series.idxmin())
    peak_month = seasonality.loc[peak_idx, "Month"]
    trough_month = seasonality.loc[trough_idx, "Month"]
    peak_value = float(seasonality.loc[peak_idx, "FDI_weighted_pct"])
    trough_value = float(seasonality.loc[trough_idx, "FDI_weighted_pct"])
    stats_panel = (
        f"Пик: {_MONTH_NAMES[int(peak_month) - 1]} ({_format_pct(peak_value)})\n"
        f"Спад: {_MONTH_NAMES[int(trough_month) - 1]} ({_format_pct(trough_value)})\n"
        f"Разрыв пика/дна: {_format_pct(seasonality['FDI_weighted_pct'].max() - seasonality['FDI_weighted_pct'].min())}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.96,
        "Сезонная сводка",
        fontsize=12,
        fontweight="bold",
        color=_PLOT_DARK,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].text(
        0.05,
        0.67,
        stats_panel,
        fontsize=10,
        color=_PLOT_DARK,
        transform=axes[1, 1].transAxes,
    )

    top_month = peak_month
    bottom_month = trough_month
    axes[1, 1].text(
        0.05,
        0.36,
        "Ключевые окна:",
        fontsize=10,
        fontweight="bold",
        color=_PLOT_DARK,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].text(
        0.05,
        0.25,
        f"макс. концентрация: {_MONTH_NAMES[int(top_month) - 1]}",
        fontsize=10,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].text(
        0.05,
        0.16,
        f"минимальная концентрация: {_MONTH_NAMES[int(bottom_month) - 1]}",
        fontsize=10,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].text(
        0.05,
        0.07,
        "Сезонность строится по средним значениям долей замороженности для каждого месяца за весь период.",
        fontsize=9,
        style="italic",
        color="#5a5a5a",
        transform=axes[1, 1].transAxes,
    )

    fig.suptitle("Сезонный профиль заморозки (Уровень-5): где и как меняется доля")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(
        fig,
        ASSETS_CHARTS_DIR / "final_freeze_level5_seasonality.png",
        LEGACY_PLOTS_DIR / "freeze_seasonality.png",
    )


def chart_education_profile(education_summary: pd.DataFrame) -> None:
    if education_summary.empty:
        return

    plot_df = education_summary.sort_values("Weighted_PSI_pct", ascending=False).head(
        12
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), width_ratios=[1.6, 1])

    _decorate_axis(axes[0])
    axes[0].barh(
        plot_df["Item_name"].map(lambda x: _short_label(x, 64)),
        plot_df["Weighted_PSI_pct"],
        color=_PLOT_ACCENT,
        alpha=0.95,
    )
    for i, row in plot_df.reset_index(drop=True).iterrows():
        axes[0].text(
            row["Weighted_PSI_pct"] + 0.5,
            i,
            f"{row['Weighted_PSI_pct']:.1f}% (простая: {row['PSI_pct']:.1f}%)",
            va="center",
            fontsize=8,
        )
    axes[0].set_title("Образовательные услуги: приоритетный мониторинг")
    axes[0].set_xlabel("Взвешенная доля заморозки, %")

    axes[1].axis("off")
    comp_weight = plot_df["Avg_weight"].sum()
    max_weight = plot_df["Avg_weight"].max()
    lines = [
        "Срез образовательных услуг для отдельного мониторинга:",
        f"Объектов на контроле: {len(plot_df)}",
        f"Суммарный вес: {comp_weight:.4f}",
        f"Макс. вес: {max_weight:.4f}",
        f"Диапазон простой доли заморозки: {plot_df['PSI_pct'].min():.1f}% — {plot_df['PSI_pct'].max():.1f}%",
    ]
    for index, line in enumerate(lines):
        axes[1].text(
            0.02,
            0.85 - index * 0.14,
            line,
            color=_PLOT_DARK,
            fontsize=10,
            fontweight="bold" if index == 0 else "normal",
            transform=axes[1].transAxes,
        )
    axes[1].text(
        0.02,
        0.2,
        "Эти позиции выделяются отдельной группой,\nчтобы отслеживать динамику без смешивания с общей структурой.",
        fontsize=9,
        color="#666",
        transform=axes[1].transAxes,
    )

    save_figure(fig, ASSETS_CHARTS_DIR / "final_freeze_level5_education.png")


def write_report(
    item_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    exclusions_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
) -> Path:
    report_path = ARCHIVE_RESULTS_DIR / "final_freeze_level5_report.md"
    latest = monthly_summary.iloc[-1]
    effective_exclusions = cast(
        pd.DataFrame,
        exclusions_summary[exclusions_summary["In_level5_scope"]].copy(),
    )
    strict_top = cast(
        pd.DataFrame, item_summary.head(5).loc[:, ["Item_code", "Item_name", "PSI_pct"]]
    )
    flexible_top = cast(
        pd.DataFrame, item_summary.tail(5).loc[:, ["Item_code", "Item_name", "PSI_pct"]]
    )
    yearly_report = cast(
        pd.DataFrame,
        yearly_summary.rename(
            columns={
                "Year": "Год",
                "Avg_monthly_FDI_weighted_pct": "Средняя взвешенная доля заморозки, %",
                "Avg_monthly_FDI_simple_pct": "Средняя простая доля заморозки, %",
                "Active_base_pct": "Активная база, %",
                "Headline_MoM": "ИПЦ, месяц к месяцу",
                "Included_MoM": "Индекс активной части корзины",
            }
        ),
    )
    strict_top_report = cast(
        pd.DataFrame,
        strict_top.rename(
            columns={
                "Item_code": "Код",
                "Item_name": "Позиция",
                "PSI_pct": "Доля месяцев с заморозкой, %",
            }
        ),
    )
    flexible_top_report = cast(
        pd.DataFrame,
        flexible_top.rename(
            columns={
                "Item_code": "Код",
                "Item_name": "Позиция",
                "PSI_pct": "Доля месяцев с заморозкой, %",
            }
        ),
    )
    education_report = cast(
        pd.DataFrame,
        education_summary.head(10)
        .loc[:, ["Item_code", "Item_name", "PSI_pct", "Avg_weight"]]
        .rename(
            columns={
                "Item_code": "Код",
                "Item_name": "Позиция",
                "PSI_pct": "Доля месяцев с заморозкой, %",
                "Avg_weight": "Средний вес",
            }
        ),
    )
    documented_exclusion_codes = ", ".join(
        str(code) for code in exclusions_summary["Item_code"].tolist()
    )
    effective_exclusion_codes = ", ".join(
        str(code) for code in effective_exclusions["Item_code"].tolist()
    )

    lines = [
        "# Итоговый анализ заморозки цен по Level-5",
        "",
        "## Контур анализа и источники",
        "",
        "- Контур Level-5 построен по `data/raw/micro_sprav.csv` и согласован с подтверждениями из `docs/LEVEL5_FREEZE_EXCLUSIONS.md`.",
        "- Точные исключения взяты из `docs/LEVEL5_FREEZE_EXCLUSIONS.md`: только нотариальные и ЖКХ позиции Level-5.",
        "- Широкие агрегаты и укрупнённые группы убраны из рабочей корзины и не смешиваются с Level-5 позициями.",
        "- Образовательные услуги не исключаются из корзины, а вынесены в отдельный мониторинговый блок.",
        "",
        "## Рабочая корзина",
        "",
        f"- Период наблюдений: {monthly_summary['Date'].min().date()} — {monthly_summary['Date'].max().date()}",
        f"- Позиции Level-5 после исключений: {item_summary['Item_code'].nunique()}",
        f"- Документированный список кодов на исключение: {documented_exclusion_codes}",
        f"- Исключения, реально присутствовавшие в Level-5 и удалённые из расчёта: {effective_exclusions['Item_code'].nunique()} ({effective_exclusion_codes})",
        f"- Последняя взвешенная доля заморозки: {latest['FDI_weighted_pct']:.2f}%",
        f"- Последняя активная база: {latest['Active_base_pct']:.2f}%",
        f"- Последний общий ИПЦ (месяц к месяцу): {latest['Headline_MoM']:.2f}",
        f"- Последний индекс активной части корзины: {latest['Implied_active_MoM']:.2f}",
        "",
        "## Средние годовые показатели заморозки",
        "",
        "Таблица ниже показывает, как в среднем по каждому году менялась доля замороженных позиций и размер активной части корзины.",
        "",
        dataframe_to_markdown(yearly_report),
        "",
        "## Наиболее жёсткие позиции Level-5",
        "",
        "Это позиции, у которых заморозка встречалась чаще всего на протяжении наблюдаемого периода.",
        "",
        dataframe_to_markdown(strict_top_report),
        "",
        "## Наименее жёсткие позиции Level-5",
        "",
        "Это позиции, в которых заморозка почти не наблюдалась, а цены двигались свободнее.",
        "",
        dataframe_to_markdown(flexible_top_report),
        "",
        "## Образовательные услуги: отдельный блок мониторинга",
        "",
        f"- В отдельном мониторинге оставлено позиций: {education_summary['Item_code'].nunique()}",
        "- Этот блок нужен, чтобы следить за образовательными услугами отдельно и не смешивать их со списком исключений ЖКХ и нотариальных услуг.",
        "",
        dataframe_to_markdown(education_report),
        "",
        "## Сформированные артефакты",
        "",
        "- Основной CSV: `archive/results/final_freeze_level5_monthly.csv`",
        "- Основной CSV: `archive/results/final_freeze_level5_items.csv`",
        "- Основной CSV: `archive/results/final_freeze_level5_yearly.csv`",
        "- Основной CSV: `archive/results/final_freeze_level5_exclusions.csv`",
        "- Основной CSV: `archive/results/final_freeze_level5_education.csv`",
        "- Основные диаграммы: `assets/charts/final_freeze_level5_*.png`",
        "- CSV для совместимости со старым контуром: `data/freeze_diffusion_index.csv`",
        "- Диаграммы для совместимости со старым контуром: `data/plots/psi_yearly.png`, `data/plots/psi_top_bottom.png`, `data/plots/freeze_diffusion_index.png`, `data/plots/freeze_seasonality.png`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_final_freeze_analysis() -> AnalysisOutputs:
    monthly, weights, item_names, micro_sprav = load_freeze_inputs()
    level5 = build_level5_scope(monthly, weights, item_names, micro_sprav)
    included, _ = apply_exclusions(level5)
    included = add_freeze_flags(included)
    included["Is_education"] = is_education_item(included)

    monthly_summary = compute_monthly_summary(included, monthly)
    component_monthly = compute_component_monthly_summary(included)
    item_summary = compute_item_summary(included)
    yearly_summary = compute_yearly_summary(monthly_summary)
    exclusions_summary = compute_exclusions_summary(level5, micro_sprav, item_names)
    education_summary = cast(
        pd.DataFrame, item_summary[item_summary["Is_education"]].copy()
    )

    write_csv(item_summary, ARCHIVE_RESULTS_DIR / "final_freeze_level5_items.csv")
    write_csv(monthly_summary, ARCHIVE_RESULTS_DIR / "final_freeze_level5_monthly.csv")
    write_csv(yearly_summary, ARCHIVE_RESULTS_DIR / "final_freeze_level5_yearly.csv")
    write_csv(
        exclusions_summary, ARCHIVE_RESULTS_DIR / "final_freeze_level5_exclusions.csv"
    )
    write_csv(
        education_summary, ARCHIVE_RESULTS_DIR / "final_freeze_level5_education.csv"
    )
    write_csv(
        cast(
            pd.DataFrame,
            monthly_summary.loc[
                :,
                [
                    "Date",
                    "FDI_weighted_pct",
                    "FDI_simple_pct",
                    "Active_base_pct",
                    "Headline_MoM",
                    "Implied_active_MoM",
                ],
            ],
        ).rename(columns={"Headline_MoM": "MoM"}),
        DATA_DIR / "freeze_diffusion_index.csv",
    )

    chart_yearly_psi(yearly_summary, item_summary, component_monthly)
    chart_top_bottom_items(item_summary)
    chart_freeze_diffusion(monthly_summary)
    chart_freeze_seasonality(monthly_summary, component_monthly)
    chart_education_profile(education_summary)
    report_path = write_report(
        item_summary,
        monthly_summary,
        yearly_summary,
        exclusions_summary,
        education_summary,
    )

    return AnalysisOutputs(
        item_summary=item_summary,
        monthly_summary=monthly_summary,
        yearly_summary=yearly_summary,
        exclusions_summary=exclusions_summary,
        education_summary=education_summary,
        report_path=report_path,
    )
