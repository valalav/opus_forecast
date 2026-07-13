from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_INDEX_COLUMNS = {"Date", "Region_code", "Item_code", "MoM", "YoY"}
REQUIRED_WEIGHT_COLUMNS = {"Day", "Region_code", "Item_code", "Weight_gross"}


class LoaderError(ValueError):
    pass


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"(?<=\d)\s+(?=\d)", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_csv_flexible(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read comma or semicolon CSV files and normalize space-padded decimals."""
    path = Path(path)
    if "sep" not in kwargs:
        with path.open("r", encoding=kwargs.get("encoding", "utf-8")) as handle:
            sample = handle.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            kwargs["sep"] = dialect.delimiter
        except csv.Error:
            kwargs["sep"] = ";" if sample.count(";") >= sample.count(",") else ","
    df = pd.read_csv(path, **kwargs)
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    for column in df.columns:
        if df[column].dtype == object:
            coerced = _coerce_numeric(df[column])
            if coerced.notna().sum() > 0 and coerced.notna().sum() >= df[column].notna().sum() * 0.8:
                df[column] = coerced
    return df


def read_csv_readonly(path: str | Path, **kwargs) -> pd.DataFrame:
    return read_csv_flexible(path, **kwargs)


def load_indices(path: str | Path, region_code: int = 7) -> pd.DataFrame:
    df = read_csv_flexible(path)
    missing = REQUIRED_INDEX_COLUMNS - set(df.columns)
    if missing:
        raise LoaderError(f"indices file missing columns: {sorted(missing)}")
    df = df.loc[df["Region_code"].eq(region_code), ["Date", "Item_code", "MoM", "YoY"]].copy()
    df["date"] = pd.to_datetime(df.pop("Date"), errors="coerce")
    df["item_code"] = pd.to_numeric(df.pop("Item_code"), errors="coerce").astype("Int64")
    df["mom_index"] = _coerce_numeric(df.pop("MoM"))
    df["yoy_index"] = _coerce_numeric(df.pop("YoY"))
    df = df.dropna(subset=["date", "item_code", "mom_index"])
    return df.sort_values(["date", "item_code"]).reset_index(drop=True)


def load_long_item_indices(path: str | Path, region_code: int = 7) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    missing = REQUIRED_INDEX_COLUMNS - set(raw.columns)
    if missing:
        raise LoaderError(f"indices file missing columns: {sorted(missing)}")
    frame = raw.loc[raw["Region_code"].eq(region_code), ["Date", "Region_code", "Item_code", "MoM", "YoY"]].copy()
    frame["date"] = pd.to_datetime(frame.pop("Date"), errors="coerce")
    frame["region_code"] = frame.pop("Region_code").astype(int)
    frame["item_code"] = pd.to_numeric(frame.pop("Item_code"), errors="coerce").astype("Int64")
    frame["mom_index"] = _coerce_numeric(frame.pop("MoM"))
    frame["yoy_index"] = _coerce_numeric(frame.pop("YoY"))
    return frame.dropna(subset=["date", "item_code", "mom_index"]).reset_index(drop=True)


def load_wide_component_indices(path: str | Path) -> pd.DataFrame:
    df = read_csv_flexible(path)
    if "Код" not in df.columns or "Товар" not in df.columns:
        raise LoaderError("wide component file must contain Код and Товар")
    month_cols = [col for col in df.columns if isinstance(col, str) and len(col) == 7 and col[4] == "-"]
    if not month_cols:
        raise LoaderError("no YYYY-MM month columns found")
    melted = df.melt(id_vars=["Код", "Товар"], value_vars=month_cols, var_name="date", value_name="index_value")
    melted["date"] = pd.to_datetime(melted["date"], format="%Y-%m", errors="coerce")
    melted["item_code"] = pd.to_numeric(melted.pop("Код"), errors="coerce").astype("Int64")
    melted["item_name"] = melted.pop("Товар").astype(str)
    melted["index_value"] = _coerce_numeric(melted["index_value"])
    return melted[["date", "item_code", "item_name", "index_value"]].dropna(subset=["date", "item_code"]).reset_index(drop=True)


def load_weights(path: str | Path, region_code: int = 7, weight_column: str = "Weight_gross") -> pd.DataFrame:
    df = read_csv_flexible(path)
    missing = REQUIRED_WEIGHT_COLUMNS - set(df.columns)
    if missing:
        raise LoaderError(f"weights file missing columns: {sorted(missing)}")
    if weight_column not in df.columns:
        raise LoaderError(f"weights file missing requested weight column: {weight_column}")
    df = df.loc[df["Region_code"].eq(region_code), ["Day", "Item_code", weight_column]].copy()
    df["weight_year"] = pd.to_datetime(df.pop("Day"), format="%m/%d/%y %H:%M:%S", errors="coerce").dt.year
    df["item_code"] = pd.to_numeric(df.pop("Item_code"), errors="coerce").astype("Int64")
    df["weight"] = _coerce_numeric(df.pop(weight_column))
    df = df.dropna(subset=["weight_year", "item_code", "weight"])
    return df.sort_values(["weight_year", "item_code"]).reset_index(drop=True)


def load_weights_table(path: str | Path, weight_column: str = "Weight") -> pd.DataFrame:
    df = read_csv_flexible(path)
    if "Item_code" not in df.columns or weight_column not in df.columns:
        raise LoaderError(f"weights table must contain Item_code and {weight_column}")
    out = df.copy()
    out["item_code"] = pd.to_numeric(out.pop("Item_code"), errors="coerce").astype("Int64")
    out["weight"] = _coerce_numeric(out.pop(weight_column))
    return out.dropna(subset=["item_code", "weight"]).reset_index(drop=True)



def load_component_basket(path: str | Path) -> pd.DataFrame:
    """Load the canonical leaf-item universe and analytical groups."""

    df = read_csv_flexible(path)
    required = {"Item_code", "Товар", "Компонент", "Субкомпонент"}
    missing = required - set(df.columns)
    if missing:
        raise LoaderError(f"component basket missing columns: {sorted(missing)}")
    out = df[list(required)].copy()
    out["item_code"] = pd.to_numeric(out.pop("Item_code"), errors="coerce").astype("Int64")
    out["item_name"] = out.pop("Товар").astype(str)
    out["component_group"] = out.pop("Компонент").astype("string")
    out["subcomponent_group"] = out.pop("Субкомпонент").astype("string")
    return out.dropna(subset=["item_code"]).drop_duplicates("item_code").reset_index(drop=True)


def load_item_names(path: str | Path) -> pd.DataFrame:
    df = read_csv_flexible(path)
    if not {"Item_code", "Item_name"}.issubset(df.columns):
        raise LoaderError("item names file must contain Item_code and Item_name")
    out = df[["Item_code", "Item_name"]].copy()
    out["item_code"] = pd.to_numeric(out.pop("Item_code"), errors="coerce").astype("Int64")
    out["item_name"] = out.pop("Item_name").astype(str)
    return out.dropna(subset=["item_code"]).drop_duplicates("item_code")


def load_headline(path: str | Path, date_column: str = "Date", mom_column: str = "mom") -> pd.DataFrame:
    df = read_csv_flexible(path)
    if date_column not in df.columns or mom_column not in df.columns:
        raise LoaderError(f"headline file must contain {date_column!r} and {mom_column!r}")
    out = df[[date_column, mom_column]].copy()
    out["date"] = pd.to_datetime(out.pop(date_column), dayfirst=True, errors="coerce")
    out["headline_mom"] = _coerce_numeric(out.pop(mom_column)) - 100.0
    return out.dropna(subset=["date", "headline_mom"]).sort_values("date")


def load_headline_cpi(path: str | Path, date_column: str = "Date", mom_column: str = "mom") -> pd.DataFrame:
    df = read_csv_flexible(path)
    if date_column not in df.columns or mom_column not in df.columns:
        raise LoaderError(f"headline file must contain {date_column!r} and {mom_column!r}")
    out = df.copy()
    out["date"] = pd.to_datetime(out.pop(date_column), dayfirst=True, errors="coerce")
    out["headline_index"] = _coerce_numeric(out.pop(mom_column))
    return out.dropna(subset=["date", "headline_index"]).reset_index(drop=True)


def attach_names(frame: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(names, on="item_code", how="left")


def filter_items(frame: pd.DataFrame, exclude_codes: Iterable[int]) -> pd.DataFrame:
    codes = {int(code) for code in exclude_codes}
    return frame.loc[~frame["item_code"].astype(int).isin(codes)].copy()
