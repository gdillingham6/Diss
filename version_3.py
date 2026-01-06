import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - defensive import guard
    missing = "numpy and pandas"
    raise SystemExit(
        "Required dependencies are missing. Install them with `pip install -r requirements.txt`."
    ) from exc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    base_year_start: int = 1991
    base_year_end: int = 1995

    ge_pre_years: Tuple[int, int, int] = (1996, 1997, 1998)
    min_ge_pre_obs: int = 2

    regression_start_year: int = 1999

    vol_window_years: int = 3
    # Methodology states: require at least 30 monthly returns when L = 3
    # We generalise conservatively as 10 * L monthly returns for other L values.
    min_monthly_returns_per_window: Optional[int] = None

    shock_trailing_mean_years: int = 3

    # Data paths (relative to base_dir unless absolute paths are provided)
    trade_path: str = "comtrade_exports.csv"
    wgi_path: str = "wgi_governance.csv"
    spending_path: str = "spending_investment_panel.csv"
    controls_path: Optional[str] = "controls_panel.csv"

    # Pink Sheet monthly prices and mapping
    pinksheet_path: str = "Pink_Sheet_Data_Monthly_2025.xlsx"
    # Mapping file that assigns each Pink Sheet series to a mineral group.
    # Expected columns: series, group
    # group must be one of: A, B, C, D1, D2
    price_series_map_path: str = "pinksheet_series_to_group.csv"

    # If trade data is at HS code level, provide a mapping from HS4 to group.
    # Expected columns: hs4, group
    hs4_map_path: str = "hs4_to_group.csv"

    out_dir: str = "outputs"


def _as_abs(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


# ---------------------------------------------------------------------
# Defensive utilities
# ---------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns: {missing}")


def _assert_no_dups(df: pd.DataFrame, keys: List[str], label: str) -> None:
    dups = df[df.duplicated(subset=keys, keep=False)]
    if not dups.empty:
        raise ValueError(f"{label}: duplicate keys detected on {keys}. Example rows:\n{dups.head(10)}")


def merge_with_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: List[str],
    how: str,
    label: str,
    validate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge with explicit diagnostics. Fails loudly if duplicates on merge keys exist on either side,
    unless validate indicates otherwise.
    """
    _assert_no_dups(left, on, f"{label}: left side") if validate in ("one_to_one", "one_to_many") else None
    _assert_no_dups(right, on, f"{label}: right side") if validate in ("one_to_one", "many_to_one") else None

    out = left.merge(right, on=on, how=how, validate=validate, indicator=True)

    total = len(out)
    both = int((out["_merge"] == "both").sum())
    left_only = int((out["_merge"] == "left_only").sum())
    right_only = int((out["_merge"] == "right_only").sum())

    print(f"{label}: merge result rows={total:,} matched={both:,} left_only={left_only:,} right_only={right_only:,}")

    # For left merges, right_only should be 0 by construction
    out = out.drop(columns=["_merge"])
    return out


# ---------------------------------------------------------------------
# Trade: base period structural dependence weights s_cg_base
# ---------------------------------------------------------------------

def _normalise_iso3(x: str) -> str:
    if pd.isna(x):
        return x
    return str(x).strip().upper()


def load_trade(trade_path: str) -> pd.DataFrame:
    trade = pd.read_csv(trade_path)

    # Common formats supported:
    # A) iso3, year, hs_code, fob_usd
    # B) iso3, year, hs4, fob_usd
    # C) iso3, year, group, fob_usd  (already mapped)
    # D) iso3, year, hs_group, fob_usd (textual group names, mapped later)
    _require_cols(trade, ["iso3", "year"], "trade")
    if "fob_usd" not in trade.columns:
        raise KeyError("trade: expected a value column named fob_usd")

    trade = trade.copy()
    trade["iso3"] = trade["iso3"].map(_normalise_iso3)
    trade["year"] = pd.to_numeric(trade["year"], errors="coerce").astype("Int64")

    trade = trade.dropna(subset=["iso3", "year", "fob_usd"]).copy()
    trade["fob_usd"] = pd.to_numeric(trade["fob_usd"], errors="coerce")
    trade = trade.dropna(subset=["fob_usd"]).copy()

    return trade


def load_hs4_group_map(hs4_map_path: str) -> pd.DataFrame:
    hs_map = pd.read_csv(hs4_map_path)
    _require_cols(hs_map, ["hs4", "group"], "hs4_to_group map")
    hs_map = hs_map.copy()
    hs_map["hs4"] = hs_map["hs4"].astype(str).str.zfill(4)
    hs_map["group"] = hs_map["group"].astype(str).str.strip()
    _assert_no_dups(hs_map, ["hs4"], "hs4_to_group map")
    return hs_map


def ensure_trade_group(trade: pd.DataFrame, hs4_map_path: str) -> pd.DataFrame:
    """
    Ensure a column named group exists with values in {A,B,C,D1,D2}.
    If group already exists, it is used as is (after stripping).
    If hs_group exists (text labels), it is mapped using a conservative fixed dictionary.
    If hs4 or hs_code exists, hs4 is constructed and mapped using hs4_to_group.csv.
    """
    trade = trade.copy()

    if "group" in trade.columns:
        trade["group"] = trade["group"].astype(str).str.strip()
        return trade

    if "hs_group" in trade.columns:
        # This mapping mirrors your current code conventions.
        # It does not define which HS headings belong to each class; it only translates labels.
        label_map = {
            "iron": "A",
            "core_base": "B",
            "other_base": "C",
            "energy": "D1",
            "precious": "D2",
        }
        trade["group"] = trade["hs_group"].map(label_map)
        trade = trade.dropna(subset=["group"]).copy()
        return trade

    if "hs4" in trade.columns or "hs_code" in trade.columns:
        if "hs4" not in trade.columns:
            trade["hs_code"] = trade["hs_code"].astype(str)
            trade["hs4"] = trade["hs_code"].str.replace(r"\D", "", regex=True).str[:4].str.zfill(4)

        hs_map = load_hs4_group_map(hs4_map_path)
        trade = merge_with_diagnostics(
            trade,
            hs_map,
            on=["hs4"],
            how="left",
            label="merge_trade_hs4_to_group",
            validate="many_to_one",
        )
        trade = trade.dropna(subset=["group"]).copy()
        return trade

    raise KeyError("trade: unable to infer mineral group. Provide group, hs_group, hs4, or hs_code.")


def compute_group_exports_and_total_exports(trade: pd.DataFrame) -> pd.DataFrame:
    """
    Build a country year group panel with:
    group_exports_cyg = sum exports within group g
    total_exports_cy = sum exports across all HS codes (not only mapped groups)
    share_cyg = group_exports_cyg / total_exports_cy
    """
    _require_cols(trade, ["iso3", "year", "fob_usd"], "trade")
    if "group" not in trade.columns:
        raise KeyError("trade: expected group column before aggregation")

    # total exports across all records in the trade file
    total = trade.groupby(["iso3", "year"], as_index=False)["fob_usd"].sum().rename(columns={"fob_usd": "total_exports"})

    group_exports = (
        trade.dropna(subset=["group"])
        .groupby(["iso3", "year", "group"], as_index=False)["fob_usd"]
        .sum()
        .rename(columns={"fob_usd": "group_exports"})
    )

    out = merge_with_diagnostics(
        group_exports,
        total,
        on=["iso3", "year"],
        how="left",
        label="merge_group_exports_to_total_exports",
        validate="many_to_one",
    )
    out["share"] = out["group_exports"] / out["total_exports"]
    return out


def compute_base_shares(
    shares_cyg: pd.DataFrame,
    base_year_start: int,
    base_year_end: int,
    min_year_obs: int,
) -> pd.DataFrame:
    """
    s_cg_base is the mean share over base years for each country and group.

    Note on "insufficient export data":
    The methodology states that countries with insufficient base period export data are excluded.
    It does not specify a numeric rule, so we implement a conservative requirement:
    at least min_year_obs non missing country year shares within the base years.
    """
    base = shares_cyg[(shares_cyg["year"] >= base_year_start) & (shares_cyg["year"] <= base_year_end)].copy()
    base = base.dropna(subset=["share"]).copy()

    base_stats = (
        base.groupby(["iso3", "group"], as_index=False)
        .agg(
            s_cg_base=("share", "mean"),
            n_base_years=("share", "size"),
        )
    )
    base_stats = base_stats[base_stats["n_base_years"] >= min_year_obs].copy()

    # mineral dependence measure in base period, optional but useful for diagnostics
    mineral_dep = base_stats.groupby("iso3", as_index=False)["s_cg_base"].sum().rename(columns={"s_cg_base": "mineral_dep_base"})
    base_stats = merge_with_diagnostics(
        base_stats,
        mineral_dep,
        on=["iso3"],
        how="left",
        label="merge_base_shares_to_mineral_dep",
        validate="many_to_one",
    )
    return base_stats


# ---------------------------------------------------------------------
# Prices: global volatility indices and group price indices
# ---------------------------------------------------------------------

def load_price_series_map(map_path: str) -> pd.DataFrame:
    m = pd.read_csv(map_path)
    _require_cols(m, ["series", "group"], "price_series_to_group map")
    m = m.copy()
    m["series"] = m["series"].astype(str).str.strip()
    m["group"] = m["group"].astype(str).str.strip()
    valid = {"A", "B", "C", "D1", "D2"}
    bad = sorted(set(m["group"]) - valid)
    if bad:
        raise ValueError(f"price series map: invalid group values: {bad}. Expected one of {sorted(valid)}")
    _assert_no_dups(m, ["series"], "price series map")
    return m


def load_pink_sheet_monthly(pinksheet_path: str) -> pd.DataFrame:
    """
    Load Pink Sheet monthly data into long format with columns: date, series, price
    The loader is intentionally flexible because Pink Sheet spreadsheets can vary.
    """
    raw = pd.read_excel(pinksheet_path)

    # Identify a date column
    date_col_candidates = [c for c in raw.columns if str(c).strip().lower() in {"date", "time", "month"}]
    if date_col_candidates:
        date_col = date_col_candidates[0]
        raw = raw.rename(columns={date_col: "date"})
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    else:
        # Common alternative: Year and Month columns
        if {"year", "month"} <= {str(c).strip().lower() for c in raw.columns}:
            col_year = [c for c in raw.columns if str(c).strip().lower() == "year"][0]
            col_month = [c for c in raw.columns if str(c).strip().lower() == "month"][0]
            raw["date"] = pd.to_datetime(
                dict(year=pd.to_numeric(raw[col_year], errors="coerce"), month=pd.to_numeric(raw[col_month], errors="coerce"), day=1),
                errors="coerce",
            )
        else:
            raise ValueError("Pink Sheet file: could not identify a date column. Expected Date or Year and Month.")

    raw = raw.dropna(subset=["date"]).copy()

    # Melt all non date columns as series
    value_cols = [c for c in raw.columns if c != "date"]
    long = raw.melt(id_vars=["date"], value_vars=value_cols, var_name="series", value_name="price")

    long["series"] = long["series"].astype(str).str.strip()
    long["price"] = pd.to_numeric(long["price"], errors="coerce")
    long = long.dropna(subset=["price"]).copy()

    long = long.sort_values(["series", "date"]).reset_index(drop=True)
    return long


def compute_monthly_log_returns(prices_long: pd.DataFrame) -> pd.DataFrame:
    _require_cols(prices_long, ["date", "series", "price"], "prices_long")
    df = prices_long.copy()
    df["log_price"] = np.log(df["price"])
    df["r"] = df.groupby("series")["log_price"].diff()
    df = df.dropna(subset=["r"]).copy()
    return df[["date", "series", "r"]]


def compute_commodity_volatility_by_year(
    returns_long: pd.DataFrame,
    window_years: int,
    min_returns: int,
) -> pd.DataFrame:
    """
    σ_jt = std dev of monthly log returns over Jan(t-L) to Dec(t-1)
    """
    _require_cols(returns_long, ["date", "series", "r"], "returns_long")

    min_year = int(returns_long["date"].dt.year.min())
    max_year = int(returns_long["date"].dt.year.max())

    out_rows = []
    for series, g in returns_long.groupby("series"):
        g = g.sort_values("date")
        for t in range(min_year + window_years, max_year + 1):
            start = pd.Timestamp(year=t - window_years, month=1, day=1)
            end = pd.Timestamp(year=t - 1, month=12, day=31)
            w = g[(g["date"] >= start) & (g["date"] <= end)]
            n = int(w["r"].notna().sum())
            if n >= min_returns:
                sigma = float(w["r"].std(ddof=1))
            else:
                sigma = np.nan
            out_rows.append({"series": series, "year": t, "sigma": sigma, "n_returns": n})

    out = pd.DataFrame(out_rows)
    return out


def compute_group_volatility_indices(
    pinksheet_path: str,
    price_series_map_path: str,
    window_years: int,
    min_returns: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    1) commodity_vol: series year sigma n_returns group
    2) group_vol: year group G_g_t
    """
    prices_long = load_pink_sheet_monthly(pinksheet_path)
    series_map = load_price_series_map(price_series_map_path)

    prices_long = merge_with_diagnostics(
        prices_long,
        series_map,
        on=["series"],
        how="inner",
        label="merge_prices_to_series_map",
        validate="many_to_one",
    )

    returns_long = compute_monthly_log_returns(prices_long[["date", "series", "price"]])
    returns_long = merge_with_diagnostics(
        returns_long,
        series_map,
        on=["series"],
        how="left",
        label="merge_returns_to_series_map",
        validate="many_to_one",
    )

    commodity_vol = compute_commodity_volatility_by_year(
        returns_long[["date", "series", "r"]],
        window_years=window_years,
        min_returns=min_returns,
    )
    commodity_vol = merge_with_diagnostics(
        commodity_vol,
        series_map,
        on=["series"],
        how="left",
        label="merge_commodity_vol_to_series_map",
        validate="many_to_one",
    )

    group_vol = (
        commodity_vol.dropna(subset=["sigma"])
        .groupby(["year", "group"], as_index=False)["sigma"]
        .mean()
        .rename(columns={"sigma": "G_g_t"})
    )

    _assert_no_dups(group_vol, ["year", "group"], "group_volatility_indices")

    return commodity_vol, group_vol


def compute_group_price_index_annual(
    pinksheet_path: str,
    price_series_map_path: str,
) -> pd.DataFrame:
    """
    Construct an annual group price index P_g_t as the simple average of member series annual averages.

    The methodology specifies Δlog(P_g_t) at group level but does not specify the exact aggregator.
    We use the same simple average principle as for group volatility indices.
    """
    prices_long = load_pink_sheet_monthly(pinksheet_path)
    series_map = load_price_series_map(price_series_map_path)

    prices_long = merge_with_diagnostics(
        prices_long,
        series_map,
        on=["series"],
        how="inner",
        label="merge_prices_to_series_map_for_annual_index",
        validate="many_to_one",
    )

    prices_long["year"] = prices_long["date"].dt.year.astype(int)
    series_annual = prices_long.groupby(["series", "group", "year"], as_index=False)["price"].mean().rename(columns={"price": "price_annual_avg"})

    group_index = series_annual.groupby(["group", "year"], as_index=False)["price_annual_avg"].mean().rename(columns={"price_annual_avg": "P_g_t"})
    _assert_no_dups(group_index, ["group", "year"], "group_price_index_annual")

    group_index = group_index.sort_values(["group", "year"]).copy()
    group_index["dlog_P_g_t"] = group_index.groupby("group")["P_g_t"].transform(lambda s: np.log(s).diff())
    return group_index


# ---------------------------------------------------------------------
# Country level exposure and shocks
# ---------------------------------------------------------------------

def compute_country_volatility_exposure(
    base_shares: pd.DataFrame,
    group_vol: pd.DataFrame,
    min_weight_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    VolExposure_ct = sum_g s_cg_base * G_g_t

    We also compute weight_covered_ct = sum_g s_cg_base where G_g_t is observed.
    If weight_covered_ct < min_weight_coverage, VolExposure_ct is set to missing.
    """
    _require_cols(base_shares, ["iso3", "group", "s_cg_base"], "base_shares")
    _require_cols(group_vol, ["year", "group", "G_g_t"], "group_vol")

    expanded = base_shares[["iso3", "group", "s_cg_base"]].merge(group_vol, on=["group"], how="left")
    expanded["component"] = expanded["s_cg_base"] * expanded["G_g_t"]

    by_ct = (
        expanded.groupby(["iso3", "year"], as_index=False)
        .agg(
            VolExposure_ct=("component", "sum"),
            weight_covered_ct=("s_cg_base", lambda s: float(s[expanded.loc[s.index, "G_g_t"].notna()].sum())),
            mineral_dep_base=("s_cg_base", "sum"),
        )
    )

    by_ct.loc[by_ct["weight_covered_ct"] < min_weight_coverage, "VolExposure_ct"] = np.nan
    return by_ct


def compute_country_price_shocks(
    base_shares: pd.DataFrame,
    group_price_index: pd.DataFrame,
    trailing_mean_years: int,
    min_weight_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    Shock_ct = sum_g s_cg_base * dlog_P_g_t
    mean_shock_L_ct = mean of Shock over t-1 back to t-L

    Coverage rule:
    If the weighted share with observed group price changes is below min_weight_coverage,
    Shock_ct is set to missing for that country year.
    """
    _require_cols(base_shares, ["iso3", "group", "s_cg_base"], "base_shares")
    _require_cols(group_price_index, ["group", "year", "dlog_P_g_t"], "group_price_index")

    expanded = base_shares[["iso3", "group", "s_cg_base"]].merge(
        group_price_index[["group", "year", "dlog_P_g_t"]],
        on=["group"],
        how="left",
    )
    expanded["component"] = expanded["s_cg_base"] * expanded["dlog_P_g_t"]

    shock = (
        expanded.groupby(["iso3", "year"], as_index=False)
        .agg(
            Shock_ct=("component", "sum"),
            weight_covered_ct=("s_cg_base", lambda s: float(s[expanded.loc[s.index, "dlog_P_g_t"].notna()].sum())),
            mineral_dep_base=("s_cg_base", "sum"),
        )
    )
    shock.loc[shock["weight_covered_ct"] < min_weight_coverage, "Shock_ct"] = np.nan
    shock = shock.sort_values(["iso3", "year"]).copy()

    # Trailing mean: average of Shock_ct over years t-1 down to t-L
    shock["mean_shock_L_ct"] = (
        shock.groupby("iso3")["Shock_ct"]
        .transform(lambda s: s.shift(1).rolling(trailing_mean_years, min_periods=trailing_mean_years).mean())
    )
    return shock


def load_wgi(wgi_path: str) -> pd.DataFrame:
    wgi = pd.read_csv(wgi_path)
    _require_cols(wgi, ["iso3", "year", "gov_effectiveness"], "wgi_governance")
    wgi = wgi.copy()
    wgi["iso3"] = wgi["iso3"].map(_normalise_iso3)
    wgi["year"] = pd.to_numeric(wgi["year"], errors="coerce").astype("Int64")
    wgi["gov_effectiveness"] = pd.to_numeric(wgi["gov_effectiveness"], errors="coerce")
    wgi = wgi.dropna(subset=["iso3", "year", "gov_effectiveness"]).copy()
    return wgi


def compute_ge_pre(wgi: pd.DataFrame, ge_years: Tuple[int, int, int], min_obs: int) -> pd.DataFrame:
    y0, y1, y2 = ge_years
    ge = wgi[wgi["year"].isin([y0, y1, y2])].copy()
    ge_pre = (
        ge.groupby("iso3", as_index=False)
        .agg(
            GE_pre_raw=("gov_effectiveness", "mean"),
            n_ge_pre=("gov_effectiveness", "size"),
        )
    )
    ge_pre = ge_pre[ge_pre["n_ge_pre"] >= min_obs].copy()
    return ge_pre


def standardise_ge_pre(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise GE_pre_raw to mean 0 and sd 1 within the estimation sample (current panel).
    """
    if "GE_pre_raw" not in panel.columns:
        raise KeyError("panel: expected GE_pre_raw before standardisation")

    mu = float(panel["GE_pre_raw"].mean())
    sd = float(panel["GE_pre_raw"].std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("GE_pre standardisation: sd is zero or missing")

    panel = panel.copy()
    panel["GE_pre_c"] = (panel["GE_pre_raw"] - mu) / sd
    return panel


# ---------------------------------------------------------------------
# Outcomes and controls
# ---------------------------------------------------------------------

def load_spending(spending_path: str) -> pd.DataFrame:
    s = pd.read_csv(spending_path)
    _require_cols(s, ["iso3", "year"], "spending")
    s = s.copy()
    s["iso3"] = s["iso3"].map(_normalise_iso3)
    s["year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
    return s


def build_predetermined_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline controls in the methodology:
    lagged log GDP per capita, lagged trade openness, and population growth.

    This function expects that the merged panel contains:
    gdp_pc_const_usd OR a precomputed log_gdp_pc
    and either trade_open OR exports_gdp and imports_gdp
    and population
    """
    df = df.sort_values(["iso3", "year"]).copy()

    if "log_gdp_pc" not in df.columns:
        if "gdp_pc_const_usd" in df.columns:
            df["log_gdp_pc"] = np.log(pd.to_numeric(df["gdp_pc_const_usd"], errors="coerce"))
        else:
            raise KeyError("controls: need either log_gdp_pc or gdp_pc_const_usd")

    if "trade_open" not in df.columns:
        if {"exports_gdp", "imports_gdp"} <= set(df.columns):
            df["trade_open"] = pd.to_numeric(df["exports_gdp"], errors="coerce") + pd.to_numeric(df["imports_gdp"], errors="coerce")
        else:
            raise KeyError("controls: need either trade_open or both exports_gdp and imports_gdp")

    if "population" not in df.columns:
        raise KeyError("controls: need population to compute population growth")

    df["log_gdp_pc_l1"] = df.groupby("iso3")["log_gdp_pc"].shift(1)
    df["trade_open_l1"] = df.groupby("iso3")["trade_open"].shift(1)
    df["pop_growth"] = df.groupby("iso3")["population"].transform(lambda s: s.pct_change())

    return df


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    cfg = Config()
    base_dir = os.getcwd()
    out_dir = _as_abs(base_dir, cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    min_returns = cfg.min_monthly_returns_per_window
    if min_returns is None:
        min_returns = 10 * cfg.vol_window_years
        if cfg.vol_window_years == 3:
            # Align exactly with the methodology threshold when L = 3
            min_returns = 30

    print("Loading trade data")
    trade = load_trade(_as_abs(base_dir, cfg.trade_path))
    trade = ensure_trade_group(trade, _as_abs(base_dir, cfg.hs4_map_path))
    shares_cyg = compute_group_exports_and_total_exports(trade)

    print("Computing base period export shares s_cg_base")
    # Conservative base period coverage rule
    base_shares = compute_base_shares(
        shares_cyg,
        base_year_start=cfg.base_year_start,
        base_year_end=cfg.base_year_end,
        min_year_obs=3,
    )
    base_shares.to_csv(os.path.join(out_dir, "export_shares_base.csv"), index=False)

    print("Computing predetermined global volatility indices G_g_t from Pink Sheet monthly prices")
    commodity_vol, group_vol = compute_group_volatility_indices(
        pinksheet_path=_as_abs(base_dir, cfg.pinksheet_path),
        price_series_map_path=_as_abs(base_dir, cfg.price_series_map_path),
        window_years=cfg.vol_window_years,
        min_returns=min_returns,
    )
    commodity_vol.to_csv(os.path.join(out_dir, "commodity_volatility_by_year.csv"), index=False)
    group_vol.to_csv(os.path.join(out_dir, f"group_volatility_indices_L{cfg.vol_window_years}.csv"), index=False)

    print("Computing country year volatility exposure VolExposure_ct")
    vol_exposure = compute_country_volatility_exposure(base_shares, group_vol)
    vol_exposure.to_csv(os.path.join(out_dir, "volatility_exposure_country_year.csv"), index=False)

    print("Computing annual group price index and country shocks Shock_ct")
    group_price_index = compute_group_price_index_annual(
        pinksheet_path=_as_abs(base_dir, cfg.pinksheet_path),
        price_series_map_path=_as_abs(base_dir, cfg.price_series_map_path),
    )
    group_price_index.to_csv(os.path.join(out_dir, "group_price_index_annual.csv"), index=False)

    shocks = compute_country_price_shocks(
        base_shares=base_shares,
        group_price_index=group_price_index,
        trailing_mean_years=cfg.shock_trailing_mean_years,
    )
    shocks.to_csv(os.path.join(out_dir, f"price_shocks_country_year_L{cfg.shock_trailing_mean_years}.csv"), index=False)

    print("Loading spending outcomes")
    spending = load_spending(_as_abs(base_dir, cfg.spending_path))

    # Optional controls panel merge, if present
    controls = None
    if cfg.controls_path and os.path.exists(_as_abs(base_dir, cfg.controls_path)):
        controls = pd.read_csv(_as_abs(base_dir, cfg.controls_path))
        _require_cols(controls, ["iso3", "year"], "controls_panel")
        controls = controls.copy()
        controls["iso3"] = controls["iso3"].map(_normalise_iso3)
        controls["year"] = pd.to_numeric(controls["year"], errors="coerce").astype("Int64")

    print("Loading WGI government effectiveness and constructing GE_pre_c")
    wgi = load_wgi(_as_abs(base_dir, cfg.wgi_path))
    ge_pre = compute_ge_pre(wgi, cfg.ge_pre_years, cfg.min_ge_pre_obs)
    ge_pre.to_csv(os.path.join(out_dir, "GE_pre_raw.csv"), index=False)

    # Merge core panel pieces
    panel = spending.copy()
    panel = merge_with_diagnostics(panel, vol_exposure, on=["iso3", "year"], how="left", label="merge_spending_to_vol_exposure", validate="one_to_one")
    panel = merge_with_diagnostics(panel, shocks, on=["iso3", "year"], how="left", label="merge_panel_to_shocks", validate="one_to_one")
    panel = merge_with_diagnostics(panel, ge_pre, on=["iso3"], how="left", label="merge_panel_to_ge_pre", validate="many_to_one")

    if controls is not None:
        panel = merge_with_diagnostics(panel, controls, on=["iso3", "year"], how="left", label="merge_panel_to_controls", validate="one_to_one")

    # Sample restriction: regression starts in 1999
    panel = panel[panel["year"] >= cfg.regression_start_year].copy()

    # Drop countries without predetermined GE and drop observations without volatility exposure
    panel = panel.dropna(subset=["GE_pre_raw", "VolExposure_ct"]).copy()

    # Standardise GE_pre within the estimation sample
    panel = standardise_ge_pre(panel)

    # Build baseline controls (lags and growth)
    panel = build_predetermined_controls(panel)

    # Interaction term
    panel["VolExposure_x_GEpre"] = panel["VolExposure_ct"] * panel["GE_pre_c"]

    # Baseline regression ready subset: drop rows with missing predetermined controls
    baseline_ready_cols = ["log_gdp_pc_l1", "trade_open_l1", "pop_growth"]
    baseline_ready = panel.dropna(subset=baseline_ready_cols).copy()
    baseline_ready.to_csv(os.path.join(out_dir, "final_regression_panel_baseline_ready.csv"), index=False)


    # Persist final regression panel
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel.to_csv(os.path.join(out_dir, "final_regression_panel.csv"), index=False)

    # Convenience splits for baseline outcomes, if present
    for outcome, fname in [
        ("edu_gov_exp_share", "final_panel_education.csv"),
        ("health_gov_exp_share", "final_panel_health.csv"),
    ]:
        if outcome in panel.columns:
            sub = panel.dropna(subset=[outcome]).copy()
            sub.to_csv(os.path.join(out_dir, fname), index=False)

    print("Done. Outputs written to:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        raise
