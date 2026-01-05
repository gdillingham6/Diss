import os
import json
import numpy as np
import pandas as pd


def safe_merge(left, right, on, how="left", suffixes=("", "_r"), check_dups=True, label="merge"):
    if check_dups:
        if right.duplicated(subset=on).any():
            dups = right[right.duplicated(subset=on, keep=False)]
            raise ValueError(f"{label}: right side has duplicate keys\n{dups.head()}")
    out = left.merge(right, on=on, how=how, suffixes=suffixes)
    return out


def compute_group_shares(trade, group_map):
    trade = trade.copy()
    trade["group"] = trade["hs_group"].map(group_map)
    trade = trade[~trade["group"].isna()]
    grouped = trade.groupby(["iso3", "year", "group"], as_index=False)["fob_usd"].sum()
    grouped = grouped.rename(columns={"fob_usd": "group_exports"})
    totals = grouped.groupby(["iso3", "year"], as_index=False)["group_exports"].sum()
    totals = totals.rename(columns={"group_exports": "total_exports"})
    merged = grouped.merge(totals, on=["iso3", "year"], how="left")
    merged["share"] = merged["group_exports"] / merged["total_exports"]
    return merged


def compute_structural_dependence_panel(trade, price_weights):
    base_window = (1993, 2002)
    group_map = {
        "iron": "A",
        "core_base": "B",
        "other_base": "C",
        "energy": "D1",
        "precious": "D2",
    }
    base = trade[(trade["year"] >= base_window[0]) & (trade["year"] <= base_window[1])].copy()
    base_shares = compute_group_shares(base, group_map)
    base_shares = base_shares.rename(columns={"share": "base_share"})
    current_shares = compute_group_shares(trade, group_map)
    current_shares = current_shares.rename(columns={"share": "current_share"})
    joined = current_shares.merge(
        base_shares[["iso3", "group", "base_share"]],
        on=["iso3", "group"],
        how="left",
    )
    joined = joined[joined["base_share"].notna()].copy()
    joined["Exposure_base_annual_c_t"] = joined["current_share"] - joined["base_share"]
    c_annual = joined[["iso3", "year", "group", "Exposure_base_annual_c_t"]].copy()
    c_annual = c_annual.sort_values(["iso3", "group", "year"])
    c_annual["Exposure_base_3yr_c_t"] = (
        c_annual.groupby(["iso3", "group"])["Exposure_base_annual_c_t"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    c_3yr = c_annual[["iso3", "year", "group", "Exposure_base_3yr_c_t"]].copy()
    g_annual = trade.copy()
    g_annual["group"] = g_annual["hs_group"].map(group_map)
    g_annual = g_annual[~g_annual["group"].isna()]
    g_annual = g_annual.groupby(["year", "group"], as_index=False)["fob_usd"].sum()
    g_annual = g_annual.rename(columns={"fob_usd": "group_exports"})
    g_annual = g_annual.sort_values(["group", "year"])
    g_annual["growth"] = (
        g_annual.groupby("group")["group_exports"].pct_change()
    )
    g_annual["growth"] = g_annual["growth"].replace([np.inf, -np.inf], np.nan)
    g_annual["growth"] = g_annual["growth"].fillna(0.0)
    g_annual["sigma_3yr_B1"] = (
        g_annual.groupby("group")["growth"]
        .transform(lambda s: s.rolling(3, min_periods=3).std())
    )
    g_3yr = g_annual[["year", "group", "sigma_3yr_B1"]].copy()
    g_3yr = g_3yr.dropna(subset=["sigma_3yr_B1"])
    g_3yr_comp = price_weights.copy()
    g_3yr_comp = g_3yr_comp.rename(columns={"group": "group", "weight": "weight"})
    g_3yr_comp["weight_norm"] = (
        g_3yr_comp.groupby("group")["weight"].transform(
            lambda x: x / x.sum() if x.sum() != 0 else x
        )
    )
    g_3yr_comp = g_3yr_comp[["group", "weight_norm"]].drop_duplicates()
    return c_annual, c_3yr, g_annual, g_3yr, g_3yr_comp


def add_nonlinear_terms(panel, exposure_col="Exposure_base_annual_c_t"):
    panel = panel.copy()
    if exposure_col in panel.columns:
        panel["Exposure_sq"] = panel[exposure_col] ** 2
        panel["Exposure_cu"] = panel[exposure_col] ** 3
    else:
        print(f"Exposure column {exposure_col} not found in panel")
    return panel


def load_controls_panel_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def main():
    base_dir = "."
    trade_path = os.path.join(base_dir, "comtrade_exports.csv")
    price_weights_path = os.path.join(base_dir, "mineral_exposure_weights_step1.csv")
    spending_path = os.path.join(base_dir, "spending_investment_panel.csv")
    controls_path = os.path.join(base_dir, "controls_panel.csv")
    countries_path = os.path.join(base_dir, "un_countries.csv")
    wgi_path = os.path.join(base_dir, "wgi_governance.csv")
    wdi_gdp_path = os.path.join(base_dir, "wdi_gdp.csv")
    wdi_gdppc_path = os.path.join(base_dir, "wdi_gdp_pc.csv")
    wdi_pop_path = os.path.join(base_dir, "wdi_population.csv")
    wdi_inv_path = os.path.join(base_dir, "wdi_inv_gdp.csv")
    wdi_health_path = os.path.join(base_dir, "wdi_health_gov_gdp.csv")
    wdi_edu_path = os.path.join(base_dir, "wdi_edu_gov_gdp.csv")
    wdi_debt_path = os.path.join(base_dir, "wdi_debt_gni.csv")
    tot_path = os.path.join(base_dir, "wdi_tot.csv")
    print("Loading UN Comtrade exports panel...")
    if not os.path.exists(trade_path):
        print("comtrade_exports.csv not found in current directory")
        return
    trade = pd.read_csv(trade_path)
    rename_map = {
        "year": "year",
        "reporterISO": "iso3",
        "cmdCode": "hs_group",
        "fob_usd": "fob_usd",
    }
    for col in rename_map:
        if col not in trade.columns:
            raise KeyError(f"Expected column {col} in comtrade_exports.csv")
    trade = trade.rename(columns=rename_map)
    trade["year"] = trade["year"].astype(int)
    trade["hs_group"] = trade["hs_group"].astype(str)
    trade = trade[~trade["fob_usd"].isna()]
    trade = trade[trade["fob_usd"] > 0]
    trade = trade[~trade["iso3"].isna()]
    trade = trade[~trade["hs_group"].isna()]
    print("Trade data shape:", trade.shape)
    if not os.path.exists(price_weights_path):
        print("mineral_exposure_weights_step1.csv not found in current directory")
        return
    price_weights = pd.read_csv(price_weights_path)
    if "group" not in price_weights.columns or "weight" not in price_weights.columns:
        raise KeyError("mineral_exposure_weights_step1.csv must have group and weight columns")
    print("Computing structural dependence and trade exposure panels...")
    c_annual, c_3yr, g_annual, g_3yr, g_3yr_comp = compute_structural_dependence_panel(
        trade, price_weights
    )
    c_annual.to_csv(os.path.join(base_dir, "exposure_panel_country_year_annual.csv"), index=False)
    c_3yr.to_csv(os.path.join(base_dir, "exposure_panel_country_year_3yr.csv"), index=False)
    g_annual.to_csv(os.path.join(base_dir, "exposure_panel_group_level_annual.csv"), index=False)
    g_3yr.to_csv(os.path.join(base_dir, "exposure_panel_group_level_3yr.csv"), index=False)
    if os.path.exists(spending_path):
        print("Loading spending and investment panel...")
        controls = pd.read_csv(spending_path)
        if "iso3" not in controls.columns or "year" not in controls.columns:
            raise KeyError("spending_investment_panel.csv must contain iso3 and year columns")
        if "iso3" in controls.columns:
            dup_controls_before = controls.duplicated(subset=["iso3", "year"]).sum()
            print(f"Duplicate iso3-year rows in spending_investment_panel before merges: {dup_controls_before}")
        if os.path.exists(wdi_gdp_path):
            wdi_gdp = pd.read_csv(wdi_gdp_path)
            if {"iso3", "year", "gdp_const_usd"} <= set(wdi_gdp.columns):
                controls = safe_merge(
                    controls,
                    wdi_gdp[["iso3", "year", "gdp_const_usd"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_gdp",
                )
            else:
                print("wdi_gdp.csv missing expected columns iso3, year, gdp_const_usd")
        else:
            print("wdi_gdp.csv not found")
        if os.path.exists(wdi_gdppc_path):
            wdi_gdppc = pd.read_csv(wdi_gdppc_path)
            if {"iso3", "year", "gdp_pc_const_usd"} <= set(wdi_gdppc.columns):
                controls = safe_merge(
                    controls,
                    wdi_gdppc[["iso3", "year", "gdp_pc_const_usd"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_gdp_pc",
                )
            else:
                print("wdi_gdp_pc.csv missing expected columns iso3, year, gdp_pc_const_usd")
        else:
            print("wdi_gdp_pc.csv not found")
        if os.path.exists(wdi_pop_path):
            wdi_pop = pd.read_csv(wdi_pop_path)
            if {"iso3", "year", "population"} <= set(wdi_pop.columns):
                controls = safe_merge(
                    controls,
                    wdi_pop[["iso3", "year", "population"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_population",
                )
            else:
                print("wdi_population.csv missing expected columns iso3, year, population")
        else:
            print("wdi_population.csv not found")
        if os.path.exists(wdi_inv_path):
            wdi_inv = pd.read_csv(wdi_inv_path)
            if {"iso3", "year", "inv_gdp"} <= set(wdi_inv.columns):
                controls = safe_merge(
                    controls,
                    wdi_inv[["iso3", "year", "inv_gdp"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_inv",
                )
            else:
                print("wdi_inv_gdp.csv missing expected columns iso3, year, inv_gdp")
        else:
            print("wdi_inv_gdp.csv not found")
        if os.path.exists(wdi_health_path):
            wdi_health = pd.read_csv(wdi_health_path)
            if {"iso3", "year", "health_gov_gdp"} <= set(wdi_health.columns):
                controls = safe_merge(
                    controls,
                    wdi_health[["iso3", "year", "health_gov_gdp"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_health",
                )
            else:
                print("wdi_health_gov_gdp.csv missing expected columns iso3, year, health_gov_gdp")
        else:
            print("wdi_health_gov_gdp.csv not found")
        if os.path.exists(wdi_edu_path):
            wdi_edu = pd.read_csv(wdi_edu_path)
            if {"iso3", "year", "edu_gov_gdp"} <= set(wdi_edu.columns):
                controls = safe_merge(
                    controls,
                    wdi_edu[["iso3", "year", "edu_gov_gdp"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_edu",
                )
            else:
                print("wdi_edu_gov_gdp.csv missing expected columns iso3, year, edu_gov_gdp")
        else:
            print("wdi_edu_gov_gdp.csv not found")
        if os.path.exists(wdi_debt_path):
            wdi_debt = pd.read_csv(wdi_debt_path)
            if {"iso3", "year", "debt_gni"} <= set(wdi_debt.columns):
                controls = safe_merge(
                    controls,
                    wdi_debt[["iso3", "year", "debt_gni"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wdi_debt",
                )
            else:
                print("wdi_debt_gni.csv missing expected columns iso3, year, debt_gni")
        else:
            print("wdi_debt_gni.csv not found")
        if os.path.exists(wgi_path):
            wgi = pd.read_csv(wgi_path)
            if {"iso3", "year", "gov_effectiveness"} <= set(wgi.columns):
                controls = safe_merge(
                    controls,
                    wgi[["iso3", "year", "gov_effectiveness"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_wgi",
                )
            else:
                print("wgi_governance.csv missing expected columns iso3, year, gov_effectiveness")
        else:
            print("wgi_governance.csv not found")
        if os.path.exists(tot_path):
            tot = pd.read_csv(tot_path)
            if {"iso3", "year", "tot"} <= set(tot.columns):
                controls = safe_merge(
                    controls,
                    tot[["iso3", "year", "tot"]],
                    on=["iso3", "year"],
                    how="left",
                    label="merge_tot",
                )
            else:
                print("wdi_tot.csv missing expected columns iso3, year, tot")
        else:
            print("wdi_tot.csv not found, TOT will be missing")
        if os.path.exists("volatility_panel_2000_2024_epsilon.csv"):
            vol = pd.read_csv("volatility_panel_2000_2024_epsilon.csv")
            if "iso3" not in vol.columns or "year" not in vol.columns:
                raise ValueError("volatility_panel_2000_2024_epsilon.csv must contain iso3 and year")
            if "volatility_index_3yr" not in vol.columns:
                raise ValueError("volatility_panel_2000_2024_epsilon.csv must contain volatility_index_3yr")
            vol_use = vol[["iso3", "year", "volatility_index_3yr"]]
            controls = controls.merge(vol_use, on=["iso3", "year"], how="left")
        else:
            print("Warning: volatility_panel_2000_2024_epsilon.csv not found, volatility_index_3yr will be missing")
        if "gov_effectiveness" in controls.columns and "volatility_index_3yr" in controls.columns:
            controls["vol_x_gov_eff"] = controls["gov_effectiveness"] * controls["volatility_index_3yr"]
        if "iso3" in controls.columns:
            dup_controls_after = controls.duplicated(subset=["iso3", "year"]).sum()
            if dup_controls_after > 0:
                raise ValueError("controls_after_C has duplicate iso3 year rows")
        c_annual_path = os.path.join(base_dir, "exposure_panel_country_year_annual.csv")
        c_3yr_path = os.path.join(base_dir, "exposure_panel_country_year_3yr.csv")
        if not os.path.exists(c_annual_path) or not os.path.exists(c_3yr_path):
            raise FileNotFoundError("Exposure panel files not found after compute_structural_dependence_panel")
        c_annual_loaded = pd.read_csv(c_annual_path)
        c_3yr_loaded = pd.read_csv(c_3yr_path)
        panel = c_annual_loaded.merge(
            controls,
            on=["iso3", "year"],
            how="left",
        )
        if "population" in panel.columns:
            panel = panel[~panel["population"].isna()]
        else:
            print("population column missing in panel")
        panel["log_gdp_pc"] = np.log(panel["gdp_pc_const_usd"]) if "gdp_pc_const_usd" in panel.columns else np.nan
        panel["log_gdp"] = np.log(panel["gdp_const_usd"]) if "gdp_const_usd" in panel.columns else np.nan
        panel["log_population"] = np.log(panel["population"]) if "population" in panel.columns else np.nan
        panel = add_nonlinear_terms(panel, exposure_col="Exposure_base_annual_c_t")
        panel.to_csv(os.path.join(base_dir, "final_panel_with_controls.csv"), index=False)
        dep_edu = "edu_gov_gdp"
        dep_health = "health_gov_gdp"
        dep_inv = "inv_gdp"
        key_controls = [
            "iso3",
            "year",
            "Exposure_base_annual_c_t",
            "Exposure_base_3yr_c_t",
            "log_gdp_pc",
            "log_gdp",
            "log_population",
            "debt_gni",
            "tot",
            "gov_effectiveness",
        ]
        extra_cols = [
            "Exposure_sq",
            "Exposure_cu",
            "volatility_index_3yr",
            "vol_x_gov_eff",
        ]
        panel_for_controls = panel.copy()
        panel_for_controls = panel_for_controls[
            ["iso3", "year", dep_edu, dep_health, dep_inv] + key_controls + extra_cols
        ]
        panel_controls = panel_for_controls.dropna(subset=["iso3", "year"])
        panel_controls.to_csv(os.path.join(base_dir, "final_regression_panel_clean_with_logpop.csv"), index=False)
        edu_panel = panel_controls.dropna(subset=[dep_edu]).copy()
        health_panel = panel_controls.dropna(subset=[dep_health]).copy()
        inv_panel = panel_controls.dropna(subset=[dep_inv]).copy()
        edu_panel.to_csv(os.path.join(base_dir, "final_panel_education.csv"), index=False)
        health_panel.to_csv(os.path.join(base_dir, "final_panel_health.csv"), index=False)
        inv_panel.to_csv(os.path.join(base_dir, "final_panel_investment.csv"), index=False)
        print("Final panel files with education, health, and investment saved.")
    else:
        print("spending_investment_panel.csv not found; cannot build final panel with controls")


if __name__ == "__main__":
    main()

