import os
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def compute_group_shares(trade, group_map):
    trade["group"] = trade["HS2"].map(group_map)
    trade = trade.dropna(subset=["group"])
    grouped = (
        trade.groupby(["reporterISO", "year", "group"], as_index=False)["trade_value"].sum()
    )
    total = grouped.groupby(["reporterISO", "year"], as_index=False)["trade_value"].sum()
    total = total.rename(columns={"trade_value": "total_exports"})
    merged = grouped.merge(total, on=["reporterISO", "year"], how="left")
    merged["share"] = merged["trade_value"] / merged["total_exports"]
    return merged


def compute_group_weights(base_shares, price_weights):
    w = base_shares.merge(price_weights, on="group", how="left")
    w["weight"] = w["share"] * w["price_weight"]
    w["weight"] = w["weight"] / w.groupby(["reporterISO", "year"])["weight"].transform(
        lambda x: x.sum()
    )
    return w


def compute_structural_dependence_panel(trade, price_weights):
    hs2_to_group = {
        "72": "iron",
        "73": "iron",
        "26": "iron",
        "27": "energy",
        "28": "core_base",
        "29": "core_base",
        "74": "core_base",
        "75": "core_base",
        "76": "core_base",
        "78": "core_base",
        "79": "core_base",
        "80": "core_base",
        "81": "core_base",
        "82": "other_base",
        "83": "other_base",
        "71": "precious",
    }
    group_map = hs2_to_group
    base_window = (1995, 1997)
    base_shares = compute_group_shares(
        trade[(trade["year"] >= base_window[0]) & (trade["year"] <= base_window[1])],
        group_map,
    )
    base_shares = (
        base_shares.groupby(["reporterISO", "group"], as_index=False)["share"].mean()
    )
    w_base = compute_group_weights(base_shares, price_weights)
    exports_panel = compute_group_shares(trade, group_map)
    exports_panel = exports_panel.rename(columns={"share": "E_group"})
    exports_panel["base_weight"] = exports_panel.merge(
        w_base[["reporterISO", "group", "weight"]],
        on=["reporterISO", "group"],
        how="left",
    )["weight"]
    exports_panel["E_weighted"] = exports_panel["E_group"] * exports_panel["base_weight"]
    annual_panel = (
        exports_panel.groupby(["reporterISO", "year"], as_index=False)["E_weighted"].sum()
    )
    annual_panel = annual_panel.rename(columns={"E_weighted": "Exposure_base_annual_c_t"})
    exports_panel["year_3yr"] = np.floor((exports_panel["year"] - 2000) / 3).astype(int)
    panel_3yr = (
        exports_panel.groupby(["reporterISO", "year_3yr"], as_index=False)["E_weighted"].sum()
    )
    panel_3yr = panel_3yr.rename(columns={"E_weighted": "Exposure_base_3yr_c_t"})
    panel_3yr["year"] = 2000 + panel_3yr["year_3yr"] * 3 + 1
    panel_3yr = panel_3yr.drop(columns=["year_3yr"])
    exports_panel["E_weighted_comp"] = exports_panel["E_weighted"]
    panel_3yr_comp = (
        exports_panel.groupby(["reporterISO", "year_3yr", "group"], as_index=False)[
            "E_weighted_comp"
        ].sum()
    )
    panel_3yr_comp["year"] = 2000 + panel_3yr_comp["year_3yr"] * 3 + 1
    panel_3yr_comp = panel_3yr_comp.drop(columns=["year_3yr"])
    g_ann = annual_panel.copy()
    g_ann["Exposure_base_annual_c_t"] = g_ann["Exposure_base_annual_c_t"].fillna(0)
    g_3yr = panel_3yr.copy()
    g_3yr["Exposure_base_3yr_c_t"] = g_3yr["Exposure_base_3yr_c_t"].fillna(0)
    g_3yr_comp = panel_3yr_comp.copy()
    g_ann_wide = annual_panel.copy()
    trade_group = exports_panel[["reporterISO", "year", "group", "E_weighted"]].copy()
    g_ann_wide = trade_group.pivot_table(
        index=["reporterISO", "year"], columns="group", values="E_weighted", aggfunc="sum"
    ).reset_index()
    g_ann_wide = g_ann_wide.fillna(0)
    g_3yr_wide = panel_3yr_comp.pivot_table(
        index=["reporterISO", "year"], columns="group", values="E_weighted_comp", aggfunc="sum"
    ).reset_index()
    g_3yr_wide = g_3yr_wide.fillna(0)
    return g_ann, g_3yr, g_ann_wide, g_3yr_wide, g_3yr_comp


def compute_group_vol_from_pink(pink, group_map):
    pink["group"] = pink["HS2"].map(group_map)
    pink = pink.dropna(subset=["group"])
    price_changes = (
        pink.sort_values(["group", "year"])
        .groupby(["group"], as_index=False)["price"]
        .apply(lambda x: x.pct_change())
    )
    price_changes = price_changes.rename(columns={"price": "return"})
    sigma_panel = (
        price_changes.groupby(["group", "year"], as_index=False)["return"].std()
    )
    sigma_panel = sigma_panel.rename(columns={"return": "sigma_epsilon"})
    sigma_panel["sigma_epsilon"] = sigma_panel["sigma_epsilon"].fillna(0)
    sigma_panel["sigma_epsilon_3yr"] = (
        sigma_panel.sort_values(["group", "year"])
        .groupby(["group"])["sigma_epsilon"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    ewma_vol = (
        sigma_panel.sort_values(["group", "year"])
        .groupby(["group"])["sigma_epsilon"]
        .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    )
    sigma_panel["sigma_epsilon_ewma"] = ewma_vol
    tot_shocks_5yr = (
        sigma_panel.sort_values(["group", "year"])
        .groupby(["group"])["sigma_epsilon"]
        .transform(lambda x: x - x.rolling(window=5, min_periods=1).mean())
    )
    sigma_panel["tot_shock_5yr"] = tot_shocks_5yr
    tot_shock_3yr = (
        sigma_panel.sort_values(["group", "year"])
        .groupby(["group"])["sigma_epsilon"]
        .transform(lambda x: x - x.rolling(window=3, min_periods=1).mean())
    )
    sigma_panel["tot_shock_3yr"] = tot_shock_3yr
    vol_groups = sigma_panel[["group", "year", "sigma_epsilon_3yr"]].copy()
    tot_shocks_5yr_df = sigma_panel[["group", "year", "tot_shock_5yr"]].copy()
    tot_shock_3yr_df = sigma_panel[["group", "year", "tot_shock_3yr"]].copy()
    ewma_vol_df = sigma_panel[["group", "year", "sigma_epsilon_ewma"]].copy()
    return vol_groups, tot_shocks_5yr_df, tot_shock_3yr_df, ewma_vol_df


def compute_volatility_index_3yr(g_3yr_comp, vol_df):
    vol_df = vol_df.rename(columns={"sigma_epsilon_3yr": "vol_group"})
    g_vol = g_3yr_comp.merge(vol_df, on=["group", "year"], how="left")
    g_vol["E_3yr_comp"] = g_vol["E_weighted_comp"] * g_vol["vol_group"]
    vol_comp = (
        g_vol.groupby(["reporterISO", "year"], as_index=False)["E_3yr_comp"].sum()
    )
    vol_comp = vol_comp.rename(columns={"E_3yr_comp": "volatility_index_3yr"})
    return vol_comp


def merge_macro_controls_and_exposure(panel, controls):
    merged = panel.merge(
        controls,
        on=["iso3", "year"],
        how="left",
    )
    merged = merged.dropna(subset=["EducationShare", "HealthShare", "InvestmentShare"])
    return merged

def main():
    base_dir = "."
    un_trade_path = os.path.join(base_dir, "comtrade_exports.csv")
    pink_path = os.path.join(base_dir, "PINK_group_prices.csv")
    wdi_path = os.path.join(base_dir, "spending_investment_panel.csv")
    inst_path = os.path.join(base_dir, "wgi_governance.csv")
    tot_path = os.path.join(base_dir, "wdi_tot.csv")
    trade = None
    pink = None

    if os.path.exists(un_trade_path):
        print("Loading UN Comtrade exports panel...")
        trade = pd.read_csv(un_trade_path)
        trade = trade.rename(
            columns={
                "Reporter ISO": "reporterISO",
                "Reporter": "reporter",
                "Year": "year",
                "Commodity Code": "HS2",
                "Trade Value (US$)": "trade_value",
            }
        )

                if "trade_value" not in trade.columns:
            if "fob_usd" in trade.columns:
                trade["trade_value"] = trade["fob_usd"]
            else:
                raise KeyError(
                    f"No trade value column found. Available columns: {list(trade.columns)}"
                )

        trade["year"] = trade["year"].astype(int)

        if "HS2" in trade.columns:
            trade["HS2"] = trade["HS2"].astype(str).str.zfill(2)
        elif "Commodity Code" in trade.columns:
            trade["HS2"] = trade["Commodity Code"].astype(str).str.zfill(2)
        elif "cmdCode" in trade.columns:
            trade["HS2"] = trade["cmdCode"].astype(str).str.zfill(2)
        else:
            raise KeyError(
                f"No HS2-like commodity code column found. "
                f"Available columns: {list(trade.columns)}"
            )

        print("Trade data shape:", trade.shape)
    else:
        print("UN Comtrade exports file not found, expecting comtrade_exports.csv in base directory")
        return

    price_weights_path = os.path.join(base_dir, "mineral_exposure_weights_step1.csv")
    if not os.path.exists(price_weights_path):
        print("mineral_exposure_weights_step1.csv not found")
        return

    price_weights = pd.read_csv(price_weights_path)
    price_weights = price_weights.rename(columns={"group_name": "group"})
    g_ann, g_3yr, g_ann_wide, g_3yr_wide, g_3yr_comp = compute_structural_dependence_panel(
        trade, price_weights
    )


    print("Exposure panel (annual) shape:", g_ann.shape)
    print("Exposure panel (3yr) shape:", g_3yr.shape)
    if os.path.exists(pink_path):
        print("Loading PINK group prices...")
        pink = pd.read_csv(pink_path)
        pink["year"] = pink["year"].astype(int)
        pink["HS2"] = pink["HS2"].astype(str).str.zfill(2)
    else:
        print("PINK_group_prices.csv not found, cannot compute volatility index from PINK")
        return
    hs2_to_group = {
        "72": "iron",
        "73": "iron",
        "26": "iron",
        "27": "energy",
        "28": "core_base",
        "29": "core_base",
        "74": "core_base",
        "75": "core_base",
        "76": "core_base",
        "78": "core_base",
        "79": "core_base",
        "80": "core_base",
        "81": "core_base",
        "82": "other_base",
        "83": "other_base",
        "71": "precious",
    }
    vol_groups, tot_shocks_5yr, tot_shock_3yr, ewma_vol = compute_group_vol_from_pink(
        pink, hs2_to_group
    )
    print("Volatility groups shape:", vol_groups.shape)
    g_3yr_comp = g_3yr_comp.merge(vol_groups, on=["group", "year"], how="left")
    g_3yr_comp["sigma_epsilon_3yr"] = g_3yr_comp["sigma_epsilon_3yr"].fillna(0)
    g_3yr_comp = g_3yr_comp.rename(columns={"sigma_epsilon_3yr": "vol_group"})
    g_3yr_comp["E_3yr_comp"] = g_3yr_comp["E_weighted_comp"] * g_3yr_comp["vol_group"]
    vol_comp = (
        g_3yr_comp.groupby(["reporterISO", "year"], as_index=False)["E_3yr_comp"].sum()
    )
    vol_comp = vol_comp.rename(columns={"E_3yr_comp": "volatility_index_3yr"})
    print("Country-level volatility index shape:", vol_comp.shape)
    un_iso3_path = os.path.join(base_dir, "un_countries.csv")
    if not os.path.exists(un_iso3_path):
        print("un_countries.csv not found; cannot map UN names to ISO3")
        return
    un_members = pd.read_csv(un_iso3_path)
    if "ISO3" in un_members.columns:
        un_members = un_members.rename(columns={"ISO3": "iso3"})
    elif "iso3" in un_members.columns:
        pass
    else:
        raise ValueError("UN members file must contain an ISO3 column")
    wdi_panel = pd.read_csv(wdi_path)
    print("WDI macro panel shape:", wdi_panel.shape)
    if "ISO3" in wdi_panel.columns:
        wdi_panel = wdi_panel.rename(columns={"ISO3": "iso3"})
    wdi_panel["year"] = wdi_panel["year"].astype(int)
    panel = wdi_panel.merge(
        g_3yr_wide,
        left_on=["iso3", "year"],
        right_on=["reporterISO", "year"],
        how="left",
    )
    panel = panel.drop(columns=["reporterISO"], errors="ignore")
    exposure_cols = ["iron", "core_base", "other_base", "energy", "precious"]
    for col in exposure_cols:
        if col not in panel.columns:
            panel[col] = 0
    panel["Exposure_base_annual_c_t"] = panel[exposure_cols].sum(axis=1)
    keep_cols = [
        "iso3",
        "year",
        "EducationShare",
        "HealthShare",
        "InvestmentShare",
        "GDPpc_const",
        "Population",
    ]
    exposure_cols_extended = exposure_cols + ["Exposure_base_annual_c_t"]
    keep_cols_extended = keep_cols + exposure_cols_extended
    panel = panel[keep_cols_extended].copy()
    inst_panel = pd.read_csv(inst_path)
    print("WGI institutions panel shape:", inst_panel.shape)
    if "ISO3" in inst_panel.columns:
        inst_panel = inst_panel.rename(columns={"ISO3": "iso3"})
    inst_panel["year"] = inst_panel["year"].astype(int)
    if "GE" in inst_panel.columns:
        inst_panel["gov_effectiveness"] = inst_panel["GE"]
    elif "gov_effectiveness" not in inst_panel.columns:
        raise ValueError("Institutions panel must contain 'GE' or 'gov_effectiveness'")
    base_years_inst = inst_panel[(inst_panel["year"] >= 1995) & (inst_panel["year"] <= 1997)]
    inst_baseline = (
        base_years_inst.groupby("iso3", as_index=False)["gov_effectiveness"].mean()
    )
    inst_baseline = inst_baseline.rename(
        columns={"gov_effectiveness": "gov_effectiveness_base"}
    )
    panel = panel.merge(inst_baseline, on="iso3", how="left")
    panel["gov_effectiveness"] = panel["gov_effectiveness_base"]
    tot_df = pd.read_csv(tot_path)
    print("WDI TOT panel shape:", tot_df.shape)
    if "ISO3" in tot_df.columns:
        tot_df = tot_df.rename(columns={"ISO3": "iso3"})
    tot_df["year"] = tot_df["year"].astype(int)
    if "TOT_index" not in tot_df.columns:
        if "tot_index" in tot_df.columns:
            tot_df = tot_df.rename(columns={"tot_index": "TOT_index"})
        else:
            raise ValueError("TOT panel must contain TOT_index or tot_index")
    tot_df = tot_df.sort_values(["iso3", "year"])
    tot_df["TOT_growth_5yr_rolling"] = (
        tot_df.groupby("iso3")["TOT_index"]
        .transform(lambda x: x.pct_change(periods=5))
        .fillna(0)
    )
    tot_df["TOT_growth_3yr_rolling"] = (
        tot_df.groupby("iso3")["TOT_index"]
        .transform(lambda x: x.pct_change(periods=3))
        .fillna(0)
    )
    tot_df["TOT_ewma"] = (
        tot_df.groupby("iso3")["TOT_index"]
        .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
        .fillna(method="bfill")
    )
    panel = panel.merge(
        tot_df[
            [
                "iso3",
                "year",
                "TOT_index",
                "TOT_growth_5yr_rolling",
                "TOT_growth_3yr_rolling",
                "TOT_ewma",
            ]
        ],
        on=["iso3", "year"],
        how="left",
    )
    panel = panel.merge(vol_comp, on=["reporterISO", "year"], how="left")
    panel = panel.drop(columns=["reporterISO"], errors="ignore")
    panel["volatility_index_3yr"] = panel["volatility_index_3yr"].fillna(0)
    controls_path = os.path.join(base_dir, "controls_panel.csv")
    if not os.path.exists(controls_path):
        print("controls_panel.csv not found; cannot merge controls")
        return
    controls = pd.read_csv(controls_path)
    if "ISO3" in controls.columns:
        controls = controls.rename(columns={"ISO3": "iso3"})
    controls["year"] = controls["year"].astype(int)
    full_panel = merge_macro_controls_and_exposure(panel, controls)
    full_panel["log_gdp_pc"] = np.log(full_panel["GDPpc_const"])
    full_panel["log_population"] = np.log(full_panel["Population"])
    if "Debt_GNI" in full_panel.columns:
        full_panel["debt_gni"] = full_panel["Debt_GNI"]
    elif "debt_gni" not in full_panel.columns:
        full_panel["debt_gni"] = np.nan
    if "Inflation" in full_panel.columns:
        full_panel["inflation"] = full_panel["Inflation"]
    elif "inflation" not in full_panel.columns:
        full_panel["inflation"] = np.nan
    full_panel["vol_x_gov_eff"] = (
        full_panel["volatility_index_3yr"] * full_panel["gov_effectiveness"]
    )
    print("Final regression panel shape:", full_panel.shape)
    education_panel = full_panel.dropna(subset=["EducationShare"]).copy()
    health_panel = full_panel.dropna(subset=["HealthShare"]).copy()
    investment_panel = full_panel.dropna(subset=["InvestmentShare"]).copy()
    education_panel.to_csv("final_panel_education.csv", index=False)
    health_panel.to_csv("final_panel_health.csv", index=False)
    investment_panel.to_csv("final_panel_investment.csv", index=False)
    full_panel.to_csv("final_regression_panel_clean.csv", index=False)
    print("Saved final_panel_education.csv, final_panel_health.csv, final_panel_investment.csv, final_regression_panel_clean.csv")
    print(full_panel[["iso3", "year", "Exposure_base_annual_c_t", "volatility_index_3yr", "gov_effectiveness", "vol_x_gov_eff"]].head(10))


if __name__ == "__main__":
    main()
