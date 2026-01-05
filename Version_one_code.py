import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

START_YEAR = 2000
END_YEAR = 2023
BASE_START_YEAR = 1995
BASE_END_YEAR = 2000


def check_duplicates(df, keys, name):
    counts = df.groupby(keys).size()
    print(
        f"{name}: unique key count =",
        counts.shape[0],
        "min rows per key =",
        counts.min(),
        "max rows per key =",
        counts.max(),
    )
    return counts.max()


def add_nonlinear(panel, var):
    if var not in panel.columns:
        print(f"Skipping nonlinear transforms, {var} not in panel")
        return panel

    mask = panel[var].notna()

    panel[var + "_sq"] = np.nan
    panel.loc[mask, var + "_sq"] = panel.loc[mask, var] ** 2

    panel[var + "_log"] = np.nan
    panel.loc[mask, var + "_log"] = np.log(panel.loc[mask, var] + 1e-10)

    if mask.sum() > 0:
        cutoff = panel.loc[mask, var].quantile(0.75)
        panel[var + "_top25"] = 0
        panel.loc[mask & (panel[var] >= cutoff), var + "_top25"] = 1
    else:
        panel[var + "_top25"] = np.nan

    return panel


def load_group_weights_for_vol(group_map):
    fname = "commodity_group_weights.csv"
    if os.path.exists(fname):
        w = pd.read_csv(fname)
        required = {"commodity", "group", "weight"}
        if not required.issubset(set(w.columns)):
            print("Warning: commodity_group_weights.csv missing columns; using equal weights")
        else:
            w = w[w["commodity"].isin(group_map.keys())].copy()
            if w.empty:
                print("Warning: weights file has no matching commodities; using equal weights")
            else:
                w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
                w = w.dropna(subset=["weight", "group"])
                if w.empty:
                    print("Warning: weights file has invalid values; using equal weights")
                else:
                    w["weight"] = w["weight"].clip(lower=0)
                    group_sums = w.groupby("group")["weight"].transform("sum")
                    w = w[group_sums > 0].copy()
                    w["weight"] = w["weight"] / group_sums[group_sums > 0]
                    if w.empty:
                        print("Warning: weights file empty after normalisation; using equal weights")
                    else:
                        return w[["commodity", "group", "weight"]]

    print("Using equal weights within each group for volatility (no usable weights file)")
    tmp = pd.DataFrame(
        [(cmd, grp) for cmd, grp in group_map.items()],
        columns=["commodity", "group"],
    )
    tmp["weight"] = 1.0
    tmp["weight"] = tmp["weight"] / tmp.groupby("group")["weight"].transform("sum")
    return tmp[["commodity", "group", "weight"]]


def compute_group_vol_from_pink(start_year=START_YEAR, end_year=END_YEAR):
    file_name = "raw/CMO-Historical-Data-Monthly-download.xlsx"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Pink Sheet file not found at {file_name}")

    df = pd.read_excel(file_name, sheet_name="Monthly Prices", header=4)
    df = df.rename(columns={"Unnamed: 0": "period"})
    df = df.dropna(subset=["period"])

    df["date"] = pd.to_datetime(df["period"], format="%YM%m")
    df["year"] = df["date"].dt.year

    df = df[(df["year"] >= start_year - 4) & (df["year"] <= end_year)]

    commodity_cols = [c for c in df.columns if c not in ["period", "date", "year"]]

    df_long = df.melt(
        id_vars=["date", "year"],
        value_vars=commodity_cols,
        var_name="commodity",
        value_name="price",
    )

    df_long["price"] = pd.to_numeric(df_long["price"], errors="coerce")
    df_long = df_long.sort_values(["commodity", "date"])

    df_long["log_ret"] = (
        df_long.groupby("commodity")["price"].transform(lambda x: np.log(x).diff())
    )

    group_map = {
        "Aluminum": "core_base",
        "Copper": "core_base",
        "Zinc": "core_base",
        "Nickel": "other_base",
        "Lead": "other_base",
        "Tin": "other_base",
        "Iron ore, cfr spot": "iron",
        "Gold": "precious",
        "Silver": "precious",
        "Platinum": "precious",
    }

    energy_list = [
        "Crude oil, average",
        "Natural gas, Europe",
        "Coal, Australian",
        "Liquefied natural gas, Japan",
    ]

    w_groups = load_group_weights_for_vol(group_map)

    me = df_long[df_long["commodity"].isin(group_map.keys())].copy()
    me["group"] = me["commodity"].map(group_map)
    me = me.dropna(subset=["group", "log_ret"])

    me = me.merge(w_groups, on=["commodity", "group"], how="left")

    missing_w = me["weight"].isna()
    if missing_w.any():
        print("Warning: some commodities had no weights; applying equal weights within group for those rows")
        me.loc[missing_w, "weight"] = 1.0
        me.loc[missing_w, "weight"] = (
            me.loc[missing_w]
            .groupby(["group", "date"])["weight"]
            .transform(lambda x: 1.0 / len(x))
        )

    def weighted_ret(g):
        return np.average(g["log_ret"], weights=g["weight"])

    group_monthly_me = (
        me.groupby(["group", "date"])
        .apply(weighted_ret)
        .reset_index(name="log_ret")
    )

    energy_df = df_long[df_long["commodity"].isin(energy_list)].copy()
    energy_df = energy_df.dropna(subset=["log_ret"])

    energy_wide = (
        energy_df
        .pivot(index="date", columns="commodity", values="log_ret")
        .dropna()
    )

    if not energy_wide.empty:
        scaler = StandardScaler()
        X = scaler.fit_transform(energy_wide)

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X)

        energy_factor = pd.DataFrame({
            "date": energy_wide.index,
            "group": "energy",
            "log_ret": pc1.flatten(),
        })
    else:
        energy_factor = pd.DataFrame(columns=["date", "group", "log_ret"])

    group_monthly = pd.concat(
        [group_monthly_me, energy_factor],
        ignore_index=True,
    )

    group_monthly = group_monthly.sort_values(["group", "date"])
    group_monthly["year"] = group_monthly["date"].dt.year

    group_monthly["sigma_3yr_month"] = (
        group_monthly
        .groupby("group")["log_ret"]
        .rolling(window=36, min_periods=36)
        .std()
        .reset_index(level=0, drop=True)
    )

    group_monthly["sigma_5yr_month"] = (
        group_monthly
        .groupby("group")["log_ret"]
        .rolling(window=60, min_periods=60)
        .std()
        .reset_index(level=0, drop=True)
    )

    def ewma_sigma(x, lam=0.9):
        sigma2 = []
        prev = np.nan
        for r in x:
            if np.isnan(r):
                sigma2.append(np.nan)
                continue
            if np.isnan(prev):
                prev = r ** 2
            else:
                prev = lam * prev + (1.0 - lam) * (r ** 2)
            sigma2.append(prev)
        return np.sqrt(np.array(sigma2))

    group_monthly["sigma_ewma_month"] = (
        group_monthly
        .groupby("group")["log_ret"]
        .transform(lambda x: ewma_sigma(x.values, lam=0.9))
    )

    vol_year_end_list = []
    for grp, df_g in group_monthly.groupby("group"):
        idx = df_g.groupby("year")["date"].idxmax()
        out = df_g.loc[idx, [
            "group", "year",
            "sigma_3yr_month", "sigma_5yr_month", "sigma_ewma_month",
        ]]
        out = out.rename(columns={
            "sigma_3yr_month": "sigma_3yr_B1",
            "sigma_5yr_month": "sigma_5yr_B3",
            "sigma_ewma_month": "sigma_ewma_B3",
        })
        vol_year_end_list.append(out)

    if not vol_year_end_list:
        raise ValueError("No rolling volatilities could be computed from Pink Sheet data")

    vol_year_end = pd.concat(vol_year_end_list, ignore_index=True)

    vol_year_end = vol_year_end[
        (vol_year_end["year"] >= start_year) & (vol_year_end["year"] <= end_year)
    ]

    return vol_year_end[["group", "year", "sigma_3yr_B1", "sigma_5yr_B3", "sigma_ewma_B3"]]


def run_sample_diagnostics(panel):
    sub = panel.copy()

    n_rows = len(sub)
    n_countries = sub["iso3"].nunique() if "iso3" in sub.columns else None
    year_min = sub["year"].min() if "year" in sub.columns else None
    year_max = sub["year"].max() if "year" in sub.columns else None

    print("Diagnostics: rows:", n_rows,
          "countries:", n_countries,
          "years:", year_min, "to", year_max)

    if "iso3" in sub.columns and "year" in sub.columns:
        obs_per_country = (
            sub.groupby("iso3")["year"]
            .nunique()
            .reset_index(name="T")
        )
        print("T per country summary:")
        print(obs_per_country["T"].describe())
        print("Countries with smallest T:")
        print(obs_per_country.sort_values("T").head(10))
        print("Countries with largest T:")
        print(obs_per_country.sort_values("T", ascending=False).head(10))

    candidate_vars = [
        "L1_Exposure_base_annual_c_t",
        "L1_exposure_annual_c_t",
        "L1_exposure_3yr_c_t",
        "Exposure_base_annual_c_t",
        "exposure_annual_c_t",
        "exposure_3yr_c_t",
        "logGDPpc",
        "openness",
        "logPopulation",
        "inflation",
        "resource_rents",
        "gov_effectiveness",
        "tot_index",
    ]

    vars_present = [v for v in candidate_vars if v in sub.columns]
    if vars_present:
        missing_share = (
            sub[vars_present]
            .isna()
            .mean()
            .sort_values(ascending=False)
        )
        print("Missing share by variable:")
        print(missing_share)

        base_var = None
        if "L1_Exposure_base_annual_c_t" in sub.columns:
            base_var = "L1_Exposure_base_annual_c_t"
        elif "Exposure_base_annual_c_t" in sub.columns:
            base_var = "Exposure_base_annual_c_t"
        elif "L1_exposure_annual_c_t" in sub.columns:
            base_var = "L1_exposure_annual_c_t"
        elif "exposure_annual_c_t" in sub.columns:
            base_var = "exposure_annual_c_t"

        if base_var is not None:
            mask_base = sub[base_var].notna()
            print("Non missing base exposure:", int(mask_base.sum()))

            baseline_macro = [
                v for v in ["logGDPpc", "openness", "logPopulation", "inflation"]
                if v in sub.columns
            ]
            if baseline_macro:
                mask_macro_base = mask_base & sub[baseline_macro].notna().all(axis=1)
            else:
                mask_macro_base = mask_base

            print(
                "Base + baseline macro (logGDPpc, openness, logPopulation, inflation):",
                int(mask_macro_base.sum())
            )

            mask_rents = mask_macro_base
            if "resource_rents" in sub.columns:
                mask_rents = mask_rents & sub["resource_rents"].notna()
            print(
                "Base + baseline macro + rents:",
                int(mask_rents.sum())
            )

            mask_inst = mask_rents
            if "gov_effectiveness" in sub.columns:
                mask_inst = mask_inst & sub["gov_effectiveness"].notna()
            print(
                "Base + baseline macro + rents + institutions:",
                int(mask_inst.sum())
            )

            mask_tot = mask_inst
            if "tot_index" in sub.columns:
                mask_tot = mask_tot & sub["tot_index"].notna()
            print(
                "Base + baseline macro + rents + institutions + ToT:",
                int(mask_tot.sum())
            )

    for var in ["inflation", "resource_rents"]:
        if var in sub.columns:
            x = sub[var].dropna()
            if not x.empty:
                desc = x.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
                print("Outlier check for", var)
                print(desc)


def main():
    if not os.path.exists("spending_investment_panel.csv"):
        print("Missing spending_investment_panel.csv")
        return
    if not os.path.exists("controls_panel.csv"):
        print("Missing controls_panel.csv")
        return

    spend = pd.read_csv("spending_investment_panel.csv")
    controls = pd.read_csv("controls_panel.csv")

    if "iso3" not in spend.columns or "year" not in spend.columns:
        raise ValueError("spending_investment_panel.csv must contain iso3 and year")
    if "iso3" not in controls.columns or "year" not in controls.columns:
        raise ValueError("controls_panel.csv must contain iso3 and year")

    max_dup_spend = check_duplicates(spend, ["iso3", "year"], "spend")
    max_dup_controls = check_duplicates(controls, ["iso3", "year"], "controls")
    if max_dup_spend > 1 or max_dup_controls > 1:
        raise ValueError("Spend or controls has duplicate iso3 year rows")

    if os.path.exists("wdi_rents.csv"):
        rents = pd.read_csv("wdi_rents.csv")
        if "iso3" not in rents.columns or "year" not in rents.columns:
            raise ValueError("wdi_rents.csv must contain iso3 and year")
        rents_use = rents[["iso3", "year", "resource_rents"]]
        controls = controls.merge(rents_use, on=["iso3", "year"], how="left")
    else:
        print("Warning: wdi_rents.csv not found, resource_rents will be missing")

    if os.path.exists("wgi_governance.csv"):
        gov = pd.read_csv("wgi_governance.csv")
        if "iso3" not in gov.columns or "year" not in gov.columns:
            raise ValueError("wgi_governance.csv must contain iso3 and year")
        gov_use = gov[["iso3", "year", "gov_effectiveness"]]
        controls = controls.merge(gov_use, on=["iso3", "year"], how="left")
    else:
        print("Warning: wgi_governance.csv not found, gov_effectiveness will be missing")

    if os.path.exists("wdi_tot.csv"):
        tot = pd.read_csv("wdi_tot.csv")
        if "iso3" not in tot.columns or "year" not in tot.columns:
            raise ValueError("wdi_tot.csv must contain iso3 and year")
        tot_use = tot[["iso3", "year", "tot_index"]]
        controls = controls.merge(tot_use, on=["iso3", "year"], how="left")
    else:
        print("Warning: wdi_tot.csv not found, tot_index will be missing")

    max_dup_controls_after = check_duplicates(controls, ["iso3", "year"], "controls_after_C")
    if max_dup_controls_after > 1:
        raise ValueError("controls_after_C has duplicate iso3 year rows")

    files = {
        "g_ann": "exposure_panel_group_level_annual.csv",
        "g_3yr": "exposure_panel_group_level_3yr.csv",
        "c_ann": "exposure_panel_country_year_annual.csv",
        "c_3yr": "exposure_panel_country_year_3yr.csv",
    }
    for key, fname in files.items():
        if not os.path.exists(fname):
            print("Missing exposure file:", fname)
            return

    g_ann = pd.read_csv(files["g_ann"])
    g_3yr = pd.read_csv(files["g_3yr"])
    c_ann = pd.read_csv(files["c_ann"])
    c_3yr = pd.read_csv(files["c_3yr"])

    required_cols_g_ann = {
        "reporterISO", "year", "group",
        "exposure_annual_c_g_t", "weight_c_g_t", "sigma",
    }
    if not required_cols_g_ann.issubset(set(g_ann.columns)):
        raise ValueError("g_ann missing columns:" + str(required_cols_g_ann))

    check_duplicates(g_ann, ["reporterISO", "year"], "g_ann (long)")

    base_window = g_ann[
        (g_ann["year"] >= BASE_START_YEAR) & (g_ann["year"] <= BASE_END_YEAR)
    ].copy()

    if base_window.empty:
        print("Warning: no data in desired base period; using earliest available years per country for base weights")
        g_ann["year_rank"] = g_ann.groupby("reporterISO")["year"].rank(method="first")
        base_window = g_ann[g_ann["year_rank"] <= 5].copy()

    base_weights = (
        base_window
        .groupby(["reporterISO", "group"])["weight_c_g_t"]
        .mean()
        .reset_index()
        .rename(columns={"weight_c_g_t": "weight_c_g_0"})
    )

    g_ann = g_ann.merge(base_weights, on=["reporterISO", "group"], how="left")

    if g_ann["weight_c_g_0"].isna().any():
        print("Warning: some country-groups have missing base weights; their base exposures will be missing")

    g_ann["exposure_base_annual_c_g_t"] = g_ann["weight_c_g_0"] * g_ann["sigma"]

    sum_w_ann = g_ann.groupby(["reporterISO", "year"])["weight_c_g_t"].transform("sum")
    g_ann["weight_norm"] = 0.0
    nonzero_mask_ann = sum_w_ann > 0
    g_ann.loc[nonzero_mask_ann, "weight_norm"] = (
        g_ann.loc[nonzero_mask_ann, "weight_c_g_t"] / sum_w_ann[nonzero_mask_ann]
    )
    g_ann["exposure_norm_annual_c_g_t"] = g_ann["weight_norm"] * g_ann["sigma"]

    g_ann_wide_tv = g_ann.pivot(
        index=["reporterISO", "year"],
        columns="group",
        values="exposure_annual_c_g_t",
    ).reset_index()
    g_ann_wide_tv.columns.name = None
    g_ann_wide_tv = g_ann_wide_tv.rename(columns={
        "reporterISO": "iso3",
        "iron": "E_iron_tv",
        "core_base": "E_core_base_tv",
        "other_base": "E_other_base_tv",
        "energy": "E_energy_tv",
        "precious": "E_precious_tv",
    })
    for col in ["E_iron_tv", "E_core_base_tv", "E_other_base_tv", "E_energy_tv", "E_precious_tv"]:
        if col in g_ann_wide_tv.columns:
            g_ann_wide_tv[col] = g_ann_wide_tv[col].fillna(0.0)
    check_duplicates(g_ann_wide_tv, ["iso3", "year"], "g_ann_wide_tv")

    g_ann_wide_base = g_ann.pivot(
        index=["reporterISO", "year"],
        columns="group",
        values="exposure_base_annual_c_g_t",
    ).reset_index()
    g_ann_wide_base.columns.name = None
    g_ann_wide_base = g_ann_wide_base.rename(columns={
        "reporterISO": "iso3",
        "iron": "E_iron",
        "core_base": "E_core_base",
        "other_base": "E_other_base",
        "energy": "E_energy",
        "precious": "E_precious",
    })
    for col in ["E_iron", "E_core_base", "E_other_base", "E_energy", "E_precious"]:
        if col in g_ann_wide_base.columns:
            g_ann_wide_base[col] = g_ann_wide_base[col].fillna(0.0)
    check_duplicates(g_ann_wide_base, ["iso3", "year"], "g_ann_wide_base")

    g_ann_comp_wide = g_ann.pivot(
        index=["reporterISO", "year"],
        columns="group",
        values="exposure_norm_annual_c_g_t",
    ).reset_index()
    g_ann_comp_wide.columns.name = None
    g_ann_comp_wide = g_ann_comp_wide.rename(columns={
        "reporterISO": "iso3",
        "iron": "E_iron_comp",
        "core_base": "E_core_base_comp",
        "other_base": "E_other_base_comp",
        "energy": "E_energy_comp",
        "precious": "E_precious_comp",
    })
    for col in ["E_iron_comp", "E_core_base_comp", "E_other_base_comp", "E_energy_comp", "E_precious_comp"]:
        if col in g_ann_comp_wide.columns:
            g_ann_comp_wide[col] = g_ann_comp_wide[col].fillna(0.0)
    check_duplicates(g_ann_comp_wide, ["iso3", "year"], "g_ann_comp_wide")

    required_cols_g3 = {
        "reporterISO", "year", "group",
        "exposure_3yr_c_g_t", "weight_c_g_t", "sigma_3yr",
    }
    if not required_cols_g3.issubset(set(g_3yr.columns)):
        raise ValueError("g_3yr missing columns:" + str(required_cols_g3))

    check_duplicates(g_3yr, ["reporterISO", "year"], "g_3yr (long)")

    vol_rolling = compute_group_vol_from_pink(START_YEAR, END_YEAR)
    g_3yr = g_3yr.merge(vol_rolling, on=["group", "year"], how="left")

    if g_3yr["sigma_3yr_B1"].isna().all():
        raise ValueError("B1 rolling volatility sigma_3yr_B1 is entirely missing")

    g_3yr["sigma_3yr_old"] = g_3yr["sigma_3yr"]
    g_3yr["sigma_3yr"] = g_3yr["sigma_3yr_B1"]

    sum_w_3yr = g_3yr.groupby(["reporterISO", "year"])["weight_c_g_t"].transform("sum")
    g_3yr["weight_norm"] = 0.0
    nonzero_mask_3yr = sum_w_3yr > 0
    g_3yr.loc[nonzero_mask_3yr, "weight_norm"] = (
        g_3yr.loc[nonzero_mask_3yr, "weight_c_g_t"] / sum_w_3yr[nonzero_mask_3yr]
    )
    g_3yr["exposure_norm_3yr_c_g_t"] = g_3yr["weight_norm"] * g_3yr["sigma_3yr"]

    g_3yr_wide = g_3yr.pivot(
        index=["reporterISO", "year"],
        columns="group",
        values="exposure_3yr_c_g_t",
    ).reset_index()
    g_3yr_wide.columns.name = None
    g_3yr_wide = g_3yr_wide.rename(columns={
        "reporterISO": "iso3",
        "iron": "E_iron_3yr",
        "core_base": "E_core_base_3yr",
        "other_base": "E_other_base_3yr",
        "energy": "E_energy_3yr",
        "precious": "E_precious_3yr",
    })
    for col in ["E_iron_3yr", "E_core_base_3yr", "E_other_base_3yr", "E_energy_3yr", "E_precious_3yr"]:
        if col in g_3yr_wide.columns:
            g_3yr_wide[col] = g_3yr_wide[col].fillna(0.0)
    check_duplicates(g_3yr_wide, ["iso3", "year"], "g_3yr_wide")

    g_3yr_comp_wide = g_3yr.pivot(
        index=["reporterISO", "year"],
        columns="group",
        values="exposure_norm_3yr_c_g_t",
    ).reset_index()
    g_3yr_comp_wide.columns.name = None
    g_3yr_comp_wide = g_3yr_comp_wide.rename(columns={
        "reporterISO": "iso3",
        "iron": "E_iron_3yr_comp",
        "core_base": "E_core_base_3yr_comp",
        "other_base": "E_other_base_3yr_comp",
        "energy": "E_energy_3yr_comp",
        "precious": "E_precious_3yr_comp",
    })
    for col in ["E_iron_3yr_comp", "E_core_base_3yr_comp", "E_other_base_3yr_comp",
                "E_energy_3yr_comp", "E_precious_3yr_comp"]:
        if col in g_3yr_comp_wide.columns:
            g_3yr_comp_wide[col] = g_3yr_comp_wide[col].fillna(0.0)
    check_duplicates(g_3yr_comp_wide, ["iso3", "year"], "g_3yr_comp_wide")

    required_cols_c_ann = {"reporterISO", "year", "exposure_annual_c_t"}
    if not required_cols_c_ann.issubset(set(c_ann.columns)):
        raise ValueError("c_ann missing columns:" + str(required_cols_c_ann))

    c_ann = c_ann.rename(columns={"reporterISO": "iso3"})
    max_dup_c_ann = check_duplicates(c_ann, ["iso3", "year"], "c_ann_before")
    if max_dup_c_ann > 1:
        c_ann = c_ann.groupby(["iso3", "year"], as_index=False)["exposure_annual_c_t"].sum()
    max_dup_c_ann_after = check_duplicates(c_ann, ["iso3", "year"], "c_ann_after")
    if max_dup_c_ann_after > 1:
        raise ValueError("c_ann still has duplicate iso3 year rows")

    required_cols_c3 = {"reporterISO", "year", "exposure_3yr_c_t"}
    if not required_cols_c3.issubset(set(c_3yr.columns)):
        raise ValueError("c_3yr missing columns:" + str(required_cols_c3))

    c_3yr = c_3yr.rename(columns={"reporterISO": "iso3"})
    max_dup_c3 = check_duplicates(c_3yr, ["iso3", "year"], "c_3yr_before")
    if max_dup_c3 > 1:
        c_3yr = c_3yr.groupby(["iso3", "year"], as_index=False)["exposure_3yr_c_t"].sum()
    max_dup_c3_after = check_duplicates(c_3yr, ["iso3", "year"], "c_3yr_after")
    if max_dup_c3_after > 1:
        raise ValueError("c_3yr still has duplicate iso3 year rows")

    panel = spend.merge(controls, on=["iso3", "year"], how="left")
    max_dup_panel_base = check_duplicates(panel, ["iso3", "year"], "panel_base")
    if max_dup_panel_base > 1:
        raise ValueError("panel_base has duplicate iso3 year rows")

    panel = panel.merge(g_ann_wide_base, on=["iso3", "year"], how="left")
    panel = panel.merge(g_ann_wide_tv, on=["iso3", "year"], how="left")
    panel = panel.merge(g_3yr_wide, on=["iso3", "year"], how="left")
    panel = panel.merge(g_ann_comp_wide, on=["iso3", "year"], how="left")
    panel = panel.merge(g_3yr_comp_wide, on=["iso3", "year"], how="left")
    panel = panel.merge(c_ann, on=["iso3", "year"], how="left")
    panel = panel.merge(c_3yr, on=["iso3", "year"], how="left")

    for col in ["exposure_annual_c_t", "exposure_3yr_c_t"]:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    max_dup_panel_full = check_duplicates(panel, ["iso3", "year"], "panel_full")
    if max_dup_panel_full > 1:
        raise ValueError("panel_full has duplicate iso3 year rows")

    panel = panel[(panel["year"] >= START_YEAR) & (panel["year"] <= END_YEAR)]

    bad_codes = [
        "WLD", "HIC", "MIC", "UMC", "LMC", "LIC",
        "EAP", "ECA", "LAC", "MNA", "NAC", "SAS", "SSA",
        "EUU", "EMU", "ARB","LMY", "IDX", "OSS"
    ]
    panel = panel[~panel["iso3"].isin(bad_codes)].copy()

    base_group_cols = [
        c
        for c in ["E_iron", "E_core_base", "E_other_base", "E_energy", "E_precious"] 
        if c in panel.columns
    ]
    if base_group_cols:
        panel["Exposure_base_annual_c_t"] = panel[base_group_cols].sum(axis=1)
    else:
        panel["Exposure_base_annual_c_t"] = np.nan
        print("Warning: no base group exposure columns found when constructing total base exposure")

    if os.path.exists("structural_dependence_base.csv"):
        dep = pd.read_csv("structural_dependence_base.csv")
        if "iso3" not in dep.columns:
            raise ValueError("structural_dependence_base.csv must contain iso3")
        if "Dep_S" in dep.columns and "Dep_O" in dep.columns:
            dep_use = dep[["iso3", "Dep_S", "Dep_O"]].copy()
        else:
            if "MineralShare_c0" in dep.columns:
                dep = dep.rename(columns={"MineralShare_c0": "Dep_O"})
            if "Dep_O" not in dep.columns:
                raise ValueError(
                    "structural_dependence_base.csv needs Dep_O or MineralShare_c0"
                )
            if "Dep_S" not in dep.columns:
                if "logGDPpc_base" in dep.columns:
                    dep["Dep_S"] = dep["logGDPpc_base"]
                else:
                    dep["Dep_S"] = dep["Dep_O"]
            dep_use = dep[["iso3", "Dep_S", "Dep_O"]].copy()
        panel = panel.merge(dep_use, on="iso3", how="left")
        print("Constructed structural and observed mineral dependence (Dep_S, Dep_O)")
    else:
        print("structural_dependence_base.csv not found; Dep_S and Dep_O not merged")

    try:
        infl = pd.read_csv("wdi_inflation.csv")
    except FileNotFoundError:
        print("wdi_inflation.csv not found; keeping existing inflation values")
    else:
        infl = infl.rename(columns={"inflation": "inflation_imf"})
        merge_cols = [c for c in ["iso3", "year"] if c in panel.columns and c in infl.columns]
        if len(merge_cols) == 2:
            panel = panel.merge(infl, on=merge_cols, how="left")
            if "inflation" in panel.columns:
                mask_imf = panel["inflation_imf"].notna()
                n_replaced = mask_imf.sum()
                print(
                    "Replaced inflation with IMF series for",
                    n_replaced,
                    "country-year observations",
                )
                panel.loc[mask_imf, "inflation"] = panel.loc[mask_imf, "inflation_imf"]
                panel = panel.drop(columns=["inflation_imf"])
            else:
                panel = panel.rename(columns={"inflation_imf": "inflation"})
                print(
                    "Created inflation column from IMF series for",
                    len(panel),
                    "rows",
                )
        else:
            print(
                "Could not merge IMF inflation: iso3/year keys missing in panel or wdi_inflation.csv"
            )

    run_sample_diagnostics(panel)

    if "exports_gdp" in panel.columns:
        mask_exp = panel["exports_gdp"] > 0
        base_expo_cols = [
            "E_iron",
            "E_core_base",
            "E_other_base",
            "E_energy",
            "E_precious",
            "E_iron_3yr",
            "E_core_base_3yr",
            "E_other_base_3yr",
            "E_energy_3yr",
            "E_precious_3yr",
            "Exposure_base_annual_c_t",
            "exposure_annual_c_t",
            "exposure_3yr_c_t",
        ]
        for col in base_expo_cols:
            if col in panel.columns:
                alt_col = col + "_alt"
                panel[alt_col] = np.nan
                panel.loc[mask_exp, alt_col] = (
                    panel.loc[mask_exp, col] * 100.0 / panel.loc[mask_exp, "exports_gdp"]
                )
    else:
        print("Warning: exports_gdp not found, skipping alternative exposures")

    lag_vars = [
        "E_iron",
        "E_core_base",
        "E_other_base",
        "E_energy",
        "E_precious",
        "Exposure_base_annual_c_t",
        "exposure_annual_c_t",
        "E_iron_3yr",
        "E_core_base_3yr",
        "E_other_base_3yr",
        "E_energy_3yr",
        "E_precious_3yr",
        "exposure_3yr_c_t",
        "E_iron_comp",
        "E_core_base_comp",
        "E_other_base_comp",
        "E_energy_comp",
        "E_precious_comp",
        "E_iron_3yr_comp",
        "E_core_base_3yr_comp",
        "E_other_base_3yr_comp",
        "E_energy_3yr_comp",
        "E_precious_3yr_comp",
        "E_iron_alt",
        "E_core_base_alt",
        "E_other_base_alt",
        "E_energy_alt",
        "E_precious_alt",
        "E_iron_3yr_alt",
        "E_core_base_3yr_alt",
        "E_other_base_3yr_alt",
        "E_energy_3yr_alt",
        "E_precious_3yr_alt",
        "Exposure_base_annual_c_t_alt",
        "exposure_annual_c_t_alt",
        "exposure_3yr_c_t_alt",
    ]

    for var in lag_vars:
        if var in panel.columns:
            panel[f"L1_{var}"] = panel.groupby("iso3")[var].shift(1)

    panel = add_nonlinear(panel, "L1_Exposure_base_annual_c_t")
    panel = add_nonlinear(panel, "L1_Exposure_base_annual_c_t_alt")

    if "iso3" in panel.columns and "year" in panel.columns:
        obs_per_country = panel.groupby("iso3")["year"].nunique().reset_index(name="T")
        short_iso3 = obs_per_country[obs_per_country["T"] < 10]["iso3"]
        if not short_iso3.empty:
            print(f"Dropping countries with fewer than 10 fiscal years: {len(short_iso3)}")
            panel = panel[~panel["iso3"].isin(short_iso3)]

    if "population" in panel.columns:
        panel["logPopulation"] = np.nan
        mask_pop = panel["population"] > 0
        panel.loc[mask_pop, "logPopulation"] = np.log(panel.loc[mask_pop, "population"])

    aggregate_iso3 = [
        "WLD",
        "HIC",
        "UMC",
        "LMC",
        "MIC",
        "LIC",
        "AFE",
        "AFW",
        "ARB",
        "CEB",
        "EAR",
        "EAS",
        "ECS",
        "EUU",
        "HPC",
        "IBD",
        "IBT",
        "IDA",
        "LDC",
        "LCN",
        "LTE",
        "MEA",
        "OED",
        "PRE",
        "PSS",
        "PST",
        "SST",
        "SSF",
        "TEA",
        "TEC",
        "TLA",
        "TMN",
        "TSA",
        "TSS",
    ]

    if "iso3" in panel.columns:
        n_before = panel["iso3"].nunique()
        panel = panel[~panel["iso3"].isin(aggregate_iso3)]
        n_after = panel["iso3"].nunique()
        print("Dropped aggregate ISO3 codes:", n_before - n_after)
 
    un = pd.read_csv("un_countries.csv", encoding="cp1252")
    un_iso3 = set(un["ISO3"].astype(str).str.upper().str.strip())

    n_before_un = panel["iso3"].nunique()
    panel = panel[panel["iso3"].isin(un_iso3)]
    n_after_un = panel["iso3"].nunique()
    print("Kept UN-style countries only:", n_after_un, "of", n_before_un)

    panel_controls = panel.copy()
    key_controls = ["logGDPpc", "openness", "inflation", "logPopulation"]
    for v in key_controls:
        if v in panel_controls.columns:
            panel_controls = panel_controls[panel_controls[v].notna()]

    panel_controls.to_csv("final_regression_panel_clean.csv", index=False)
    print("Saved final_regression_panel_clean.csv with shape", panel_controls.shape)

    inv_panel = panel_controls.copy()
    if "logInv" in inv_panel.columns:
        inv_panel = inv_panel[inv_panel["logInv"].notna()]
    if "InvestmentShare" in inv_panel.columns:
        inv_panel = inv_panel[inv_panel["InvestmentShare"] > 0]
    inv_panel.to_csv("final_panel_investment.csv", index=False)
    print("Saved final_panel_investment.csv with shape", inv_panel.shape)

    edu_panel = panel_controls.copy()
    if "logEdu" in edu_panel.columns:
        edu_panel = edu_panel[edu_panel["logEdu"].notna()]
    if "EducationShare" in edu_panel.columns:
        edu_panel = edu_panel[edu_panel["EducationShare"] > 0]
    edu_panel.to_csv("final_panel_education.csv", index=False)
    print("Saved final_panel_education.csv with shape", edu_panel.shape)

    health_panel = panel_controls.copy()
    if "logHealth" in health_panel.columns:
        health_panel = health_panel[health_panel["logHealth"].notna()]
    if "HealthShare" in health_panel.columns:
        health_panel = health_panel[health_panel["HealthShare"] > 0]
    health_panel.to_csv("final_panel_health.csv", index=False)
    print("Saved final_panel_health.csv with shape", health_panel.shape)
	


if __name__ == "__main__":
    main()
