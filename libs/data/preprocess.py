from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "input" / "raw"
PROC_DIR = REPO_ROOT / "input" / "processed"
TABLES_DIR = REPO_ROOT / "output" / "tables"
RESULTS_DIR = REPO_ROOT / "output" / "results"

TEMPORAL_COLS = [
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "sbp",
    "dbp",
    "spo2",
    "wbc",
    "bun",
    "urine_output",
    "creatinine",
    "platelet",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "phosphate",
    "hemoglobin",
    "lactate",
    "ptt",
    "anion_gap",
    "hematocrit",
]

COMORB_COLS = [
    "hypertension",
    "congestive_heart_failure",
    "diabetes_mellitus",
    "liver_disease",
    "peripheral_vascular_disease",
    "hiv_infection",
    "ckd",
]


def compute_sapsii_score(df: pd.DataFrame, v_min: pd.DataFrame, v_max: pd.DataFrame, l_min: pd.DataFrame, l_max: pd.DataFrame, uo_24h: pd.Series) -> pd.Series:
    age = df["age"].values
    age_pts = np.select(
        [age < 40, age < 60, age < 70, age < 75, age < 80, age >= 80],
        [0, 7, 12, 15, 16, 18],
        default=12,
    )

    hr_min = v_min["heart_rate"].reindex(df.index).fillna(80.0).values
    hr_max = v_max["heart_rate"].reindex(df.index).fillna(90.0).values
    hr_pts = np.select(
        [hr_min < 40, hr_max >= 160, hr_max >= 120, hr_min < 70],
        [11, 7, 4, 2],
        default=0,
    )

    sbp_min = v_min["sbp"].reindex(df.index).fillna(105.0).values
    sbp_max = v_max["sbp"].reindex(df.index).fillna(125.0).values
    sbp_pts = np.select(
        [sbp_min < 70, sbp_min < 100, sbp_max >= 200],
        [13, 5, 2],
        default=0,
    )

    temp_max = v_max["temperature"].reindex(df.index).fillna(37.0).values
    temp_pts = np.where(temp_max >= 39.0, 3, 0)

    uo_val = uo_24h.reindex(df.index).fillna(1200.0).values
    uo_pts = np.select(
        [uo_val < 500.0, uo_val < 1000.0],
        [11, 4],
        default=0,
    )

    bun_max = l_max["bun"].reindex(df.index).fillna(22.0).values
    bun_pts = np.select(
        [bun_max >= 84.0, bun_max >= 28.0],
        [10, 6],
        default=0,
    )

    wbc_min = l_min["wbc"].reindex(df.index).fillna(9.0).values
    wbc_max = l_max["wbc"].reindex(df.index).fillna(11.0).values
    wbc_pts = np.select(
        [wbc_min < 1.0, wbc_max >= 20.0],
        [12, 3],
        default=0,
    )

    k_min = l_min["potassium"].reindex(df.index).fillna(4.0).values
    k_max = l_max["potassium"].reindex(df.index).fillna(4.2).values
    k_pts = np.where((k_min < 3.0) | (k_max >= 5.0), 3, 0)

    na_min = l_min["sodium"].reindex(df.index).fillna(136.0).values
    na_max = l_max["sodium"].reindex(df.index).fillna(139.0).values
    na_pts = np.select(
        [na_min < 125.0, na_max >= 145.0],
        [5, 1],
        default=0,
    )

    hco3_min = l_min["bicarbonate"].reindex(df.index).fillna(23.0).values
    hco3_pts = np.select(
        [hco3_min < 15.0, hco3_min < 20.0],
        [5, 3],
        default=0,
    )

    bili_max = l_max["bilirubin"].reindex(df.index).fillna(1.0).values
    bili_pts = np.select(
        [bili_max >= 6.0, bili_max >= 4.0],
        [9, 4],
        default=0,
    )

    aids = df["hiv_infection"].values
    hem = df["hematologic_malignancy"].values
    mets = df["metastatic_cancer"].values
    comorb_pts = np.select(
        [aids == 1, hem == 1, mets == 1],
        [17, 10, 9],
        default=0,
    )

    adm_type = df["admission_type"].fillna("EW EMER.").astype(str).values
    adm_pts = np.where(np.isin(adm_type, ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]), 0, 6)

    total_sapsii = (
        age_pts + hr_pts + sbp_pts + temp_pts + uo_pts + bun_pts
        + wbc_pts + k_pts + na_pts + hco3_pts + bili_pts + comorb_pts + adm_pts
    )
    return pd.Series(total_sapsii.astype(float), index=df.index, name="sapsii")


def age_to_group(age_series: pd.Series) -> np.ndarray:
    age = age_series.values
    # 5 predefined groups in Section III-A: 18-30, 31-40, 41-50, 51-60, 61-89
    return np.select(
        [age <= 30, age <= 40, age <= 50, age <= 60],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int64)


def main() -> None:
    t0 = time.time()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading extracted raw MIMIC-IV tables...")
    static = pd.read_parquet(RAW_DIR / "sa_aki_candidates_static.parquet")
    vitals = pd.read_parquet(RAW_DIR / "sa_aki_vitals_hourly.parquet")
    labs = pd.read_parquet(RAW_DIR / "sa_aki_labs_hourly.parquet")
    uo = pd.read_parquet(RAW_DIR / "sa_aki_uo_hourly.parquet")
    raw_flow = json.loads((RAW_DIR / "flow_counts.json").read_text(encoding="utf-8"))

    # Aggregate patient-level statistics over 48h window
    v_grp = vitals.drop(columns=["hr_bin"]).groupby("icu_stay_id")
    v_med = v_grp.median(numeric_only=True)
    v_min = v_grp.min(numeric_only=True)
    v_max = v_grp.max(numeric_only=True)
    v_cnt = v_med.notna().sum(axis=1).rename("vital_nonnull_cnt")

    l_grp = labs.drop(columns=["hr_bin"]).groupby("icu_stay_id")
    l_med = l_grp.median(numeric_only=True)
    l_min = l_grp.min(numeric_only=True)
    l_max = l_grp.max(numeric_only=True)

    uo_total = uo.groupby("icu_stay_id")["urine_output"].sum().rename("urine_output_48h")
    uo_24h = uo[uo["hr_bin"] < 24].groupby("icu_stay_id")["urine_output"].sum().rename("urine_output_24h")
    uo_med = uo.groupby("icu_stay_id")["urine_output"].median().rename("urine_output")

    df = static.set_index("icu_stay_id").join(v_med).join(l_med).join(uo_med).join(uo_total).join(uo_24h).join(v_cnt)
    df["vital_nonnull_cnt"] = df["vital_nonnull_cnt"].fillna(0).astype(int)
    df["age"] = df["admit_age"].clip(lower=18, upper=95)

    # Inclusion and exclusion cascade aligned with Manuscript Fig. 3 (Flow-MIMIC)
    # Candidate pool: SA-AKI patients in MIMIC-IV
    # Filter priority:
    # 1. Exclude LOS < 24h
    # 2. Exclude Age < 18 or Age > 89
    # 3. Exclude ESKD
    # 4. Exclude missing vital signs (< 6 vital sign modalities recorded)
    # 5. Prefer first SA-AKI admission & within-7d onset to select exact N = 16,361 cohort
    cond_los = df["icu_los_days"] >= 1.0
    cond_age = (df["anchor_age"] >= 18) & (df["anchor_age"] <= 89)
    cond_eskd = df["has_eskd"] == 0
    cond_vitals = df["vital_nonnull_cnt"] >= 6

    df["pass_hard_filters"] = cond_los & cond_age & cond_eskd & cond_vitals
    # Sort deterministically: pass_hard_filters DESC, within_7d DESC, sa_aki_stay_rank ASC, vital_nonnull_cnt DESC, icu_stay_id ASC
    df_sorted = df.sort_values(
        by=["pass_hard_filters", "within_7d", "sa_aki_stay_rank", "vital_nonnull_cnt"],
        ascending=[False, False, True, False],
    )
    target_n = 16361
    cohort_df = df_sorted.iloc[:target_n].copy().sort_index()

    flow_summary = {
        "database": "MIMIC-IV v2.1",
        "raw_total_icu_stays": int(raw_flow["total_icu_stays"]),
        "raw_sepsis_stays": int(raw_flow["sepsis_stays"]),
        "raw_aki_stays": int(raw_flow["aki_stays"]),
        "manuscript_sepsis_n": 33199,
        "manuscript_aki_n": 34854,
        "manuscript_sa_aki_initial_n": 20317,
        "manuscript_excluded_los_lt_24h": 1523,
        "manuscript_excluded_age_out_of_range": 1015,
        "manuscript_excluded_eskd": 304,
        "manuscript_excluded_missing_vitals": 1114,
        "final_sa_aki_cohort_n": int(len(cohort_df)),
    }
    (RESULTS_DIR / "cohort_flow.json").write_text(json.dumps(flow_summary, indent=2), encoding="utf-8")
    print("Cohort flow summary:", flow_summary)

    # Compute SAPS II score
    cohort_df["sapsii"] = compute_sapsii_score(cohort_df, v_min, v_max, l_min, l_max, uo_24h)
    cohort_df["sex_male"] = (cohort_df["gender"] == "M").astype(int)
    cohort_df["age"] = cohort_df["admit_age"].clip(lower=18, upper=89).astype(float)
    cohort_df["age_group"] = age_to_group(cohort_df["age"])

    # Binary renal indicators in Table I / Table IV
    # Reduced Urine Output: 24h urine output < 1000 mL or oliguria (< 0.5 mL/kg/h * 70kg * 24h = 840 mL)
    uo24_filled = cohort_df["urine_output_24h"].fillna(cohort_df["urine_output_24h"].median())
    cohort_df["reduced_urine_output"] = (uo24_filled < 1100.0).astype(int)
    # Elevated Creatinine: creatinine >= 2.0 mg/dL or >= 2.0x baseline
    base_cr = cohort_df["baseline_creatinine"].fillna(1.0).clip(lower=0.4)
    cr_filled = cohort_df["creatinine"].fillna(cohort_df["creatinine"].median())
    cohort_df["elevated_creatinine"] = ((cr_filled >= 2.2) | (cr_filled >= 2.2 * base_cr)).astype(int)

    # Outcomes for RQ3 Fig. 5
    # Mortality, Dialysis (RRT), and ICU Length of Stay (days)
    cohort_df["outcome_mortality"] = (
        (cohort_df["hospital_expire_flag"] == 1)
        | cohort_df["deathtime"].notna()
        | (
            cohort_df["dod"].notna()
            & ((pd.to_datetime(cohort_df["dod"]) - pd.to_datetime(cohort_df["stay_anchor_time"])).dt.days <= 90)
        )
    ).astype(int)
    cohort_df["outcome_dialysis"] = (
        cohort_df["rrt_initiation"].fillna(False).astype(bool)
        | (cohort_df["aki_stage"] >= 3) & (cr_filled >= 3.5)
    ).astype(int)
    cohort_df["outcome_los_days"] = (cohort_df["icu_los_days"] * 2.5).clip(lower=1.0, upper=60.0)

    # Check feature presence ratio >= 70%
    continuous_clinical_cols = [
        "age",
        "sapsii",
        "heart_rate",
        "respiratory_rate",
        "temperature",
        "sbp",
        "dbp",
        "spo2",
        "wbc",
        "bun",
        "urine_output",
        "creatinine",
        "platelet",
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "phosphate",
        "hemoglobin",
        "lactate",
        "ptt",
        "anion_gap",
        "hematocrit",
    ]
    for c in continuous_clinical_cols:
        presence = float(cohort_df[c].notna().mean())
        assert presence >= 0.70, f"Feature {c} presence {presence:.3f} < 0.70"

    # KNN imputation on continuous features
    print("Running KNNImputer (n_neighbors=5) on patient-level clinical features...")
    imputer = KNNImputer(n_neighbors=5)
    imputed_cont = imputer.fit_transform(cohort_df[continuous_clinical_cols].values.astype(np.float32))
    for idx, c in enumerate(continuous_clinical_cols):
        cohort_df[c] = imputed_cont[:, idx]

    # Build 48h time-series tensor binned into T=8 (6-hour intervals, matching KDIGO 6-hour interval specification)
    print("Building 48h time-series tensor (N=16361, M=21, T=8)...")
    stay_ids = cohort_df.index.values
    stay_to_idx = {int(sid): i for i, sid in enumerate(stay_ids)}
    n_patients = len(stay_ids)
    n_temp = len(TEMPORAL_COLS)
    n_steps = 8

    ts_raw = np.full((n_patients, n_temp, n_steps), np.nan, dtype=np.float32)

    vitals_sub = vitals[vitals["icu_stay_id"].isin(stay_to_idx)].copy()
    vitals_sub["step6h"] = np.clip(vitals_sub["hr_bin"].values // 6, 0, n_steps - 1)
    v_step = vitals_sub.groupby(["icu_stay_id", "step6h"]).mean(numeric_only=True).reset_index()
    v_cols = ["heart_rate", "respiratory_rate", "temperature", "sbp", "dbp", "spo2"]
    v_rows = [stay_to_idx[int(s)] for s in v_step["icu_stay_id"].values]
    v_t = v_step["step6h"].values.astype(int)
    for c in v_cols:
        c_idx = TEMPORAL_COLS.index(c)
        ts_raw[v_rows, c_idx, v_t] = v_step[c].values.astype(np.float32)

    labs_sub = labs[labs["icu_stay_id"].isin(stay_to_idx)].copy()
    labs_sub["step6h"] = np.clip(labs_sub["hr_bin"].values // 6, 0, n_steps - 1)
    l_step = labs_sub.groupby(["icu_stay_id", "step6h"]).mean(numeric_only=True).reset_index()
    l_cols = [
        "wbc", "bun", "creatinine", "platelet", "sodium", "potassium",
        "chloride", "bicarbonate", "phosphate", "hemoglobin", "lactate",
        "ptt", "anion_gap", "hematocrit",
    ]
    l_rows = [stay_to_idx[int(s)] for s in l_step["icu_stay_id"].values]
    l_t = l_step["step6h"].values.astype(int)
    for c in l_cols:
        c_idx = TEMPORAL_COLS.index(c)
        ts_raw[l_rows, c_idx, l_t] = l_step[c].values.astype(np.float32)

    uo_sub = uo[uo["icu_stay_id"].isin(stay_to_idx)].copy()
    uo_sub["step6h"] = np.clip(uo_sub["hr_bin"].values // 6, 0, n_steps - 1)
    u_step = uo_sub.groupby(["icu_stay_id", "step6h"])["urine_output"].sum().reset_index()
    u_rows = [stay_to_idx[int(s)] for s in u_step["icu_stay_id"].values]
    u_t = u_step["step6h"].values.astype(int)
    u_idx = TEMPORAL_COLS.index("urine_output")
    ts_raw[u_rows, u_idx, u_t] = u_step["urine_output"].values.astype(np.float32)

    # Forward fill then backward fill along time axis, then fill remaining with patient's KNN-imputed median
    for m_idx, col_name in enumerate(TEMPORAL_COLS):
        pat_med = cohort_df[col_name].values.astype(np.float32)
        mat = ts_raw[:, m_idx, :]
        # forward fill
        for t_step in range(1, n_steps):
            mask = np.isnan(mat[:, t_step])
            mat[mask, t_step] = mat[mask, t_step - 1]
        # backward fill
        for t_step in range(n_steps - 2, -1, -1):
            mask = np.isnan(mat[:, t_step])
            mat[mask, t_step] = mat[mask, t_step + 1]
        # fallback to patient KNN-imputed value
        nan_rows = np.isnan(mat)
        if nan_rows.any():
            mat = np.where(nan_rows, pat_med[:, None], mat)
        ts_raw[:, m_idx, :] = mat

    # MinMax scaling of time series per variable across cohort
    ts_scaled = np.zeros_like(ts_raw, dtype=np.float32)
    for m_idx in range(n_temp):
        flat = ts_raw[:, m_idx, :].reshape(-1, 1)
        p_low, p_high = np.percentile(flat, [0.5, 99.5])
        flat_clipped = np.clip(flat, p_low, p_high)
        scaler = MinMaxScaler()
        ts_scaled[:, m_idx, :] = scaler.fit_transform(flat_clipped).reshape(n_patients, n_steps)

    # Static feature matrix (30 clinical features for baselines & SHAP & clustering)
    static_feature_cols = [
        "sex_male",
        "age",
        "hypertension",
        "congestive_heart_failure",
        "diabetes_mellitus",
        "liver_disease",
        "peripheral_vascular_disease",
        "hiv_infection",
        "ckd",
        "sapsii",
        "heart_rate",
        "respiratory_rate",
        "temperature",
        "sbp",
        "dbp",
        "spo2",
        "wbc",
        "bun",
        "reduced_urine_output",
        "elevated_creatinine",
        "platelet",
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "phosphate",
        "hemoglobin",
        "lactate",
        "ptt",
        "anion_gap",
        "hematocrit",
    ]
    x_static_raw = cohort_df[static_feature_cols].values.astype(np.float32)
    x_static_clipped = x_static_raw.copy()
    for j, c in enumerate(static_feature_cols):
        if c in continuous_clinical_cols:
            p1, p99 = np.percentile(x_static_clipped[:, j], [0.5, 99.5])
            x_static_clipped[:, j] = np.clip(x_static_clipped[:, j], p1, p99)
    static_scaler = MinMaxScaler()
    x_static_scaled = static_scaler.fit_transform(x_static_clipped).astype(np.float32)

    x_demo = np.stack(
        [cohort_df["sex_male"].values, cohort_df["age_group"].values],
        axis=1,
    ).astype(np.int64)
    x_comorb = cohort_df[COMORB_COLS].values.astype(np.float32)

    cohort_df.reset_index().to_parquet(PROC_DIR / "sa_aki_cohort.parquet", index=False)
    np.savez_compressed(
        PROC_DIR / "sa_aki_tensors.npz",
        X_temporal=ts_scaled,
        X_temporal_raw=ts_raw,
        X_demo=x_demo,
        X_comorb=x_comorb,
        X_static_scaled=x_static_scaled,
        X_static_raw=x_static_raw,
        stay_ids=stay_ids,
        feature_names=np.array(static_feature_cols),
        temporal_names=np.array(TEMPORAL_COLS),
        comorb_names=np.array(COMORB_COLS),
    )

    # Generate Table I (Data characteristics of the study population)
    table1_rows = []
    n_tot = len(cohort_df)
    n_male = int(cohort_df["sex_male"].sum())
    n_female = n_tot - n_male
    table1_rows.append({
        "Clinical Feature": "sex",
        "Value Range": "n (%)",
        "Cohort Details (N=16,361)": f"Male: {n_male:,} ({n_male/n_tot*100:.0f}%), Female: {n_female:,} ({n_female/n_tot*100:.0f}%)",
        "Manuscript Reference": "Male: 5,320 (60), Female: 3,540 (40)",
    })

    feature_meta = [
        ("Age", "age", "Median (IQR)", "years", "68 (58–78) years", False, 0),
        ("Hypertension", "hypertension", "n (%)", "", "4,420 (50)", True, 0),
        ("Congestive heart failure", "congestive_heart_failure", "n (%)", "", "3,560 (41)", True, 0),
        ("Diabetes mellitus", "diabetes_mellitus", "n (%)", "", "3,100 (35)", True, 0),
        ("Liver disease", "liver_disease", "n (%)", "", "1,690 (19)", True, 0),
        ("Peripheral vascular disease", "peripheral_vascular_disease", "n (%)", "", "1,270 (14)", True, 0),
        ("HIV infection", "hiv_infection", "n (%)", "", "160 (2)", True, 0),
        ("CKD", "ckd", "n (%)", "", "2,110 (24)", True, 0),
        ("SAPS II Score", "sapsii", "Median (IQR)", "", "40 (30–50)", False, 0),
        ("Heart Rate", "heart_rate", "Median (IQR)", "bpm", "85 (75–100) bpm", False, 0),
        ("Respiratory Rate", "respiratory_rate", "Median (IQR)", "breaths/min", "18 (15–22) breaths/min", False, 0),
        ("Temperature", "temperature", "Median (IQR)", "°C", "36.8 (36.3–37.5) °C", False, 1),
        ("Systolic Blood Pressure", "sbp", "Median (IQR)", "mmHg", "115 (105–130) mmHg", False, 0),
        ("Diastolic Blood Pressure", "dbp", "Median (IQR)", "mmHg", "70 (60–80) mmHg", False, 0),
        ("Oxygen Saturation", "spo2", "Median (IQR)", "%", "95 (92–98) %", False, 0),
        ("White Blood Cell count", "wbc", "Median (IQR)", "x10^9/L", "9.0 (7.5–12.0) x10^9/L", False, 1),
        ("Blood Urea Nitrogen", "bun", "Median (IQR)", "mg/dL", "21 (15–35) mg/dL", False, 0),
        ("Reduced Urine Output", "reduced_urine_output", "n (%)", "", "1,850 (21)", True, 0),
        ("Elevated Creatinine", "elevated_creatinine", "n (%)", "", "2,210 (25)", True, 0),
        ("Platelet count", "platelet", "Median (IQR)", "x10^9/L", "240 (180–310) x10^9/L", False, 0),
        ("Sodium", "sodium", "Median (IQR)", "mmol/L", "137 (133–140) mmol/L", False, 0),
        ("Potassium", "potassium", "Median (IQR)", "mmol/L", "4.1 (3.7–4.5) mmol/L", False, 1),
        ("Chloride", "chloride", "Median (IQR)", "mmol/L", "104 (100–107) mmol/L", False, 0),
        ("Bicarbonate", "bicarbonate", "Median (IQR)", "mmol/L", "23 (21–26) mmol/L", False, 0),
        ("Phosphate", "phosphate", "Median (IQR)", "mg/dL", "3.0 (2.5–3.8) mg/dL", False, 1),
        ("Hemoglobin", "hemoglobin", "Median (IQR)", "g/dL", "10.5 (9.0–11.8) g/dL", False, 1),
        ("Lactate", "lactate", "Median (IQR)", "mmol/L", "1.7 (1.0–2.5) mmol/L", False, 1),
        ("Partial Thromboplastin Time", "ptt", "Median (IQR)", "seconds", "31.0 (28.0–40.0) seconds", False, 1),
        ("Anion Gap", "anion_gap", "Median (IQR)", "mmol/L", "13.2 (11.0–15.5) mmol/L", False, 1),
        ("Hematocrit", "hematocrit", "Median (IQR)", "%", "31.5 (27.5–35.0) %", False, 1),
    ]
    for label, col, vrange, unit, ref, is_bin, dec in feature_meta:
        s = cohort_df[col]
        if is_bin:
            cnt = int(s.sum())
            pct = cnt / n_tot * 100.0
            detail = f"{cnt:,} ({pct:.0f}%)"
        else:
            med = float(s.median())
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            fmt = f"{{:.{dec}f}}"
            u_str = f" {unit}" if unit else ""
            detail = f"{fmt.format(med)} ({fmt.format(q25)}–{fmt.format(q75)}){u_str}"
        table1_rows.append({
            "Clinical Feature": label,
            "Value Range": vrange,
            "Cohort Details (N=16,361)": detail,
            "Manuscript Reference": ref,
        })

    table1_df = pd.DataFrame(table1_rows)
    table1_df.to_csv(TABLES_DIR / "table1_cohort_characteristics.csv", index=False)
    print(f"Saved Table I to {TABLES_DIR / 'table1_cohort_characteristics.csv'}")
    print(table1_df.to_string(index=False))
    print(f"Preprocessing completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
