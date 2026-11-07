from __future__ import annotations

import json
from pathlib import Path
import os
import time
import pandas as pd
import psycopg


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "input" / "raw"


FLOW_COUNTS_SQL = """
SELECT
    count(*) AS total_icu_stays,
    count(DISTINCT patient_id) AS total_patients,
    count(*) FILTER (WHERE has_sepsis) AS sepsis_stays,
    count(*) FILTER (WHERE has_aki) AS aki_stays,
    count(DISTINCT patient_id) FILTER (WHERE has_aki) AS aki_patients,
    count(*) FILTER (WHERE has_sepsis AND has_aki) AS sa_aki_all_stays,
    count(*) FILTER (
        WHERE has_sepsis AND has_aki
          AND aki_onset_time <= sepsis_onset_time + interval '7 day'
    ) AS sa_aki_within_7d_stays
FROM tiger_label.label_onset_mimic;
"""


CANDIDATES_STATIC_SQL = """

WITH cand_stays AS (
    SELECT
        l.icu_stay_id,
        l.patient_id,
        l.encounter_id,
        l.aki_onset_time,
        l.aki_stage,
        l.has_aki,
        l.sepsis_onset_time,
        l.has_sepsis,
        l.sa_aki_onset_time,
        l.has_sa_aki,
        (l.aki_onset_time <= l.sepsis_onset_time + interval '7 day') AS within_7d,
        s.stay_anchor_time,
        s.icu_discharge_time,
        s.hospital_admit_time,
        s.hospital_discharge_time,
        s.icu_los_days,
        EXTRACT(EPOCH FROM (s.hospital_discharge_time - s.hospital_admit_time)) / 3600.0 AS hosp_los_hours,
        s.deathtime,
        s.hospital_expire_flag,
        s.admission_type,
        s.gender,
        s.anchor_age,
        s.anchor_year,
        (s.anchor_age + EXTRACT(YEAR FROM s.hospital_admit_time)::int - s.anchor_year) AS admit_age,
        s.dod,
        s.is_first_stay,
        ROW_NUMBER() OVER (
            PARTITION BY l.patient_id
            ORDER BY s.stay_anchor_time, l.icu_stay_id
        ) AS sa_aki_stay_rank,
        ak.baseline_creatinine,
        o.mortality_24h,
        o.mortality_48h,
        o.mortality_72h,
        o.rrt_initiation,
        o.vasopressor_escalation,
        o.aki_stage_progression,
        GREATEST(
            COALESCE(
                 CASE WHEN l.aki_onset_time >= s.stay_anchor_time AND l.aki_onset_time <= s.icu_discharge_time
                      THEN l.aki_onset_time
                      ELSE s.stay_anchor_time
                 END,
                 s.stay_anchor_time
            ),
            s.stay_anchor_time
        ) AS obs_start_time
    FROM tiger_label.label_onset_mimic l
    INNER JOIN tiger_stage.stay_master_mimic s
        ON s.icu_stay_id = l.icu_stay_id
    LEFT JOIN tiger_label.aki_labels_mimic ak
        ON ak.icu_stay_id = l.icu_stay_id
    LEFT JOIN tiger_label.outcome_followup_mimic o
        ON o.icu_stay_id = l.icu_stay_id
    WHERE l.has_sepsis AND l.has_aki
),
comorb AS (
    SELECT
        d.hadm_id,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 3) BETWEEN '401' AND '405')
              OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 3) BETWEEN 'I10' AND 'I15')
            THEN 1 ELSE 0 END) AS hypertension,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(d.icd_code, 1, 3) = '428'
                 OR SUBSTR(d.icd_code, 1, 5) IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493')
                 OR SUBSTR(d.icd_code, 1, 4) BETWEEN '4254' AND '4259'
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('I43','I50')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','P290')
            ))
            THEN 1 ELSE 0 END) AS congestive_heart_failure,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 3) = '250')
              OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 3) IN ('E10','E11','E12','E13','E14'))
            THEN 1 ELSE 0 END) AS diabetes_mellitus,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('570','571')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('0706','0709','5733','5734','5738','5739','V427','4560','4561','4562')
                 OR SUBSTR(d.icd_code, 1, 4) BETWEEN '5722' AND '5728'
                 OR SUBSTR(d.icd_code, 1, 5) IN ('07022','07023','07032','07033','07044','07054')
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('B18','K70','K71','K72','K73','K74','K76')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('I850','I859','I864','I982','Z944')
            ))
            THEN 1 ELSE 0 END) AS liver_disease,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('440','441')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('0930','4373','4471','5571','5579','V434')
                 OR SUBSTR(d.icd_code, 1, 4) BETWEEN '4431' AND '4439'
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('I70','I71','I73')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('I771','I790','I792','K551','K558','K559','Z958','Z959')
            ))
            THEN 1 ELSE 0 END) AS peripheral_vascular_disease,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 3) IN ('042','043','044'))
              OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 3) IN ('B20','B21','B22','B24'))
            THEN 1 ELSE 0 END) AS hiv_infection,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('582','585','586','V56')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('5880','V420','V451')
                 OR SUBSTR(d.icd_code, 1, 4) BETWEEN '5830' AND '5837'
                 OR SUBSTR(d.icd_code, 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(d.icd_code, 1, 3) IN ('N18','N19')
                 OR SUBSTR(d.icd_code, 1, 4) IN ('I120','I131','N032','N033','N034','N035','N036','N037','N052','N053','N054','N055','N056','N057','N250','Z490','Z491','Z492','Z940','Z992')
            ))
            THEN 1 ELSE 0 END) AS ckd,
        MAX(CASE
            WHEN d.icd_code IN ('5856', 'N186')
            THEN 1 ELSE 0 END) AS has_eskd,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 3) BETWEEN '200' AND '208')
              OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 3) BETWEEN 'C81' AND 'C96')
            THEN 1 ELSE 0 END) AS hematologic_malignancy,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 3) BETWEEN '196' AND '199')
              OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 3) BETWEEN 'C77' AND 'C80')
            THEN 1 ELSE 0 END) AS metastatic_cancer
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN (SELECT DISTINCT encounter_id FROM cand_stays) c
        ON c.encounter_id = d.hadm_id
    GROUP BY d.hadm_id
)
SELECT
    cs.*,
    COALESCE(cb.hypertension, 0) AS hypertension,
    COALESCE(cb.congestive_heart_failure, 0) AS congestive_heart_failure,
    COALESCE(cb.diabetes_mellitus, 0) AS diabetes_mellitus,
    COALESCE(cb.liver_disease, 0) AS liver_disease,
    COALESCE(cb.peripheral_vascular_disease, 0) AS peripheral_vascular_disease,
    COALESCE(cb.hiv_infection, 0) AS hiv_infection,
    COALESCE(cb.ckd, 0) AS ckd,
    COALESCE(cb.has_eskd, 0) AS has_eskd,
    COALESCE(cb.hematologic_malignancy, 0) AS hematologic_malignancy,
    COALESCE(cb.metastatic_cancer, 0) AS metastatic_cancer
FROM cand_stays cs
LEFT JOIN comorb cb
    ON cb.hadm_id = cs.encounter_id
ORDER BY cs.icu_stay_id;
"""


VITALS_HOURLY_SQL = """

WITH cand AS (
    SELECT
        l.icu_stay_id,
        s.stay_anchor_time,
        s.icu_discharge_time,
        GREATEST(
            COALESCE(
                CASE WHEN l.aki_onset_time >= s.stay_anchor_time AND l.aki_onset_time <= s.icu_discharge_time
                     THEN l.aki_onset_time
                     ELSE s.stay_anchor_time
                END,
                s.stay_anchor_time
            ),
            s.stay_anchor_time
        ) AS obs_start
    FROM tiger_label.label_onset_mimic l
    INNER JOIN tiger_stage.stay_master_mimic s
        ON s.icu_stay_id = l.icu_stay_id
    WHERE l.has_sepsis AND l.has_aki
)
SELECT
    ce.stay_id AS icu_stay_id,
    LEAST(
        GREATEST(
            FLOOR(EXTRACT(EPOCH FROM (ce.charttime - c.obs_start)) / 3600.0)::int,
            0
        ),
        47
    ) AS hr_bin,
    AVG(CASE WHEN ce.itemid = 220045 AND ce.valuenum > 0 AND ce.valuenum < 300 THEN ce.valuenum END) AS heart_rate,
    AVG(CASE WHEN ce.itemid IN (220210, 224690) AND ce.valuenum > 0 AND ce.valuenum < 70 THEN ce.valuenum END) AS respiratory_rate,
    AVG(CASE
        WHEN ce.itemid = 223761 AND ce.valuenum > 70 AND ce.valuenum < 120 THEN (ce.valuenum - 32.0) / 1.8
        WHEN ce.itemid = 223762 AND ce.valuenum > 10 AND ce.valuenum < 50 THEN ce.valuenum
    END) AS temperature,
    AVG(CASE WHEN ce.itemid IN (220179, 220050, 225309) AND ce.valuenum > 0 AND ce.valuenum < 400 THEN ce.valuenum END) AS sbp,
    AVG(CASE WHEN ce.itemid IN (220180, 220051, 225310) AND ce.valuenum > 0 AND ce.valuenum < 300 THEN ce.valuenum END) AS dbp,
    AVG(CASE WHEN ce.itemid = 220277 AND ce.valuenum > 0 AND ce.valuenum <= 100 THEN ce.valuenum END) AS spo2
FROM mimiciv_icu.chartevents ce
INNER JOIN cand c
    ON c.icu_stay_id = ce.stay_id
WHERE ce.itemid IN (
    220045,
    220210, 224690,
    223761, 223762,
    220179, 220050, 225309,
    220180, 220051, 225310,
    220277
)
  AND ce.valuenum IS NOT NULL
  AND ce.charttime >= c.stay_anchor_time - interval '6 hour'
  AND ce.charttime <= GREATEST(c.obs_start + interval '48 hour', c.stay_anchor_time + interval '48 hour')
GROUP BY ce.stay_id, 2;
"""


LABS_HOURLY_SQL = """

WITH cand AS (
    SELECT
        l.icu_stay_id,
        l.encounter_id,
        s.stay_anchor_time,
        s.hospital_admit_time,
        s.icu_discharge_time,
        GREATEST(
            COALESCE(
                CASE WHEN l.aki_onset_time >= s.stay_anchor_time AND l.aki_onset_time <= s.icu_discharge_time
                     THEN l.aki_onset_time
                     ELSE s.stay_anchor_time
                END,
                s.stay_anchor_time
            ),
            s.stay_anchor_time
        ) AS obs_start
    FROM tiger_label.label_onset_mimic l
    INNER JOIN tiger_stage.stay_master_mimic s
        ON s.icu_stay_id = l.icu_stay_id
    WHERE l.has_sepsis AND l.has_aki
)
SELECT
    c.icu_stay_id,
    LEAST(
        GREATEST(
            FLOOR(EXTRACT(EPOCH FROM (le.charttime - c.obs_start)) / 3600.0)::int,
            0
        ),
        47
    ) AS hr_bin,
    AVG(CASE WHEN le.itemid IN (51301, 51300) AND le.valuenum > 0 AND le.valuenum <= 500 THEN le.valuenum END) AS wbc,
    AVG(CASE WHEN le.itemid = 51006 AND le.valuenum > 0 AND le.valuenum <= 300 THEN le.valuenum END) AS bun,
    AVG(CASE WHEN le.itemid = 50912 AND le.valuenum > 0 AND le.valuenum <= 150 THEN le.valuenum END) AS creatinine,
    AVG(CASE WHEN le.itemid = 51265 AND le.valuenum > 0 AND le.valuenum <= 2000 THEN le.valuenum END) AS platelet,
    AVG(CASE WHEN le.itemid IN (50983, 50824) AND le.valuenum > 0 AND le.valuenum <= 250 THEN le.valuenum END) AS sodium,
    AVG(CASE WHEN le.itemid IN (50971, 50822) AND le.valuenum > 0 AND le.valuenum <= 30 THEN le.valuenum END) AS potassium,
    AVG(CASE WHEN le.itemid IN (50902, 50806) AND le.valuenum > 0 AND le.valuenum <= 250 THEN le.valuenum END) AS chloride,
    AVG(CASE WHEN le.itemid IN (50882, 50803) AND le.valuenum > 0 AND le.valuenum <= 100 THEN le.valuenum END) AS bicarbonate,
    AVG(CASE WHEN le.itemid = 50970 AND le.valuenum > 0 AND le.valuenum <= 50 THEN le.valuenum END) AS phosphate,
    AVG(CASE WHEN le.itemid IN (51222, 50811) AND le.valuenum > 0 AND le.valuenum <= 30 THEN le.valuenum END) AS hemoglobin,
    AVG(CASE WHEN le.itemid = 50813 AND le.valuenum > 0 AND le.valuenum <= 50 THEN le.valuenum END) AS lactate,
    AVG(CASE WHEN le.itemid = 51275 AND le.valuenum > 0 AND le.valuenum <= 300 THEN le.valuenum END) AS ptt,
    AVG(CASE WHEN le.itemid = 50868 AND le.valuenum >= 0 AND le.valuenum <= 100 THEN le.valuenum END) AS anion_gap,
    AVG(CASE WHEN le.itemid IN (51221, 50810) AND le.valuenum > 0 AND le.valuenum <= 100 THEN le.valuenum END) AS hematocrit,
    AVG(CASE WHEN le.itemid = 50885 AND le.valuenum > 0 AND le.valuenum <= 100 THEN le.valuenum END) AS bilirubin
FROM mimiciv_hosp.labevents le
INNER JOIN cand c
    ON c.encounter_id = le.hadm_id
WHERE le.itemid IN (
    51301, 51300,
    51006,
    50912,
    51265,
    50983, 50824,
    50971, 50822,
    50902, 50806,
    50882, 50803,
    50970,
    51222, 50811,
    50813,
    51275,
    50868,
    51221, 50810,
    50885
)
  AND le.valuenum IS NOT NULL
  AND le.charttime >= c.stay_anchor_time - interval '24 hour'
  AND le.charttime <= GREATEST(c.obs_start + interval '48 hour', c.stay_anchor_time + interval '48 hour')
GROUP BY c.icu_stay_id, 2;
"""


UO_HOURLY_SQL = """

WITH cand AS (
    SELECT
        l.icu_stay_id,
        s.stay_anchor_time,
        s.icu_discharge_time,
        GREATEST(
            COALESCE(
                CASE WHEN l.aki_onset_time >= s.stay_anchor_time AND l.aki_onset_time <= s.icu_discharge_time
                     THEN l.aki_onset_time
                     ELSE s.stay_anchor_time
                END,
                s.stay_anchor_time
            ),
            s.stay_anchor_time
        ) AS obs_start
    FROM tiger_label.label_onset_mimic l
    INNER JOIN tiger_stage.stay_master_mimic s
        ON s.icu_stay_id = l.icu_stay_id
    WHERE l.has_sepsis AND l.has_aki
)
SELECT
    oe.stay_id AS icu_stay_id,
    LEAST(
        GREATEST(
            FLOOR(EXTRACT(EPOCH FROM (oe.charttime - c.obs_start)) / 3600.0)::int,
            0
        ),
        47
    ) AS hr_bin,
    SUM(CASE WHEN oe.itemid = 227488 AND oe.value > 0 THEN -1.0 * oe.value ELSE oe.value END) AS urine_output
FROM mimiciv_icu.outputevents oe
INNER JOIN cand c
    ON c.icu_stay_id = oe.stay_id
WHERE oe.itemid IN (
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 226558, 227488, 227489
)
  AND oe.value IS NOT NULL
  AND oe.charttime >= c.stay_anchor_time - interval '6 hour'
  AND oe.charttime <= GREATEST(c.obs_start + interval '48 hour', c.stay_anchor_time + interval '48 hour')
GROUP BY oe.stay_id, 2;
"""


def run_query_df(conn: psycopg.Connection, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 8; SET work_mem = '1GB';")
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("Connecting to PostgreSQL mimiciv on /tmp:5432...")
    with psycopg.connect(host=os.environ.get("MIMIC_PG_HOST", "/tmp"), port=int(os.environ.get("MIMIC_PG_PORT", "5432")), dbname=os.environ.get("MIMIC_PG_DB", "mimiciv"), user=os.environ.get("MIMIC_PG_USER", "postgres")) as conn:
        print("1/5 Querying global flow counts...")
        flow_df = run_query_df(conn, FLOW_COUNTS_SQL)
        flow_dict = flow_df.iloc[0].to_dict()
        flow_dict = {k: int(v) for k, v in flow_dict.items()}
        (RAW_DIR / "flow_counts.json").write_text(json.dumps(flow_dict, indent=2), encoding="utf-8")
        print("Flow counts:", flow_dict)

        print("2/5 Extracting static candidate cohort & comorbidities...")
        t1 = time.time()
        static_df = run_query_df(conn, CANDIDATES_STATIC_SQL)
        static_df.to_parquet(RAW_DIR / "sa_aki_candidates_static.parquet", index=False)
        print(f"Saved sa_aki_candidates_static.parquet: {len(static_df)} rows ({time.time() - t1:.1f}s)")

        print("3/5 Extracting 48h hourly urine output...")
        t1 = time.time()
        uo_df = run_query_df(conn, UO_HOURLY_SQL)
        uo_df.to_parquet(RAW_DIR / "sa_aki_uo_hourly.parquet", index=False)
        print(f"Saved sa_aki_uo_hourly.parquet: {len(uo_df)} rows ({time.time() - t1:.1f}s)")

        print("4/5 Extracting 48h hourly lab results...")
        t1 = time.time()
        labs_df = run_query_df(conn, LABS_HOURLY_SQL)
        labs_df.to_parquet(RAW_DIR / "sa_aki_labs_hourly.parquet", index=False)
        print(f"Saved sa_aki_labs_hourly.parquet: {len(labs_df)} rows ({time.time() - t1:.1f}s)")

        print("5/5 Extracting 48h hourly vital signs...")
        t1 = time.time()
        vitals_df = run_query_df(conn, VITALS_HOURLY_SQL)
        vitals_df.to_parquet(RAW_DIR / "sa_aki_vitals_hourly.parquet", index=False)
        print(f"Saved sa_aki_vitals_hourly.parquet: {len(vitals_df)} rows ({time.time() - t1:.1f}s)")

    print(f"All MIMIC-IV SA-AKI raw extractions completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
