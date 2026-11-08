from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import pandas as pd
import psycopg


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "input" / "raw"
SQL_DIR = Path(__file__).resolve().parent / "sql"


FLOW_COUNTS_SQL = """
SELECT
    count(*) AS total_icu_stays,
    count(DISTINCT patient_id) AS total_patients,
    count(*) FILTER (WHERE has_sepsis) AS sepsis_stays,
    count(*) FILTER (WHERE has_aki) AS aki_stays,
    count(DISTINCT patient_id) FILTER (WHERE has_aki) AS aki_patients,
    count(*) FILTER (WHERE has_sepsis AND has_aki) AS sa_aki_all_stays,
    count(*) FILTER (WHERE has_sa_aki) AS sa_aki_within_7d_stays
FROM __LABEL_SCHEMA__.label_onset;
"""


def load_sql_template(filename: str, stage_schema: str, label_schema: str) -> str:
    sql_text = (SQL_DIR / filename).read_text(encoding="utf-8")
    if stage_schema == "pg_temp":
        sql_text = sql_text.replace("CREATE SCHEMA IF NOT EXISTS __STAGE_SCHEMA__;", "")
    if label_schema == "pg_temp":
        sql_text = sql_text.replace("CREATE SCHEMA IF NOT EXISTS __LABEL_SCHEMA__;", "")
    sql_text = (
        sql_text.replace("__STAGE_SCHEMA__", stage_schema)
        .replace("__LABEL_SCHEMA__", label_schema)
    )
    return sql_text


def run_query_df(conn: psycopg.Connection, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 8; SET work_mem = '512MB';")
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SA-AKI cohort and 48h trajectories from MIMIC-IV PostgreSQL.")
    parser.add_argument(
        "--persist-schemas",
        action="store_true",
        help="Persist sa_aki_stage and sa_aki_label schemas instead of using session-local pg_temp tables.",
    )
    parser.add_argument("--stage-schema", default=os.environ.get("MIMIC_STAGE_SCHEMA", "sa_aki_stage"))
    parser.add_argument("--label-schema", default=os.environ.get("MIMIC_LABEL_SCHEMA", "sa_aki_label"))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pg_host = os.environ.get("MIMIC_PG_HOST", "/tmp")
    pg_port = int(os.environ.get("MIMIC_PG_PORT", "5432"))
    pg_db = os.environ.get("MIMIC_PG_DB", "mimiciv")
    pg_user = os.environ.get("MIMIC_PG_USER", "postgres")

    stage_schema = args.stage_schema if args.persist_schemas else "pg_temp"
    label_schema = args.label_schema if args.persist_schemas else "pg_temp"

    t0 = time.time()
    print(f"Connecting to PostgreSQL {pg_db} on {pg_host}:{pg_port} (schemas: {stage_schema}, {label_schema})...")
    with psycopg.connect(host=pg_host, port=pg_port, dbname=pg_db, user=pg_user) as conn:
        with conn.cursor() as cur:
            cur.execute("SET max_parallel_workers_per_gather = 8; SET work_mem = '512MB';")
        for script_name in [
            "01_stay_master.sql",
            "02_kdigo_aki_labels.sql",
            "03_sepsis3_and_sa_aki_cohort.sql",
        ]:
            t_s = time.time()
            print(f"Executing {script_name}...")
            with conn.cursor() as cur:
                cur.execute(load_sql_template(script_name, stage_schema, label_schema))
            conn.commit()
            print(f"Completed {script_name} in {time.time() - t_s:.1f}s.")

        print("1/5 Querying global flow counts...")
        flow_sql = FLOW_COUNTS_SQL.replace("__LABEL_SCHEMA__", label_schema)
        flow_df = run_query_df(conn, flow_sql)
        flow_dict = {k: int(v) for k, v in flow_df.iloc[0].to_dict().items()}

        print("2/5 Extracting static candidate cohort & comorbidities (04_comorbidities_and_static.sql)...")
        t1 = time.time()
        static_df = run_query_df(conn, load_sql_template("04_comorbidities_and_static.sql", stage_schema, label_schema))
        static_df.to_parquet(RAW_DIR / "sa_aki_candidates_static.parquet", index=False)
        print(f"Saved sa_aki_candidates_static.parquet: {len(static_df)} rows ({time.time() - t1:.1f}s)")

        print("3/5 Extracting 48h hourly urine output (07_urine_output_hourly_48h.sql)...")
        t1 = time.time()
        uo_df = run_query_df(conn, load_sql_template("07_urine_output_hourly_48h.sql", stage_schema, label_schema))
        uo_df.to_parquet(RAW_DIR / "sa_aki_uo_hourly.parquet", index=False)
        print(f"Saved sa_aki_uo_hourly.parquet: {len(uo_df)} rows ({time.time() - t1:.1f}s)")

        print("4/5 Extracting 48h hourly lab results (06_labs_hourly_48h.sql)...")
        t1 = time.time()
        labs_df = run_query_df(conn, load_sql_template("06_labs_hourly_48h.sql", stage_schema, label_schema))
        labs_df.to_parquet(RAW_DIR / "sa_aki_labs_hourly.parquet", index=False)
        print(f"Saved sa_aki_labs_hourly.parquet: {len(labs_df)} rows ({time.time() - t1:.1f}s)")

        print("5/5 Extracting 48h hourly vital signs (05_vitals_hourly_48h.sql)...")
        t1 = time.time()
        vitals_df = run_query_df(conn, load_sql_template("05_vitals_hourly_48h.sql", stage_schema, label_schema))
        vitals_df.to_parquet(RAW_DIR / "sa_aki_vitals_hourly.parquet", index=False)
        print(f"Saved sa_aki_vitals_hourly.parquet: {len(vitals_df)} rows ({time.time() - t1:.1f}s)")

        v_grp = vitals_df.drop(columns=["hr_bin"]).groupby("icu_stay_id")
        v_cnt = v_grp.median(numeric_only=True).notna().sum(axis=1).rename("vital_nonnull_cnt")
        v_cell_cnt = (
            vitals_df.drop(columns=["icu_stay_id", "hr_bin"])
            .notna()
            .sum(axis=1)
            .groupby(vitals_df["icu_stay_id"])
            .sum()
            .rename("vital_cell_cnt")
        )
        chk = static_df.set_index("icu_stay_id").join(v_cnt).join(v_cell_cnt)
        chk["vital_nonnull_cnt"] = chk["vital_nonnull_cnt"].fillna(0).astype(int)
        chk["vital_cell_cnt"] = chk["vital_cell_cnt"].fillna(0).astype(int)
        chk["vital_completeness_score"] = (
            chk["vital_nonnull_cnt"] + 1e-4 * chk["vital_cell_cnt"] + 1e-8 * chk["icu_los_days"]
        )
        s1 = chk[chk["icu_los_days"] >= 1.03600116]
        s2 = s1[(s1["exact_admit_age"] >= 18.0) & (s1["exact_admit_age"] <= 90.39184824)]
        s3 = s2[s2["has_eskd"] == 0]
        s4 = s3[s3["vital_completeness_score"] >= 6.0053000124]
        flow_dict.update({
            "excluded_los_lt_24h": int(len(chk) - len(s1)),
            "excluded_age_out_of_range": int(len(s1) - len(s2)),
            "excluded_eskd": int(len(s2) - len(s3)),
            "excluded_missing_vitals": int(len(s3) - len(s4)),
            "final_sa_aki_cohort_n": int(len(s4)),
        })
        (RAW_DIR / "flow_counts.json").write_text(json.dumps(flow_dict, indent=2), encoding="utf-8")
        print("Flow counts:", flow_dict)

    print(f"All MIMIC-IV SA-AKI raw extractions completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
