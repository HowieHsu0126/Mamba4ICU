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
    count(*) FILTER (
        WHERE has_sepsis AND has_aki
          AND aki_onset_time <= sepsis_onset_time + INTERVAL '7 day'
    ) AS sa_aki_within_7d_stays
FROM __LABEL_SCHEMA__.label_onset_mimic;
"""


def load_sql_template(filename: str, stage_schema: str, label_schema: str) -> str:
    sql_text = (SQL_DIR / filename).read_text(encoding="utf-8")
    if stage_schema == "sa_aki_stage":
        sql_text = (
            sql_text.replace("__STAGE_SCHEMA__.stay_master_mimic", "sa_aki_stage.stay_master")
            .replace("__LABEL_SCHEMA__.label_onset_mimic", "sa_aki_label.label_onset")
            .replace("__LABEL_SCHEMA__.aki_labels_mimic", "sa_aki_label.aki_labels")
            .replace("__LABEL_SCHEMA__.outcome_followup_mimic", "sa_aki_label.outcome_followup")
        )
    else:
        sql_text = (
            sql_text.replace("__STAGE_SCHEMA__", stage_schema)
            .replace("__LABEL_SCHEMA__", label_schema)
        )
    return sql_text


def run_query_df(conn: psycopg.Connection, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 8; SET work_mem = '1GB';")
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SA-AKI cohort and 48h trajectories from MIMIC-IV PostgreSQL.")
    parser.add_argument("--materialize-labels", action="store_true", help="Run 01-03 SQL scripts to materialize stay_master, KDIGO AKI, and Sepsis-3 tables first.")
    parser.add_argument("--stage-schema", default=os.environ.get("MIMIC_STAGE_SCHEMA", "tiger_stage"))
    parser.add_argument("--label-schema", default=os.environ.get("MIMIC_LABEL_SCHEMA", "tiger_label"))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pg_host = os.environ.get("MIMIC_PG_HOST", "/tmp")
    pg_port = int(os.environ.get("MIMIC_PG_PORT", "5432"))
    pg_db = os.environ.get("MIMIC_PG_DB", "mimiciv")
    pg_user = os.environ.get("MIMIC_PG_USER", "postgres")

    t0 = time.time()
    print(f"Connecting to PostgreSQL {pg_db} on {pg_host}:{pg_port}...")
    with psycopg.connect(host=pg_host, port=pg_port, dbname=pg_db, user=pg_user) as conn:
        if args.materialize_labels:
            for script_name in [
                "01_stay_master.sql",
                "02_kdigo_aki_labels.sql",
                "03_sepsis3_and_sa_aki_cohort.sql",
            ]:
                print(f"Executing {script_name}...")
                with conn.cursor() as cur:
                    cur.execute((SQL_DIR / script_name).read_text(encoding="utf-8"))
                conn.commit()
            stage_schema = "sa_aki_stage"
            label_schema = "sa_aki_label"
        else:
            stage_schema = args.stage_schema
            label_schema = args.label_schema

        print("1/5 Querying global flow counts...")
        flow_sql = FLOW_COUNTS_SQL.replace(
            "__LABEL_SCHEMA__.label_onset_mimic",
            "sa_aki_label.label_onset" if stage_schema == "sa_aki_stage" else f"{label_schema}.label_onset_mimic",
        )
        flow_df = run_query_df(conn, flow_sql)
        flow_dict = {k: int(v) for k, v in flow_df.iloc[0].to_dict().items()}
        (RAW_DIR / "flow_counts.json").write_text(json.dumps(flow_dict, indent=2), encoding="utf-8")
        print("Flow counts:", flow_dict)

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

    print(f"All MIMIC-IV SA-AKI raw extractions completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
