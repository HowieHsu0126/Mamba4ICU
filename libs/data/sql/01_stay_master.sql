-- =============================================================================
-- 01_stay_master.sql
-- Extracts ICU stay master table from MIMIC-IV (mimiciv_icu & mimiciv_hosp,
-- with automatic compatibility views for MIT-LCP/mimic-iv v1.0 mimic_core /
-- mimic_hosp / mimic_icu schemas), computing admission age, exact fractional
-- admission age, length of stay, mortality flags, and stay rank.
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mimiciv_hosp'
    ) AND EXISTS (
        SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mimic_hosp'
    ) THEN
        EXECUTE 'CREATE SCHEMA IF NOT EXISTS mimiciv_hosp';
        IF EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'mimic_core' AND table_name = 'admissions'
        ) THEN
            EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.admissions AS SELECT * FROM mimic_core.admissions';
            EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.patients AS SELECT * FROM mimic_core.patients';
            EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.transfers AS SELECT * FROM mimic_core.transfers';
        ELSIF EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'mimic_hosp' AND table_name = 'admissions'
        ) THEN
            EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.admissions AS SELECT * FROM mimic_hosp.admissions';
            EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.patients AS SELECT * FROM mimic_hosp.patients';
            EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.transfers AS SELECT * FROM mimic_hosp.transfers';
        END IF;
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.diagnoses_icd AS SELECT * FROM mimic_hosp.diagnoses_icd';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.labevents AS SELECT * FROM mimic_hosp.labevents';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.microbiologyevents AS SELECT * FROM mimic_hosp.microbiologyevents';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_hosp.prescriptions AS SELECT * FROM mimic_hosp.prescriptions';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mimiciv_icu'
    ) AND EXISTS (
        SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mimic_icu'
    ) THEN
        EXECUTE 'CREATE SCHEMA IF NOT EXISTS mimiciv_icu';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_icu.icustays AS SELECT * FROM mimic_icu.icustays';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_icu.chartevents AS SELECT * FROM mimic_icu.chartevents';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_icu.outputevents AS SELECT * FROM mimic_icu.outputevents';
        EXECUTE 'CREATE OR REPLACE VIEW mimiciv_icu.procedureevents AS SELECT * FROM mimic_icu.procedureevents';
    END IF;
END
$$;

CREATE SCHEMA IF NOT EXISTS sa_aki_stage;

DROP TABLE IF EXISTS sa_aki_stage.stay_master CASCADE;

CREATE TABLE sa_aki_stage.stay_master AS
WITH base_stays AS (
    SELECT
        icu.stay_id AS icu_stay_id,
        icu.hadm_id AS encounter_id,
        icu.subject_id AS patient_id,
        icu.first_careunit,
        icu.last_careunit,
        icu.intime AS stay_anchor_time,
        icu.outtime AS icu_discharge_time,
        icu.los AS icu_los_days,
        adm.admittime AS hospital_admit_time,
        adm.dischtime AS hospital_discharge_time,
        adm.deathtime,
        adm.hospital_expire_flag,
        adm.admission_type,
        adm.admission_location,
        adm.discharge_location,
        COALESCE(to_jsonb(adm)->>'race', to_jsonb(adm)->>'ethnicity') AS ethnicity,
        pat.gender,
        pat.anchor_age,
        pat.anchor_year,
        (pat.anchor_age + EXTRACT(YEAR FROM adm.admittime)::int - pat.anchor_year) AS admit_age,
        (pat.anchor_age + EXTRACT(EPOCH FROM (adm.admittime - MAKE_TIMESTAMP(pat.anchor_year::int, 1, 1, 0, 0, 0.0))) / 31557600.0) AS exact_admit_age,
        pat.dod,
        ROW_NUMBER() OVER (
            PARTITION BY icu.subject_id
            ORDER BY icu.intime, icu.stay_id
        ) AS first_stay_rank
    FROM mimiciv_icu.icustays icu
    INNER JOIN mimiciv_hosp.admissions adm
        ON adm.hadm_id = icu.hadm_id
    INNER JOIN mimiciv_hosp.patients pat
        ON pat.subject_id = icu.subject_id
)
SELECT
    *,
    (first_stay_rank = 1) AS is_first_stay
FROM base_stays;

CREATE INDEX IF NOT EXISTS idx_sa_aki_stay_master_stay ON sa_aki_stage.stay_master (icu_stay_id);
CREATE INDEX IF NOT EXISTS idx_sa_aki_stay_master_patient ON sa_aki_stage.stay_master (patient_id);
CREATE INDEX IF NOT EXISTS idx_sa_aki_stay_master_hadm ON sa_aki_stage.stay_master (encounter_id);
ANALYZE sa_aki_stage.stay_master;
