-- =============================================================================
-- 01_stay_master.sql
-- Extracts ICU stay master table from MIMIC-IV (mimiciv_icu & mimiciv_hosp),
-- computing admission age, exact fractional admission age, length of stay,
-- mortality flags, and stay rank.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS __STAGE_SCHEMA__;

DROP TABLE IF EXISTS __STAGE_SCHEMA__.stay_master CASCADE;

CREATE TABLE __STAGE_SCHEMA__.stay_master AS
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
        adm.race AS ethnicity,
        pat.gender,
        pat.anchor_age,
        pat.anchor_year,
        (pat.anchor_age + EXTRACT(YEAR FROM adm.admittime)::int - pat.anchor_year) AS admit_age,
        (pat.anchor_age + EXTRACT(EPOCH FROM (adm.admittime - MAKE_TIMESTAMP(pat.anchor_year, 1, 1, 0, 0, 0))) / 31557600.0) AS exact_admit_age,
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

CREATE INDEX idx_sa_aki_stay_master_stay ON __STAGE_SCHEMA__.stay_master (icu_stay_id);
CREATE INDEX idx_sa_aki_stay_master_patient ON __STAGE_SCHEMA__.stay_master (patient_id);
CREATE INDEX idx_sa_aki_stay_master_hadm ON __STAGE_SCHEMA__.stay_master (encounter_id);
ANALYZE __STAGE_SCHEMA__.stay_master;
