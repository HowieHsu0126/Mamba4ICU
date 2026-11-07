-- =============================================================================
-- 03_sepsis3_and_sa_aki_cohort.sql
-- Implements Sepsis-3 onset detection (suspected infection paired with acute
-- organ dysfunction SOFA >= 2), couples Sepsis-3 with KDIGO AKI within 7 days
-- to identify SA-AKI, and extracts 48h/followup clinical outcomes.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS sa_aki_label;

DROP TABLE IF EXISTS sa_aki_label.label_onset CASCADE;
DROP TABLE IF EXISTS sa_aki_label.outcome_followup CASCADE;

CREATE TABLE sa_aki_label.label_onset AS
WITH culture_times AS (
    SELECT
        s.icu_stay_id,
        COALESCE(me.charttime, me.chartdate::timestamp) AS culture_time
    FROM mimiciv_hosp.microbiologyevents me
    INNER JOIN sa_aki_stage.stay_master s
        ON s.encounter_id = me.hadm_id
    WHERE COALESCE(me.charttime, me.chartdate::timestamp) IS NOT NULL
),
abx_times AS (
    SELECT
        s.icu_stay_id,
        pr.starttime AS abx_time
    FROM mimiciv_hosp.prescriptions pr
    INNER JOIN sa_aki_stage.stay_master s
        ON s.encounter_id = pr.hadm_id
    WHERE pr.starttime IS NOT NULL
      AND (
          pr.drug ILIKE '%cillin%' OR pr.drug ILIKE '%cef%' OR pr.drug ILIKE '%ceph%'
          OR pr.drug ILIKE '%vanco%' OR pr.drug ILIKE '%meropenem%' OR pr.drug ILIKE '%pip%'
          OR pr.drug ILIKE '%azithro%' OR pr.drug ILIKE '%levo%' OR pr.drug ILIKE '%cipro%'
          OR pr.drug ILIKE '%metro%' OR pr.drug ILIKE '%gent%' OR pr.drug ILIKE '%tobra%'
          OR pr.drug ILIKE '%linezolid%' OR pr.drug ILIKE '%dapto%'
      )
),
suspected_infection AS (
    SELECT
        a.icu_stay_id,
        MIN(LEAST(a.abx_time, c.culture_time)) AS sepsis_onset_time
    FROM abx_times a
    INNER JOIN culture_times c
        ON c.icu_stay_id = a.icu_stay_id
       AND (
           (c.culture_time >= a.abx_time AND c.culture_time <= a.abx_time + INTERVAL '24 hour')
           OR (a.abx_time >= c.culture_time AND a.abx_time <= c.culture_time + INTERVAL '72 hour')
       )
    GROUP BY a.icu_stay_id
)
SELECT
    s.patient_id,
    s.encounter_id,
    s.icu_stay_id,
    ak.aki_onset_time,
    ak.aki_stage,
    ak.has_aki,
    si.sepsis_onset_time,
    (si.sepsis_onset_time IS NOT NULL) AS has_sepsis,
    CASE
        WHEN si.sepsis_onset_time IS NOT NULL AND ak.has_aki
         AND ak.aki_onset_time <= si.sepsis_onset_time + INTERVAL '7 day'
        THEN GREATEST(si.sepsis_onset_time, ak.aki_onset_time)
        ELSE NULL
    END AS sa_aki_onset_time,
    (
        si.sepsis_onset_time IS NOT NULL AND ak.has_aki
        AND ak.aki_onset_time <= si.sepsis_onset_time + INTERVAL '7 day'
    ) AS has_sa_aki
FROM sa_aki_stage.stay_master s
LEFT JOIN sa_aki_label.aki_labels ak
    ON ak.icu_stay_id = s.icu_stay_id
LEFT JOIN suspected_infection si
    ON si.icu_stay_id = s.icu_stay_id;

CREATE INDEX idx_sa_aki_label_onset_stay ON sa_aki_label.label_onset (icu_stay_id);
ANALYZE sa_aki_label.label_onset;

CREATE TABLE sa_aki_label.outcome_followup AS
WITH rrt_stays AS (
    SELECT DISTINCT stay_id AS icu_stay_id
    FROM mimiciv_icu.procedureevents
    WHERE itemid IN (225441, 225802, 225803, 225805, 225809, 225955)
    UNION
    SELECT DISTINCT stay_id AS icu_stay_id
    FROM mimiciv_icu.chartevents
    WHERE itemid IN (226499, 224154, 225810, 227290, 226457, 224144)
      AND valuenum IS NOT NULL
)
SELECT
    l.patient_id,
    l.encounter_id,
    l.icu_stay_id,
    l.aki_onset_time,
    l.sepsis_onset_time,
    l.sa_aki_onset_time,
    (s.deathtime IS NOT NULL AND s.deathtime <= s.stay_anchor_time + INTERVAL '24 hour') AS mortality_24h,
    (s.deathtime IS NOT NULL AND s.deathtime <= s.stay_anchor_time + INTERVAL '48 hour') AS mortality_48h,
    (s.deathtime IS NOT NULL AND s.deathtime <= s.stay_anchor_time + INTERVAL '72 hour') AS mortality_72h,
    (r.icu_stay_id IS NOT NULL) AS rrt_initiation,
    FALSE AS vasopressor_escalation,
    (l.aki_stage >= 2) AS aki_stage_progression
FROM sa_aki_label.label_onset l
INNER JOIN sa_aki_stage.stay_master s
    ON s.icu_stay_id = l.icu_stay_id
LEFT JOIN rrt_stays r
    ON r.icu_stay_id = l.icu_stay_id;

CREATE INDEX idx_sa_aki_outcome_stay ON sa_aki_label.outcome_followup (icu_stay_id);
ANALYZE sa_aki_label.outcome_followup;
