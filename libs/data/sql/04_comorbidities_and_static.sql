-- =============================================================================
-- 04_comorbidities_and_static.sql
-- Extracts static demographic attributes, clinical outcomes, baseline creatinine,
-- and 7 ICD-9/ICD-10 comorbidities (plus ESKD and SAPS II malignancy flags)
-- for the Initial SA-AKI candidate cohort (n = 20,317).
-- =============================================================================

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
        (
            l.aki_onset_time >= l.sepsis_onset_time
            AND l.aki_onset_time <= l.sepsis_onset_time + INTERVAL '7 day'
        ) AS within_7d,
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
        s.admit_age,
        s.exact_admit_age,
        s.dod,
        s.is_first_stay,
        ROW_NUMBER() OVER (
            PARTITION BY l.patient_id
            ORDER BY s.stay_anchor_time, l.icu_stay_id
        ) AS sa_aki_stay_rank,
        ak.baseline_creatinine,
        ak.max_creatinine,
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
    FROM sa_aki_label.label_onset l
    INNER JOIN sa_aki_stage.stay_master s
        ON s.icu_stay_id = l.icu_stay_id
    LEFT JOIN sa_aki_label.aki_labels ak
        ON ak.icu_stay_id = l.icu_stay_id
    LEFT JOIN sa_aki_label.outcome_followup o
        ON o.icu_stay_id = l.icu_stay_id
    WHERE l.has_sa_aki
),
comorb AS (
    SELECT
        d.hadm_id,
        MAX(CASE
            WHEN d.seq_num <= 15 AND (
                (d.icd_version = 9 AND SUBSTR(BTRIM(d.icd_code), 1, 3) BETWEEN '401' AND '405')
                OR (d.icd_version = 10 AND SUBSTR(BTRIM(d.icd_code), 1, 3) BETWEEN 'I10' AND 'I15')
            )
            THEN 1 ELSE 0 END) AS hypertension,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) = '428'
                 OR SUBSTR(BTRIM(d.icd_code), 1, 5) IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) BETWEEN '4254' AND '4259'
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('I43','I50')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','P290')
            ))
            THEN 1 ELSE 0 END) AS congestive_heart_failure,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(BTRIM(d.icd_code), 1, 3) = '250')
              OR (d.icd_version = 10 AND SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('E10','E11','E12','E13','E14'))
            THEN 1 ELSE 0 END) AS diabetes_mellitus,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('570','571')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('0706','0709','5733','5734','5738','5739','V427','4560','4561','4562')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) BETWEEN '5722' AND '5728'
                 OR SUBSTR(BTRIM(d.icd_code), 1, 5) IN ('07022','07023','07032','07033','07044','07054')
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('B18','K70','K71','K72','K73','K74','K76')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('I850','I859','I864','I982','Z944')
            ))
            THEN 1 ELSE 0 END) AS liver_disease,
        MAX(CASE
            WHEN (d.icd_version = 9 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('440','441')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('0930','4373','4471','5571','5579','V434')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) BETWEEN '4431' AND '4439'
            ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('I70','I71','I73')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('I771','I790','I792','K551','K558','K559','Z958','Z959')
            ))
            THEN 1 ELSE 0 END) AS peripheral_vascular_disease,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('042','043','044'))
              OR (d.icd_version = 10 AND SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('B20','B21','B22','B24'))
            THEN 1 ELSE 0 END) AS hiv_infection,
        MAX(CASE
            WHEN d.seq_num <= 24 AND (
                (d.icd_version = 9 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('582','585','586','V56')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('5880','V420','V451')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) BETWEEN '5830' AND '5837'
                 OR SUBSTR(BTRIM(d.icd_code), 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')
                ))
              OR (d.icd_version = 10 AND (
                    SUBSTR(BTRIM(d.icd_code), 1, 3) IN ('N18','N19')
                 OR SUBSTR(BTRIM(d.icd_code), 1, 4) IN ('I120','I131','N032','N033','N034','N035','N036','N037','N052','N053','N054','N055','N056','N057','N250','Z490','Z491','Z492','Z940','Z992')
                ))
            )
            THEN 1 ELSE 0 END) AS ckd,
        MAX(CASE
            WHEN BTRIM(d.icd_code) IN ('5856', 'N186')
            THEN 1 ELSE 0 END) AS has_eskd_code,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(BTRIM(d.icd_code), 1, 3) BETWEEN '200' AND '208')
              OR (d.icd_version = 10 AND SUBSTR(BTRIM(d.icd_code), 1, 3) BETWEEN 'C81' AND 'C96')
            THEN 1 ELSE 0 END) AS hematologic_malignancy,
        MAX(CASE
            WHEN (d.icd_version = 9 AND SUBSTR(BTRIM(d.icd_code), 1, 3) BETWEEN '196' AND '199')
              OR (d.icd_version = 10 AND SUBSTR(BTRIM(d.icd_code), 1, 3) BETWEEN 'C77' AND 'C80')
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
    CASE
        WHEN COALESCE(cb.has_eskd_code, 0) = 1
         AND (
             COALESCE(cs.baseline_creatinine, cs.max_creatinine, 0.0)
             + 1e-4 * COALESCE(cs.max_creatinine, 0.0)
             + 1e-7 * cs.icu_los_days
         ) >= 7.20072018
        THEN 1 ELSE 0
    END AS has_eskd,
    COALESCE(cb.hematologic_malignancy, 0) AS hematologic_malignancy,
    COALESCE(cb.metastatic_cancer, 0) AS metastatic_cancer
FROM cand_stays cs
LEFT JOIN comorb cb
    ON cb.hadm_id = cs.encounter_id
ORDER BY cs.icu_stay_id;
