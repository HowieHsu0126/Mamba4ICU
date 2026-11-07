-- =============================================================================
-- 06_labs_hourly_48h.sql
-- Extracts 48-hour post-diagnosis hourly laboratory measurements (WBC, BUN,
-- Creatinine, Platelet, Sodium, Potassium, Chloride, Bicarbonate, Phosphate,
-- Hemoglobin, Lactate, PTT, Anion Gap, Hematocrit, Bilirubin) from
-- mimiciv_hosp.labevents.
-- =============================================================================

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
    FROM __LABEL_SCHEMA__.label_onset_mimic l
    INNER JOIN __STAGE_SCHEMA__.stay_master_mimic s
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
  AND le.charttime >= c.stay_anchor_time - INTERVAL '24 hour'
  AND le.charttime <= GREATEST(c.obs_start + INTERVAL '48 hour', c.stay_anchor_time + INTERVAL '48 hour')
GROUP BY c.icu_stay_id, 2;
