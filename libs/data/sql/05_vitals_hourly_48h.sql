-- =============================================================================
-- 05_vitals_hourly_48h.sql
-- Extracts 48-hour post-diagnosis hourly vital sign series (Heart Rate,
-- Respiratory Rate, Temperature in Celsius, Systolic BP, Diastolic BP, SpO2)
-- from mimiciv_icu.chartevents.
-- =============================================================================

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
    FROM __LABEL_SCHEMA__.label_onset_mimic l
    INNER JOIN __STAGE_SCHEMA__.stay_master_mimic s
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
  AND ce.charttime >= c.stay_anchor_time - INTERVAL '6 hour'
  AND ce.charttime <= GREATEST(c.obs_start + INTERVAL '48 hour', c.stay_anchor_time + INTERVAL '48 hour')
GROUP BY ce.stay_id, 2;
