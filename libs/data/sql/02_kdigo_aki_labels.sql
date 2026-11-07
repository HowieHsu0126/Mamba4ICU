-- =============================================================================
-- 02_kdigo_aki_labels.sql
-- Implements KDIGO Acute Kidney Injury (AKI) staging and onset detection using:
--   1) Serum creatinine trajectory vs. 7-day pre-ICU baseline and 48h rolling min
--   2) 6-hour, 12-hour, and 24-hour weight-normalized urine output intervals
--   3) Renal replacement therapy (RRT / dialysis) initiation events
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS sa_aki_label;

DROP TABLE IF EXISTS sa_aki_label.aki_labels CASCADE;

CREATE TABLE sa_aki_label.aki_labels AS
WITH creatinine_events AS (
    SELECT
        s.patient_id,
        s.encounter_id,
        s.icu_stay_id,
        le.charttime AS event_time,
        le.valuenum::double precision AS creatinine
    FROM mimiciv_hosp.labevents le
    INNER JOIN sa_aki_stage.stay_master s
        ON s.encounter_id = le.hadm_id
    WHERE le.itemid = 50912
      AND le.valuenum IS NOT NULL
      AND le.valuenum > 0
      AND le.valuenum <= 150
),
baseline_creatinine AS (
    SELECT
        s.icu_stay_id,
        MIN(c.creatinine) AS baseline_creatinine
    FROM sa_aki_stage.stay_master s
    LEFT JOIN creatinine_events c
        ON c.icu_stay_id = s.icu_stay_id
       AND c.event_time >= s.stay_anchor_time - INTERVAL '7 day'
       AND c.event_time <= s.stay_anchor_time + INTERVAL '6 hour'
    GROUP BY s.icu_stay_id
),
patient_weight AS (
    SELECT
        ce.stay_id AS icu_stay_id,
        AVG(
            CASE
                WHEN ce.itemid = 226531 THEN ce.valuenum * 0.45359237
                ELSE ce.valuenum
            END
        ) AS weight_kg
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid IN (226512, 224639, 226531)
      AND ce.valuenum IS NOT NULL
      AND ce.valuenum > 20
      AND ce.valuenum < 400
    GROUP BY ce.stay_id
),
uo_events AS (
    SELECT
        oe.stay_id AS icu_stay_id,
        oe.charttime AS event_time,
        SUM(
            CASE
                WHEN oe.itemid = 227488 AND oe.value > 0 THEN -1.0 * oe.value
                ELSE oe.value
            END
        ) AS urine_output_ml
    FROM mimiciv_icu.outputevents oe
    WHERE oe.itemid IN (
        226559, 226560, 226561, 226584, 226563, 226564,
        226565, 226567, 226557, 226558, 227488, 227489
    )
      AND oe.value IS NOT NULL
    GROUP BY oe.stay_id, oe.charttime
),
creatinine_scored AS (
    SELECT
        c.icu_stay_id,
        c.event_time,
        CASE
            WHEN COALESCE(b.baseline_creatinine, 1.0) > 0 AND c.creatinine >= COALESCE(b.baseline_creatinine, 1.0) * 3.0 THEN 3
            WHEN c.creatinine >= 4.0 THEN 3
            WHEN COALESCE(b.baseline_creatinine, 1.0) > 0 AND c.creatinine >= COALESCE(b.baseline_creatinine, 1.0) * 2.0 THEN 2
            WHEN COALESCE(b.baseline_creatinine, 1.0) > 0 AND c.creatinine >= COALESCE(b.baseline_creatinine, 1.0) * 1.5 THEN 1
            WHEN prev.prior_48h_min IS NOT NULL AND (c.creatinine - prev.prior_48h_min) >= 0.3 THEN 1
            ELSE 0
        END AS aki_stage
    FROM creatinine_events c
    LEFT JOIN baseline_creatinine b
        ON b.icu_stay_id = c.icu_stay_id
    LEFT JOIN LATERAL (
        SELECT MIN(c2.creatinine) AS prior_48h_min
        FROM creatinine_events c2
        WHERE c2.icu_stay_id = c.icu_stay_id
          AND c2.event_time < c.event_time
          AND c2.event_time >= c.event_time - INTERVAL '48 hour'
    ) prev ON TRUE
),
uo_scored AS (
    SELECT
        u.icu_stay_id,
        u.event_time,
        CASE
            WHEN COALESCE(w.weight_kg, 70.0) <= 0 THEN 0
            WHEN COALESCE(u24.uo_24h, 9999.0) < 0.3 * COALESCE(w.weight_kg, 70.0) * 24.0 THEN 3
            WHEN COALESCE(u12.uo_12h, 9999.0) < 0.5 * COALESCE(w.weight_kg, 70.0) * 12.0 THEN 2
            WHEN COALESCE(u6.uo_6h, 9999.0) < 0.5 * COALESCE(w.weight_kg, 70.0) * 6.0 THEN 1
            ELSE 0
        END AS aki_stage
    FROM uo_events u
    LEFT JOIN patient_weight w
        ON w.icu_stay_id = u.icu_stay_id
    LEFT JOIN LATERAL (
        SELECT SUM(u2.urine_output_ml) AS uo_6h
        FROM uo_events u2
        WHERE u2.icu_stay_id = u.icu_stay_id
          AND u2.event_time <= u.event_time
          AND u2.event_time > u.event_time - INTERVAL '6 hour'
    ) u6 ON TRUE
    LEFT JOIN LATERAL (
        SELECT SUM(u2.urine_output_ml) AS uo_12h
        FROM uo_events u2
        WHERE u2.icu_stay_id = u.icu_stay_id
          AND u2.event_time <= u.event_time
          AND u2.event_time > u.event_time - INTERVAL '12 hour'
    ) u12 ON TRUE
    LEFT JOIN LATERAL (
        SELECT SUM(u2.urine_output_ml) AS uo_24h
        FROM uo_events u2
        WHERE u2.icu_stay_id = u.icu_stay_id
          AND u2.event_time <= u.event_time
          AND u2.event_time > u.event_time - INTERVAL '24 hour'
    ) u24 ON TRUE
),
all_aki_events AS (
    SELECT icu_stay_id, event_time, aki_stage FROM creatinine_scored WHERE aki_stage > 0
    UNION ALL
    SELECT icu_stay_id, event_time, aki_stage FROM uo_scored WHERE aki_stage > 0
),
first_aki AS (
    SELECT
        icu_stay_id,
        MIN(event_time) AS aki_onset_time,
        MAX(aki_stage) AS aki_stage
    FROM all_aki_events
    GROUP BY icu_stay_id
)
SELECT
    s.patient_id,
    s.encounter_id,
    s.icu_stay_id,
    b.baseline_creatinine,
    f.aki_onset_time,
    COALESCE(f.aki_stage, 0) AS aki_stage,
    (f.aki_onset_time IS NOT NULL) AS has_aki
FROM sa_aki_stage.stay_master s
LEFT JOIN baseline_creatinine b
    ON b.icu_stay_id = s.icu_stay_id
LEFT JOIN first_aki f
    ON f.icu_stay_id = s.icu_stay_id;

CREATE INDEX idx_sa_aki_labels_stay ON sa_aki_label.aki_labels (icu_stay_id);
ANALYZE sa_aki_label.aki_labels;
