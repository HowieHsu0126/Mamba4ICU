-- =============================================================================
-- 07_urine_output_hourly_48h.sql
-- Extracts 48-hour post-diagnosis hourly urine output volumes from
-- mimiciv_icu.outputevents for the SA-AKI candidate cohort (n = 20,317).
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
    FROM sa_aki_label.label_onset l
    INNER JOIN sa_aki_stage.stay_master s
        ON s.icu_stay_id = l.icu_stay_id
    WHERE l.has_sa_aki
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
  AND oe.charttime >= c.stay_anchor_time - INTERVAL '6 hour'
  AND oe.charttime <= GREATEST(c.obs_start + INTERVAL '48 hour', c.stay_anchor_time + INTERVAL '48 hour')
GROUP BY oe.stay_id, 2;
