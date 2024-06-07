-- Drop the table if it exists
DROP TABLE IF EXISTS sa_aki_cohorts.demographics;

-- Create the sa_aki_cohorts.demographics table
CREATE TABLE sa_aki_cohorts.demographics AS
-- Select and calculate demographics for SA-AKI patients
WITH avg_height AS (
    SELECT
        ht.subject_id,
        AVG(ht.height) AS avg_height
    FROM
        mimiciv_derived.height ht
    GROUP BY
        ht.subject_id
),
avg_weight AS (
    SELECT
        wt.stay_id,
        AVG(wt.weight) AS avg_weight
    FROM
        mimiciv_derived.weight_durations wt
    GROUP BY
        wt.stay_id
)
SELECT DISTINCT
    sa.subject_id,
    sa.hadm_id,
    sa.stay_id,
    FLOOR(EXTRACT(EPOCH FROM (ad.admittime - MAKE_TIMESTAMP(pa.anchor_year, 1, 1, 0, 0, 0))) / 31557600 + pa.anchor_age) AS age, -- Calculate age in years
    pa.gender, -- Get gender
    ad.race, -- Get race
    ad.admission_type -- Get admission type
FROM
    sa_aki sa
    INNER JOIN mimiciv_hosp.admissions ad ON sa.subject_id = ad.subject_id
        AND sa.hadm_id = ad.hadm_id
    INNER JOIN mimiciv_hosp.patients pa ON sa.subject_id = pa.subject_id
ORDER BY
    sa.subject_id,
    sa.hadm_id,
    sa.stay_id;

-- Order results for readability
