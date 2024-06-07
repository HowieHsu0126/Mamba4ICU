SELECT DISTINCT
    subject_id
FROM mimiciv_derived.kdigo_stages 
WHERE aki_stage != 0
ORDER BY
    subject_id;