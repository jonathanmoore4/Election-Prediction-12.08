CREATE OR REPLACE VIEW with_polling AS
SELECT
    previous_results.*,
    polling.Conservative,
    polling.Labour,
    polling.LD,
    polling.incumbent,
    CASE
        WHEN polling.incumbent = previous_results.previous_winner THEN 1
        ELSE 0
    END AS supported_incumbent,
    CASE
        WHEN polling.incumbent = 'lab' THEN polling.Labour
        ELSE polling.Conservative
    END AS incumbent_polling,
    CASE
        WHEN polling.incumbent = 'lab' THEN polling.Conservative
        ELSE polling.Labour
    END AS opposition_polling
FROM with_previous_results AS previous_results
LEFT JOIN polling
    ON previous_results.election = polling.election;
