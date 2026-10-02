CREATE OR REPLACE VIEW with_polling AS
SELECT
    previous_results.*,
    polling.con_polling,
    polling.lab_polling,
    polling.lib_polling,
    polling.incumbent,
    CASE
        WHEN polling.incumbent = previous_results.previous_winner THEN 1
        ELSE 0
    END AS supported_incumbent,
    CASE
        WHEN polling.incumbent = 'lab' THEN polling.lab_polling
        ELSE polling.con_polling
    END AS incumbent_polling,
    CASE
        WHEN polling.incumbent = 'lab' THEN polling.con_polling
        ELSE polling.lab_polling
    END AS opposition_polling
FROM with_previous_results AS previous_results
LEFT JOIN polling
    ON previous_results.election = polling.election;
