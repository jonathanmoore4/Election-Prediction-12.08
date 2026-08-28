CREATE OR REPLACE TABLE previous_national_shares AS
SELECT
    election,
    AVG(CASE WHEN previous_winner = 'con' THEN 1.0 ELSE 0.0 END) AS previous_nat_con_share,
    AVG(CASE WHEN previous_winner = 'lib' THEN 1.0 ELSE 0.0 END) AS previous_nat_lib_share,
    AVG(CASE WHEN previous_winner = 'lab' THEN 1.0 ELSE 0.0 END) AS previous_nat_lab_share,
    AVG(CASE WHEN previous_winner = 'natSW' THEN 1.0 ELSE 0.0 END) AS previous_nat_natSW_share
FROM with_polling
GROUP BY
    election
ORDER BY
    election;

CREATE OR REPLACE TABLE with_polling_2 AS
SELECT
    a.*,
    b.previous_nat_con_share,
    b.previous_nat_lib_share,
    b.previous_nat_lab_share,
    b.previous_nat_natSW_share
FROM with_polling AS a
LEFT JOIN previous_national_shares AS b
    ON a.election = b.election;

CREATE OR REPLACE TABLE with_polling_2 AS
SELECT *,
    previous_con_share + Conservative - previous_nat_con_share AS projected_con_share, 
    previous_lib_share + LD - previous_nat_lib_share AS projected_lib_share,
    previous_lab_share + Labour - previous_nat_lab_share AS projected_lab_share
FROM with_polling_2;

