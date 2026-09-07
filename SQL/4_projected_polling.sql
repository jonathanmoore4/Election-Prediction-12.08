CREATE OR REPLACE VIEW previous_national_shares AS
SELECT
    election,
    AVG(CASE WHEN previous_winner = 'con' THEN 1.0 ELSE 0.0 END) AS previous_nat_con_share,
    AVG(CASE WHEN previous_winner = 'lib' THEN 1.0 ELSE 0.0 END) AS previous_nat_lib_share,
    AVG(CASE WHEN previous_winner = 'lab' THEN 1.0 ELSE 0.0 END) AS previous_nat_lab_share,
    AVG(CASE WHEN previous_winner = 'natSW' THEN 1.0 ELSE 0.0 END) AS previous_nat_natSW_share
FROM with_polling
GROUP BY
    election;

CREATE OR REPLACE VIEW model_data AS
SELECT
    polling_features.*,
    national_shares.previous_nat_con_share,
    national_shares.previous_nat_lib_share,
    national_shares.previous_nat_lab_share,
    national_shares.previous_nat_natSW_share,
    polling_features.previous_con_share + polling_features.Conservative - national_shares.previous_nat_con_share AS projected_con_share,
    polling_features.previous_lib_share + polling_features.LD - national_shares.previous_nat_lib_share AS projected_lib_share,
    polling_features.previous_lab_share + polling_features.Labour - national_shares.previous_nat_lab_share AS projected_lab_share
FROM with_polling AS polling_features
LEFT JOIN previous_national_shares AS national_shares
    ON polling_features.election = national_shares.election;
