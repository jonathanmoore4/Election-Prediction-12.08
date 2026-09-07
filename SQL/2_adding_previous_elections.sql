CREATE OR REPLACE VIEW actual_results AS
SELECT
    constituency_name,
    "country/region",
    election,
    majority_proportion,
    winner,
    constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    previous_election
FROM historical
UNION ALL
SELECT
    constituency_name,
    "country/region",
    election,
    majority_proportion,
    winner,
    constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    previous_election
FROM results_2024;

CREATE OR REPLACE VIEW previous_result_lookup AS
SELECT
    election,
    majority_proportion,
    winner,
    constituency_id,
    con_share,
    lib_share,
    lab_share,
    COALESCE(natSW_share, 0) AS natSW_share
FROM actual_results
UNION ALL
SELECT election, majority_proportion, winner, constituency_id, con_share, lib_share, lab_share, COALESCE(natSW_share, 0)
FROM notional_2001
UNION ALL
SELECT election, majority_proportion, winner, constituency_id, con_share, lib_share, lab_share, COALESCE(natSW_share, 0)
FROM notional_2005
UNION ALL
SELECT election, majority_proportion, winner, constituency_id, con_share, lib_share, lab_share, COALESCE(natSW_share, 0)
FROM notional_2019;

CREATE OR REPLACE VIEW with_previous_results AS
SELECT
    current_results.constituency_name,
    current_results."country/region",
    current_results.election,
    current_results.majority_proportion,
    current_results.winner,
    current_results.constituency_id,
    current_results.boundary_set,
    current_results.election_type,
    current_results.con_share,
    current_results.lib_share,
    current_results.lab_share,
    current_results.natSW_share,
    current_results.previous_election,
    previous_results.majority_proportion AS previous_majority_proportion,
    previous_results.winner AS previous_winner,
    previous_results.con_share AS previous_con_share,
    previous_results.lib_share AS previous_lib_share,
    previous_results.lab_share AS previous_lab_share,
    previous_results.natSW_share AS previous_natSW_share
FROM actual_results AS current_results
LEFT JOIN previous_result_lookup AS previous_results
    ON current_results.constituency_id = previous_results.constituency_id
    AND current_results.previous_election = previous_results.election
WHERE current_results.election <> '1997';
