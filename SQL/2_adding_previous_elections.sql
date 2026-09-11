-- Step 1: combine historical actual election results with the 2024 results.
-- CREATE OR REPLACE refreshes the view definition whenever this script runs.
-- A view is a saved query over the source tables, rather than a copied table.
CREATE OR REPLACE VIEW actual_results AS
-- Keep constituency details, election metadata, winner, and party vote shares.
-- majority_proportion is the winning vote share calculated by the cleaners,
-- not the winning party's lead over the runner-up.
-- previous_election identifies the actual or notional comparison to use later.
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
-- Append every 2024 row. UNION ALL preserves duplicates and matches columns
-- by position, so both SELECT lists must use the same column order.
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

-- Step 2: collect the results that can serve as a previous-election comparison.
-- Only fields needed for matching and previous-result features are selected.
-- Include actual results and all four notional datasets. Keep names as well
-- as IDs because the 1992 notional data can only be matched by name.
CREATE OR REPLACE VIEW previous_result_lookup AS
SELECT
    election,
    majority_proportion,
    winner,
    constituency_id,
    con_share,
    lib_share,
    lab_share,
    -- Replace a missing nationalist vote share with zero in this lookup only.
    COALESCE(natSW_share, 0) AS natSW_share,
    constituency_name
FROM actual_results
-- Add 1992 votes on 1997-2001 boundaries for comparison with 1997.
-- A typed NULL fills the ID position without inventing a constituency ID.
UNION ALL
SELECT election, majority_proportion, winner, CAST(NULL AS VARCHAR) AS constituency_id,
       con_share, lib_share, lab_share, COALESCE(natSW_share, 0), constituency_name
FROM notional_1992
-- Add 2001 votes expressed on 2005 boundaries for comparison with 2005.
UNION ALL
SELECT election, majority_proportion, winner, constituency_id, con_share, lib_share, lab_share, COALESCE(natSW_share, 0), constituency_name
FROM notional_2001
-- Add 2005 notional votes for comparison with the 2010 election.
UNION ALL
SELECT election, majority_proportion, winner, constituency_id, con_share, lib_share, lab_share, COALESCE(natSW_share, 0), constituency_name
FROM notional_2005
-- Add 2019 votes expressed on the new boundaries for comparison with 2024.
-- Each notional SELECT also replaces missing nationalist shares with zero.
UNION ALL
SELECT election, majority_proportion, winner, constituency_id, con_share, lib_share, lab_share, COALESCE(natSW_share, 0), constituency_name
FROM notional_2019;

-- Step 3: attach the selected previous election's results to each actual result.
CREATE OR REPLACE VIEW with_previous_results AS
SELECT
    -- Preserve the current election's constituency details and result fields.
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
    -- Prefix the matched comparison fields with previous_ to distinguish them
    -- from the current result. No vote-share changes are calculated here.
    previous_results.majority_proportion AS previous_majority_proportion,
    previous_results.winner AS previous_winner,
    previous_results.con_share AS previous_con_share,
    previous_results.lib_share AS previous_lib_share,
    previous_results.lab_share AS previous_lab_share,
    previous_results.natSW_share AS previous_natSW_share
-- Aliases distinguish the current result from its comparison result.
FROM actual_results AS current_results
-- A LEFT JOIN keeps current rows even when no previous result matches;
-- the previous_ result fields then contain NULL, not zero.
-- Multiple lookup rows with the same matching name/ID and election would
-- duplicate a current row. The lookup should have one row per matching key.
LEFT JOIN previous_result_lookup AS previous_results
    -- Match the explicit comparison label, which may be e.g. 2001_notional.
    ON current_results.previous_election = previous_results.election
    AND (
        -- Only 1992 notional comparisons use constituency names.
        -- Ignore capitalisation and leading/trailing spaces; other spelling
        -- differences still need explicit corrections in the source cleaners.
        (
            previous_results.election = '1992_notional'
            AND LOWER(TRIM(current_results.constituency_name))
                = LOWER(TRIM(previous_results.constituency_name))
        )
        OR
        -- All other comparisons retain the existing constituency-ID match.
        (
            previous_results.election <> '1992_notional'
            AND current_results.constituency_id = previous_results.constituency_id
        )
    )
-- Keep 1983 in actual_results and previous_result_lookup for 1987 comparisons,
-- but exclude it as a current election from downstream features and model data.
WHERE current_results.election <> '1983';
-- Include 1997 now that its 1992 notional comparison is available.
