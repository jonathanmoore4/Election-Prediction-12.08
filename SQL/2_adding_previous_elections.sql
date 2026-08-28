-- union the 2024 data with the cleaned 1997-2019 data to create a new table that includes all years
CREATE OR REPLACE TABLE all_years AS
SELECT *
FROM cleaned_1997_2019
UNION ALL
SELECT *
FROM cleaned_2024;

CREATE OR REPLACE TABLE all_years_2 AS
SELECT
    * REPLACE (COALESCE(natSW_share, 0) AS natSW_share)
FROM all_years;





CREATE OR REPLACE TABLE including_previous_years AS
SELECT 
    a.*,
    COALESCE(b.election, n2019.election, n2005.election, n2001.election) AS election,
    COALESCE(b.majority_proportion, n2019.majority_proportion, n2005.majority_proportion, n2001.majority_proportion) AS majority_proportion,
    COALESCE(b.constituency_id,n2019.constituency_id,n2005.constituency_id,n2001.constituency_id) AS constituency_id,
    COALESCE(b.winner, n2019.winner, n2005.winner, n2001.winner) AS winner,
    COALESCE(b.con_share, n2019.con_share, n2005.con_share, n2001.con_share) AS con_share,
    COALESCE(b.lib_share, n2019.lib_share, n2005.lib_share, n2001.lib_share) AS lib_share,
    COALESCE(b.lab_share, n2019.lab_share, n2005.lab_share, n2001.lab_share) AS lab_share,
    COALESCE(b.natSW_share, n2019.natSW_share, n2005.natSW_share, n2001.natSW_share) AS natSW_share
    
FROM all_years AS a
LEFT JOIN all_years_2 AS b
    ON a.constituency_id = b.constituency_id
    AND a.previous_election = b.election
LEFT JOIN cleaned_2019_notional AS n2019
    ON a.constituency_id = n2019.constituency_id
    AND a.previous_election = n2019.election
LEFT JOIN cleaned_2005_notional AS n2005
    ON a.constituency_id = n2005.constituency_id
    AND a.previous_election = n2005.election
LEFT JOIN cleaned_2001_notional AS n2001
    ON a.constituency_id = n2001.constituency_id
    AND a.previous_election = n2001.election;

CREATE OR REPLACE TABLE including_previous_years AS
SELECT *
FROM including_previous_years
WHERE election <> '1997';

ALTER TABLE including_previous_years
DROP COLUMN constituency_id_1;

ALTER TABLE including_previous_years
DROP COLUMN election_1;

ALTER TABLE including_previous_years
RENAME COLUMN winner_1 TO previous_winner;

ALTER TABLE including_previous_years
RENAME COLUMN majority_proportion_1 TO previous_majority_proportion;

ALTER TABLE including_previous_years
RENAME COLUMN con_share_1 TO previous_con_share;

ALTER TABLE including_previous_years
RENAME COLUMN lib_share_1 TO previous_lib_share;

ALTER TABLE including_previous_years
RENAME COLUMN lab_share_1 TO previous_lab_share;

ALTER TABLE including_previous_years
RENAME COLUMN natSW_share_1 TO previous_natSW_share;


