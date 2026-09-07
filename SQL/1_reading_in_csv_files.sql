
-- These tables are the explicit SQL boundary: cleaned pandas dataframes in,
-- feature views out. The input_* names are registered by apply_SQL_queries.py.
CREATE OR REPLACE TABLE historical AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    majority_proportion,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_historical;

CREATE OR REPLACE TABLE notional_2001 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    majority_proportion,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_2001;

CREATE OR REPLACE TABLE notional_2005 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    majority_proportion,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_2005;

CREATE OR REPLACE TABLE notional_2019 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    majority_proportion,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_2019;

CREATE OR REPLACE TABLE results_2024 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    majority_proportion,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    natSW_share,
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_results_2024;

CREATE OR REPLACE TABLE polling AS
SELECT
    CAST(Date AS VARCHAR) AS election,
    Conservative,
    Labour,
    LD,
    incumbent
FROM input_polling;
