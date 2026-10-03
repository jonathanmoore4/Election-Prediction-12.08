
-- These views are the explicit src/election/sql boundary: cleaned pandas dataframes in,
-- feature views out. The input_* names are loaded by apply_sql_queries.py.
CREATE VIEW historical AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    winning_party_vote_share,
    second_party_vote_share,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    "natSW_share",
    con_national_vote_share,
    lab_national_vote_share,
    lib_national_vote_share,
    "natSW_national_vote_share",
    oth_national_vote_share,
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_historical;

-- Load the 1992 results expressed on 1997-2001 boundaries.
-- The 1992 cleaner supplies constituency names but no constituency IDs or regions.
CREATE VIEW notional_1992 AS
SELECT
    constituency_name,
    CAST(election AS VARCHAR) AS election,
    winning_party_vote_share,
    second_party_vote_share,
    winner,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    "natSW_share",
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_1992;

CREATE VIEW notional_2001 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    winning_party_vote_share,
    second_party_vote_share,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    "natSW_share",
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_2001;

CREATE VIEW notional_2005 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    winning_party_vote_share,
    second_party_vote_share,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    "natSW_share",
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_2005;

CREATE VIEW notional_2019 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    winning_party_vote_share,
    second_party_vote_share,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    "natSW_share",
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_notional_2019;

CREATE VIEW results_2024 AS
SELECT
    constituency_name,
    "country/region",
    CAST(election AS VARCHAR) AS election,
    winning_party_vote_share,
    second_party_vote_share,
    winner,
    CAST(constituency_id AS VARCHAR) AS constituency_id,
    boundary_set,
    election_type,
    con_share,
    lib_share,
    lab_share,
    "natSW_share",
    CAST(previous_election AS VARCHAR) AS previous_election
FROM input_results_2024;

CREATE VIEW polling AS
SELECT
    CAST("Date" AS VARCHAR) AS election,
    con_polling,
    lab_polling,
    lib_polling,
    incumbent
FROM input_polling;
