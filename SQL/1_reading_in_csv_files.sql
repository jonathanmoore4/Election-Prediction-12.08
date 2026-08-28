
-- Read in all csvs
CREATE OR REPLACE TABLE cleaned_1997_2019 AS
SELECT * REPLACE (
    CAST(election AS VARCHAR) AS election,
    CAST(constituency_id AS VARCHAR) AS constituency_id
)
FROM read_csv_auto('Data/Processed/cleaned_1997_2019.csv', header = true);

CREATE OR REPLACE TABLE cleaned_2001_notional AS
SELECT * REPLACE (
    CAST(election AS VARCHAR) AS election,
    CAST(constituency_id AS VARCHAR) AS constituency_id
)
FROM read_csv_auto('Data/Processed/cleaned_2001_notional.csv', header = true);

CREATE OR REPLACE TABLE cleaned_2005_notional AS
SELECT * REPLACE (
    CAST(election AS VARCHAR) AS election,
    CAST(constituency_id AS VARCHAR) AS constituency_id
)
FROM read_csv_auto('Data/Processed/cleaned_2005_notional.csv', header = true);

CREATE OR REPLACE TABLE cleaned_2019_notional AS
SELECT * REPLACE (
    CAST(election AS VARCHAR) AS election,
    CAST(constituency_id AS VARCHAR) AS constituency_id
)
FROM read_csv_auto('Data/Processed/cleaned_2019_notional.csv', header = true);

CREATE OR REPLACE TABLE cleaned_2024 AS
SELECT * REPLACE (
    CAST(election AS VARCHAR) AS election,
    CAST(constituency_id AS VARCHAR) AS constituency_id
)
FROM read_csv_auto('Data/Processed/cleaned_2024.csv', header = true);

CREATE OR REPLACE TABLE cleaned_polling AS
SELECT *
FROM read_csv_auto('Data/Processed/cleaned_polling.csv', header = true);
