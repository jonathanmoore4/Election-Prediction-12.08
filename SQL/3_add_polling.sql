CREATE OR REPLACE TABLE with_polling AS
SELECT *,
FROM including_previous_years AS a
LEFT JOIN cleaned_polling AS b
ON a.election = b.Date;

ALTER TABLE with_polling
DROP COLUMN Date;

CREATE OR REPLACE TABLE with_polling AS
SELECT *,
CASE
    WHEN incumbent = previous_winner THEN 1
    ELSE 0
END AS supported_incumbent
FROM with_polling;

CREATE OR REPLACE TABLE with_polling AS
SELECT *,
CASE
    WHEN incumbent = 'lab' THEN Labour
    ELSE Conservative
END AS incumbent_polling
FROM with_polling;

CREATE OR REPLACE TABLE with_polling AS
SELECT *,
CASE
    WHEN incumbent = 'lab' THEN Conservative
    ELSE Labour
END AS opposition_polling
FROM with_polling;