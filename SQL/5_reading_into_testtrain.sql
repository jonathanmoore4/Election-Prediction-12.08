CREATE OR REPLACE TABLE TRAIN AS
SELECT *,
FROM with_polling_2
WHERE election != '2024';

CREATE OR REPLACE TABLE TEST AS
SELECT *,
FROM with_polling_2
WHERE election = '2024';

COPY TRAIN 
TO 'C:/Users/jonny/PycharmProjects/Election Prediction 12.08/data/test train/TRAIN.csv'
WITH (HEADER, DELIMITER ',');

COPY TEST 
TO 'C:/Users/jonny/PycharmProjects/Election Prediction 12.08/data/test train/TEST.csv'
WITH (HEADER, DELIMITER ',');
