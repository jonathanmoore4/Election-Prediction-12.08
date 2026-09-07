CREATE OR REPLACE VIEW train_data AS
SELECT *
FROM model_data
WHERE election != '2024';

CREATE OR REPLACE VIEW test_data AS
SELECT *
FROM model_data
WHERE election = '2024';
