-- Pass polling features through; projections are calculated in step 5 after
-- national vote-share changes are available.
CREATE OR REPLACE VIEW model_data AS
SELECT * FROM with_polling;
