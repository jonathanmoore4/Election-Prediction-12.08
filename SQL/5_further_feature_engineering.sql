-- Add features derived from the matched previous-election results.
-- Missing previous shares naturally produce a NULL margin.
CREATE OR REPLACE VIEW further_model_data AS
SELECT
    model_data.*,
    previous_winning_party_last_election_vote_share
        - previous_second_party_last_election_vote_share AS previous_margin_1st_2nd
FROM model_data;
