-- National changes use actual previous national vote shares, not seat fractions.
-- Keep the existing winning/second-place shares as holder/challenger shares.
CREATE OR REPLACE VIEW further_model_data AS
WITH national_changes AS (
    SELECT model_data.*,
        con_polling - previous_con_national_vote_share AS con_national_change,
        lab_polling - previous_lab_national_vote_share AS lab_national_change,
        lib_polling - previous_lib_national_vote_share AS lib_national_change,
        -- There is no polling source for these categories; do not invent zeroes.
        CAST(NULL AS DOUBLE) AS natSW_national_change,
        CAST(NULL AS DOUBLE) AS oth_national_change,
        previous_winning_party_last_election_vote_share
            - previous_second_party_last_election_vote_share AS previous_margin_1st_2nd
    FROM model_data
)
SELECT national_changes.*,
    previous_con_share + con_national_change AS projected_con_share,
    previous_lib_share + lib_national_change AS projected_lib_share,
    previous_lab_share + lab_national_change AS projected_lab_share,
    CASE previous_winner
        WHEN 'con' THEN con_national_change
        WHEN 'lab' THEN lab_national_change
        WHEN 'lib' THEN lib_national_change
    END AS holder_national_swing,
    challenger.national_change AS challenger_national_swing,
    CASE previous_winner
        WHEN 'con' THEN con_polling
        WHEN 'lab' THEN lab_polling
        WHEN 'lib' THEN lib_polling
    END AS holder_polling,
    challenger.polling AS challenger_polling
FROM national_changes
-- Match the stored runner-up share rather than choosing the largest remaining
-- major party: an 'oth' runner-up must not be mistaken for a supported party.
-- Exclude the holder even in a tie. A fixed order resolves runner-up ties.
LEFT JOIN LATERAL (
    SELECT candidate.polling, candidate.national_change
    FROM (VALUES
        ('con', previous_con_share, con_polling, con_national_change, 1),
        ('lib', previous_lib_share, lib_polling, lib_national_change, 2),
        ('lab', previous_lab_share, lab_polling, lab_national_change, 3),
        ('natSW', previous_natSW_share, NULL, natSW_national_change, 4)
    ) AS candidate(party, vote_share, polling, national_change, tie_order)
    WHERE candidate.party <> previous_winner
        AND candidate.vote_share = previous_second_party_last_election_vote_share
    ORDER BY candidate.tie_order
    LIMIT 1
) AS challenger ON TRUE;
