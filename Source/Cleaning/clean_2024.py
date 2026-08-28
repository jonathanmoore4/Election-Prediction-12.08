from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)

# Read csv file in and save as a data frame
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed"

election_results_2024 = pd.read_csv(RAW_DATA_DIR / "2024 election data.csv")

# Rename columns to match the rest of the cleaned election data
election_results_2024 = election_results_2024.rename(
    columns={
        "First party": "winner",
        "Region name": "country/region",
        "Constituency name": "constituency_name",
        "ONS ID" : "constituency_id"
    }
)

# Remove constituencies in Ireland and Northern Ireland
election_results_2024 = election_results_2024[
    ~election_results_2024["country/region"].isin(["Ireland", "Northern Ireland"])
]





# Define each party's vote share as party votes divided by valid votes.
party_vote_columns = {
    "Con": "con_share",
    "Lab": "lab_share",
    "LD": "lib_share",
    "RUK": "ruk_share",
    "Green": "green_share",
    "SNP": "SNP_share",
    "PC": "PC_share",
    "DUP": "dup_share",
    "SF": "sf_share",
    "SDLP": "sdlp_share",
    "UUP": "uup_share",
    "APNI": "apni_share",
    "All other candidates": "oth_share",
}

for vote_column, share_column in party_vote_columns.items():
    election_results_2024[share_column] = (
        election_results_2024[vote_column] / election_results_2024["Valid votes"]
    )

election_results_2024["natSW_share"] = election_results_2024[
    ["SNP_share", "PC_share"]
].max(axis=1)

election_results_2024 = election_results_2024.drop(columns=["SNP_share", "PC_share"])

# Define majority proportion as the largest party vote share.
share_columns = [
    column for column in party_vote_columns.values() if column not in ["SNP_share", "PC_share"]
] + ["natSW_share"]
election_results_2024["majority_proportion"] = election_results_2024[
    share_columns
].max(axis=1)




# Add election year for all 2024 results
election_results_2024["election"] = "2024"

# Add current set of boundaries for all constituencies
election_results_2024["boundary_set"] = "2023-current"

#Add descriptor showing whether election is actual or notional
election_results_2024["election_type"] = "actual"

election_results_2024["previous_election"] = "2019_notional"

# Map 2024 party abbreviations into the shared party labels used by the
# cleaned historical election datasets.
winner_mapping = {
    "Lab": "lab",
    "Con": "con",
    "Ind": "oth",
    "RUK": "oth",
    "SNP": "natSW",
    "PC": "natSW",
    "LD": "lib",
    "Green": "oth",
    "Spk" : "oth"
}
election_results_2024["winner"] = election_results_2024["winner"].replace(winner_mapping)


# Keep only the columns needed for analysis
election_results_2024 = election_results_2024[
    [
        "constituency_name",
        "country/region",
        "election",
        "majority_proportion",
        "winner",
        "constituency_id",
        "boundary_set",
        "election_type",
        "con_share",
        "lab_share",
        "lib_share",
        "natSW_share",
        "previous_election"
    ]
]



election_results_2024.to_csv(
    PROCESSED_DATA_DIR / "cleaned_2024.csv",
    index=False,
)
