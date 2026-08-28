from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)

# Build paths relative to the project root so the script works from any
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed"

# Read the 2019 notional candidate-level results into a dataframe.
notional_results_2019 = pd.read_csv(
    RAW_DATA_DIR / "candidate-level-results-notional-general-election-12-12-2019.csv"
)

# Remove constituencies in Northern Ireland.
notional_results_2019 = notional_results_2019[
    ~notional_results_2019["Country name"].isin(["Northern Ireland"])
]

# Remove Chorley because the Commons Speaker row has no main party abbreviation.
notional_results_2019 = notional_results_2019[
    notional_results_2019["Constituency name"] != "Chorley"
]



# Use Scotland and Wales as their own region labels.
scotland_wales_mask = notional_results_2019["Country name"].isin(["Scotland", "Wales"])
notional_results_2019.loc[scotland_wales_mask, "English region name"] = (
    notional_results_2019.loc[scotland_wales_mask, "Country name"]
)

# Combine SNP and Plaid Cymru into one nationalist Scotland/Wales label.
notional_results_2019["Main party abbreviation"] = notional_results_2019[
    "Main party abbreviation"
].replace({"SNP": "natSW", "PC": "natSW"})




# Drop rows without a main party abbreviation so the pivot has unique party keys.
notional_results_2019 = notional_results_2019[
    notional_results_2019["Main party abbreviation"].notna()
]

notional_results_2019 = notional_results_2019.pivot(
    index=["Constituency geographic code", "English region name", "Constituency name"],
    columns="Main party abbreviation",
    values="Candidate vote share",
).reset_index()




notional_results_2019.columns.name = None

# Add election year for all notional 2019 results.
notional_results_2019["election"] = "2019_notional"

# Add current set of boundaries for all constituencies.
notional_results_2019["boundary_set"] = "2023-current"

# Add descriptor showing whether election is actual or notional.
notional_results_2019["election_type"] = "notional"

# Add the previous election used for comparison.
notional_results_2019["previous_election"] = "2017"



notional_results_2019 = notional_results_2019.rename(
    columns={
        "English region name": "country/region",
        "Constituency name": "constituency_name",
        "Constituency geographic code": "constituency_id",
        "Con": "con_share",
        "Lab": "lab_share",
        "LD": "lib_share",
        "natSW": "natSW_share",
    }
)

# Create winner and majority columns from the largest major-party vote share.
share_columns = ["con_share", "lib_share", "lab_share", "natSW_share"]
winner_labels = {
    "con_share": "con",
    "lib_share": "lib",
    "lab_share": "lab",
    "natSW_share": "natSW",
}

notional_results_2019["winner"] = (
    notional_results_2019[share_columns].idxmax(axis=1).map(winner_labels)
)

notional_results_2019["majority_proportion"] = notional_results_2019[
    share_columns
].max(axis=1)

# Keep only the columns needed for analysis.
notional_results_2019 = notional_results_2019[
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
        "lib_share",
        "lab_share",
        "natSW_share",
        "previous_election"
    ]
]

# Save the cleaned 2019 notional results as a CSV in the processed data folder.
notional_results_2019.to_csv(
    PROCESSED_DATA_DIR / "cleaned_2019_notional.csv",
    index=False,
)
