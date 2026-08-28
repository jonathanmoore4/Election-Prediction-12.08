from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)

# Build paths relative to the project root so the script works from any
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed"

# Read the 2005 notional candidate-level results into a dataframe.
notional_results_2005 = pd.read_csv(
    RAW_DATA_DIR / "candidate-level-results-notional-general-election-05-05-2005(1).csv"
)

cleaned_1997_2019 = pd.read_csv(
    PROCESSED_DATA_DIR / "cleaned_1997_2019.csv",
    dtype={"election": str},
)


# Remove constituencies in Ireland and Northern Ireland
notional_results_2005 = notional_results_2005[
    ~notional_results_2005["Country name"].isin(["Northern Ireland"])
]


# Use Scotland and Wales as their own region labels.
scotland_wales_mask = notional_results_2005["Country name"].isin(["Scotland", "Wales"])
notional_results_2005.loc[scotland_wales_mask, "English region name"] = (
    notional_results_2005.loc[scotland_wales_mask, "Country name"]
)


# Drop rows without a main party abbreviation so the pivot has unique party keys.
notional_results_2005 = notional_results_2005[
    notional_results_2005["Main party abbreviation"].notna()
]




notional_results_2005 = notional_results_2005.pivot(
    index=["Constituency geographic code", "English region name", "Constituency name"],
    columns="Main party abbreviation",
    values="Candidate vote share"
).reset_index()

notional_results_2005.columns.name = None

# Add election year for all 2024 results
notional_results_2005["election"] = "2005_notional"

# Add current set of boundaries for all constituencies
notional_results_2005["boundary_set"] = "2010-2017"

#Add descriptor showing whether election is actual or notional
notional_results_2005["election_type"] = "notional"




notional_results_2005 = notional_results_2005.rename(
    columns={
        "English region name": "country/region",
        "Constituency name": "constituency_name",
        "Constituency geographic code" : "constituency_id",
        "Con": "con_share",
        "Lab": "lab_share",
        "LD": "lib_share",
        "SNP": "natSW_share",
        "PC": "natSW_share",
    }
)




# Create winner column from the party with the largest vote share.
share_columns = ["con_share", "lib_share", "lab_share", "natSW_share"]
winner_labels = {
    "con_share": "con",
    "lib_share": "lib",
    "lab_share": "lab",
    "natSW_share": "natSW",
}
notional_results_2005["winner"] = (
    notional_results_2005[share_columns].idxmax(axis=1).map(winner_labels)
)
notional_results_2005["majority_proportion"] = notional_results_2005[
    share_columns
].max(axis=1)

# Keep only the columns needed for analysis
notional_results_2005 = notional_results_2005[
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
        "natSW_share" 

    ]
]

# scottish boundries didn't change between 2005 and 2010, so we can use the actual 2005 results for Scotland in the notional 2005 dataset.
scottish_2005_results = cleaned_1997_2019[
    (cleaned_1997_2019["country/region"] == "Scotland")
    & (cleaned_1997_2019["election"] == "2005")
].copy()

scottish_2005_results = scottish_2005_results[notional_results_2005.columns]
scottish_2005_results["election"] = "2005_notional"


notional_results_2005 = pd.concat(
    [notional_results_2005, scottish_2005_results],
    ignore_index=True,
)

# Add the previous election used for comparison.
notional_results_2005["previous_election"] = "2001"

# Save the cleaned 2005 notional results as a CSV in the processed data folder.
notional_results_2005.to_csv(
    PROCESSED_DATA_DIR / "cleaned_2005_notional.csv",
    index=False,
)
