from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)

# Read csv file in and save as a data frame
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed"


election_results_1918_2019 = pd.read_csv(
    RAW_DATA_DIR / "1918-2019election_results.csv",
    encoding="cp1252",
)

# Keep only elections from 1997 onwards
election_results_1918_2019 = election_results_1918_2019[
    election_results_1918_2019["election"].isin(
        ["1997", "2001", "2005", "2010", "2015", "2017", "2019"]
    )
]

# Remove constituencies in Ireland and Northern Ireland
election_results_1918_2019 = election_results_1918_2019[
    ~election_results_1918_2019["country/region"].isin(["Ireland", "Northern Ireland"])
]


# Create winner and majority share columns
share_columns = ["con_share", "lib_share", "lab_share", "natSW_share", "oth_share"]
winner_labels = {
    "con_share": "con",
    "lib_share": "lib",
    "lab_share": "lab",
    "natSW_share": "natSW",
    "oth_share": "oth"
}

election_results_1918_2019["majority_proportion"] = election_results_1918_2019[
    share_columns
].max(axis=1)
has_share = election_results_1918_2019[share_columns].notna().any(axis=1)
election_results_1918_2019.loc[has_share, "winner"] = (
    election_results_1918_2019.loc[has_share, share_columns]
    .idxmax(axis=1)
    .map(winner_labels)
)



# Add descriptor showing whether election is actual or notional
election_results_1918_2019["election_type"] = "actual"

# only use selected columns
election_results_1918_2019 = election_results_1918_2019[
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

election_results_1918_2019["boundary_set"] = election_results_1918_2019["boundary_set"].replace("2010-2017", "2010-2019")


previous_election_map = {
    "1997": pd.NA,
    "2001": "1997",
    "2005": "2001_notional",
    "2010": "2005_notional",
    "2015": "2010",
    "2017": "2015",
    "2019": "2017",
}

election_results_1918_2019["previous_election"] = election_results_1918_2019[
    "election"
].map(previous_election_map)



election_results_1918_2019.to_csv(
    PROCESSED_DATA_DIR / "cleaned_1997_2019.csv",
    index=False,
)
