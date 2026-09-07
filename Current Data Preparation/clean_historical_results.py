import pandas as pd

def clean_historical_fun(df):

    # Keep only elections from 1997 onwards
    df = df[df["election"].isin(["1997", "2001", "2005", "2010", "2015", "2017", "2019"])]

    # Remove constituencies in Ireland and Northern Ireland
    df = df[~df["country/region"].isin(["Ireland", "Northern Ireland"])].copy()

    # Create winner and majority share columns
    share_columns = ["con_share", "lib_share", "lab_share", "natSW_share", "oth_share"]
    winner_labels = {
        "con_share": "con",
        "lib_share": "lib",
        "lab_share": "lab",
        "natSW_share": "natSW",
        "oth_share": "oth"
    }

    # Calculate the majority proportion for each row
    df["majority_proportion"] = df[share_columns].max(axis=1)

    # Determine the winner based on the maximum share column
    has_share = df[share_columns].notna().any(axis=1)
    df.loc[has_share, "winner"] = (
        df.loc[has_share, share_columns]
        .idxmax(axis=1)
        .map(winner_labels)
    )

    # Add descriptor showing whether election is actual or notional
    df["election_type"] = "actual"

    # only use selected columns
    df = df[
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

    df["boundary_set"] = df["boundary_set"].replace("2010-2017", "2010-2019")

    previous_election_map = {
        "1997": pd.NA,
        "2001": "1997",
        "2005": "2001_notional",
        "2010": "2005_notional",
        "2015": "2010",
        "2017": "2015",
        "2019": "2017",
    }

    df["previous_election"] = df["election"].map(previous_election_map)

    return df
