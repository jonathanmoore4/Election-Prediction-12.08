
import pandas as pd

def clean_2019_notional_fun(df):

    # Remove constituencies in Northern Ireland.
    df = df[
        ~df["Country name"].isin(["Northern Ireland"])
    ].copy()

    # Remove Chorley because the Commons Speaker row has no main party abbreviation.
    df = df[df["Constituency name"] != "Chorley"].copy()

    # Use Scotland and Wales as their own region labels.
    scotland_wales_mask = df["Country name"].isin(["Scotland", "Wales"])
    df.loc[scotland_wales_mask, "English region name"] = (
        df.loc[scotland_wales_mask, "Country name"]
    )

    # Combine SNP and Plaid Cymru into one nationalist Scotland/Wales label.
    df["Main party abbreviation"] = df["Main party abbreviation"].replace(
        {"SNP": "natSW", "PC": "natSW"}
    )

    # Drop rows without a main party abbreviation so the pivot has unique party keys.
    df = df[df["Main party abbreviation"].notna()].copy()

    df = df.pivot(
        index=["Constituency geographic code", "English region name", "Constituency name"],
        columns="Main party abbreviation",
        values="Candidate vote share",
    ).reset_index()

    df.columns.name = None

    # Add election year for all notional 2019 results.
    df["election"] = "2019_notional"

    # Add current set of boundaries for all constituencies.
    df["boundary_set"] = "2023-current"

    # Add descriptor showing whether election is actual or notional.
    df["election_type"] = "notional"

    # Add the previous election used for comparison.
    df["previous_election"] = "2017"

    df = df.rename(
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

    df["winner"] = df[share_columns].idxmax(axis=1).map(winner_labels)

    df["majority_proportion"] = df[share_columns].max(axis=1)

    # Keep only the columns needed for analysis.
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
            "natSW_share",
            "previous_election"
        ]
    ]

    return df
