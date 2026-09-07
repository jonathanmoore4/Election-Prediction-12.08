import pandas as pd

def clean_2005_notional_fun(df, historical_df):

    # Remove constituencies in Ireland and Northern Ireland
    df = df[
        ~df["Country name"].isin(["Northern Ireland"])
    ].copy()

    # Use Scotland and Wales as their own region labels.
    scotland_wales_mask = df["Country name"].isin(["Scotland", "Wales"])
    df.loc[scotland_wales_mask, "English region name"] = (
        df.loc[scotland_wales_mask, "Country name"]
    )

    # Drop rows without a main party abbreviation so the pivot has unique party keys.
    df = df[df["Main party abbreviation"].notna()].copy()

    df = df.pivot(
        index=["Constituency geographic code", "English region name", "Constituency name"],
        columns="Main party abbreviation",
        values="Candidate vote share"
    ).reset_index()

    df.columns.name = None

    # Add election year for all 2024 results
    df["election"] = "2005_notional"

    # Add current set of boundaries for all constituencies
    df["boundary_set"] = "2010-2017"

    #Add descriptor showing whether election is actual or notional
    df["election_type"] = "notional"

    df = df.rename(
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
    df["winner"] = df[share_columns].idxmax(axis=1).map(winner_labels)
    df["majority_proportion"] = df[share_columns].max(axis=1)

    # Keep only the columns needed for analysis
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

    # scottish boundries didn't change between 2005 and 2010, so we can use the actual 2005 results for Scotland in the notional 2005 dataset.
    scottish_2005_results = historical_df[
        (historical_df["country/region"] == "Scotland")
        & (historical_df["election"] == "2005")
    ].copy()

    scottish_2005_results = scottish_2005_results[df.columns]
    scottish_2005_results["election"] = "2005_notional"

    df = pd.concat(
        [df, scottish_2005_results],
        ignore_index=True,
    )

    # Add the previous election used for comparison.
    df["previous_election"] = "2001"

    return df

