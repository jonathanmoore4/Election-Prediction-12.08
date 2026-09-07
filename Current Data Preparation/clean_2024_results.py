import pandas as pd

def clean_2024_results_fun(df):

    # Rename columns to match the rest of the cleaned election data
    df = df.rename(
        columns={
            "First party": "winner",
            "Region name": "country/region",
            "Constituency name": "constituency_name",
            "ONS ID" : "constituency_id"
        }
    )

    # Remove constituencies in Ireland and Northern Ireland
    df = df[
        ~df["country/region"].isin(["Ireland", "Northern Ireland"])
    ].copy()

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
        df[share_column] = df[vote_column] / df["Valid votes"]

    df["natSW_share"] = df[["SNP_share", "PC_share"]].max(axis=1)

    df = df.drop(columns=["SNP_share", "PC_share"])

    # Define majority proportion as the largest party vote share.
    share_columns = [
        column for column in party_vote_columns.values() if column not in ["SNP_share", "PC_share"]
    ] + ["natSW_share"]
    df["majority_proportion"] = df[share_columns].max(axis=1)

    # Add election year for all 2024 results
    df["election"] = "2024"

    # Add current set of boundaries for all constituencies
    df["boundary_set"] = "2023-current"

    #Add descriptor showing whether election is actual or notional
    df["election_type"] = "actual"

    df["previous_election"] = "2019_notional"

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
    df["winner"] = df["winner"].replace(winner_mapping)

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
            "lab_share",
            "lib_share",
            "natSW_share",
            "previous_election"
        ]
    ]

    return df
