import pandas as pd

def clean_historical_fun(df):

    # Keep elections from 1983 onwards, accepting string or numeric years.
    election_years = pd.to_numeric(df["election"], errors="coerce")
    df = df[election_years >= 1983].copy()
    df["election"] = election_years.loc[df.index].astype(int).astype(str)

    # Remove constituencies in Ireland and Northern Ireland
    df = df[~df["country/region"].isin(["Ireland", "Northern Ireland"])].copy()

    # National shares use vote totals across the retained constituencies only.
    parties = ["con", "lab", "lib", "natSW", "oth"]
    vote_columns = [f"{party}_votes" for party in parties]
    # Space-only entries in the source CSV can make entire vote columns text.
    # Preserve blanks as missing and reject unexpected nonnumeric values.
    count_columns = vote_columns + ["total_votes"]
    df[count_columns] = df[count_columns].replace(r"^\s*$", pd.NA, regex=True).apply(
        pd.to_numeric, errors="raise"
    )
    national_totals = df.groupby("election")[count_columns].transform("sum", min_count=1)
    denominator = national_totals["total_votes"].where(national_totals["total_votes"].ne(0))
    national_share_columns = [f"{party}_national_vote_share" for party in parties]
    for party, column in zip(parties, national_share_columns):
        df[column] = national_totals[f"{party}_votes"] / denominator

    # Standardise Yorkshire and the Humber region labels.
    df["country/region"] = df["country/region"].replace({
        "Yorkshire & The Humber": "Yorkshire and the Humber",
        "Yorkshire and The Humber": "Yorkshire and the Humber",
    })

    # Create winner and winning-party vote-share columns
    share_columns = ["con_share", "lib_share", "lab_share", "natSW_share", "oth_share"]
    winner_labels = {
        "con_share": "con",
        "lib_share": "lib",
        "lab_share": "lab",
        "natSW_share": "natSW",
        "oth_share": "oth"
    }

    # Calculate the largest party vote share for each row
    df["winning_party_vote_share"] = df[share_columns].max(axis=1)
    # Rank the same party categories; ties count as separate places.
    df["second_party_vote_share"] = df[share_columns].apply(
        lambda shares: shares.nlargest(2).iloc[-1] if shares.count() >= 2 else float("nan"),
        axis=1,
    )

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
            "winning_party_vote_share",
            "second_party_vote_share",
            "winner",
            "constituency_id",
            "boundary_set",
            "election_type",
            "con_share",
            "lib_share",
            "lab_share",
            "natSW_share",
            *national_share_columns,
        ]
    ]

    df["boundary_set"] = df["boundary_set"].replace("2010-2017", "2010-2019")

    previous_election_map = {
        "1983": pd.NA,
        "1987": "1983",
        "1992": "1987",
        "1997": "1992_notional",
        "2001": "1997",
        "2005": "2001_notional",
        "2010": "2005_notional",
        "2015": "2010",
        "2017": "2015",
        "2019": "2017",
    }

    df["previous_election"] = df["election"].map(previous_election_map)

    return df
