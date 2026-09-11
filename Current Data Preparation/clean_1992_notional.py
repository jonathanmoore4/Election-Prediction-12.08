import pandas as pd


def clean_1992_notional_fun(df):
    """Clean the raw first worksheet of results97.xls, read with header=None.

    Return 1992 notional results without modifying the input dataframe.
    """
    df = df.copy()

    # Use the first row as column names, then remove it from the data.
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    df = df.rename(columns={"NAME": "constituency_name"})

    df = df.drop(columns=[
        "PA", "REGION", "GO region", "GO code", "euro const",
        "Electorate 97", "Con 97", "Lab 97", "LD 97", "PC 97", "SNP 97",
        "Ref 97", "Others 97", "Total 97",
    ])

    # Treat all missing values as zero before calculating results.
    df = df.fillna(0)
    party_labels = {
        "CON 92": "con",
        "LAB 92": "lab",
        "LD 92": "lib",
        "NAT 92": "natSW",
        "OTHERS 92": "oth",
    }
    votes = df[list(party_labels)].apply(pd.to_numeric)
    total_votes = pd.to_numeric(df["TOTAL 92"])
    denominator = total_votes.where(total_votes.ne(0))

    # Ties use the first party above, matching the notebook's behaviour.
    df["winner"] = votes.idxmax(axis=1).map(party_labels)
    df["majority_proportion"] = (votes.max(axis=1) / denominator).fillna(0)

    for party, column in {
        "con": "CON 92", "lib": "LD 92", "lab": "LAB 92", "natSW": "NAT 92",
    }.items():
        df[f"{party}_share"] = (votes[column] / denominator).fillna(0)

    df["election"] = "1992_notional"
    df["boundary_set"] = "1997-2001"
    df["previous_election"] = "1987"
    df["election_type"] = "notional"

    return df.drop(columns=[*party_labels, "TOTAL 92"])
