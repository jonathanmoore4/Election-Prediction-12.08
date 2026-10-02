import pandas as pd


def clean_polling_fun(df):
    # Keep the source schema at ingestion; expose consistent polling feature names.
    df = df[["Date", "Conservative", "Labour", "LD"]].rename(columns={
        "Conservative": "con_polling",
        "Labour": "lab_polling",
        "LD": "lib_polling",
    })

    df = df[df["Date"].shift(-1).eq("GE")].copy()

    df["Date"] = df["Date"].astype(str).str[:4]

    incumbent_by_year = {
        "1983": "con",
        "1987": "con",
        "1992": "con",
        "1997": "con",
        "2001": "lab",
        "2005": "lab",
        "2010": "lab",
        "2015": "con",
        "2017": "con",
        "2019": "con",
        "2024": "con",
    }

    df["incumbent"] = df["Date"].map(incumbent_by_year)
    df[["con_polling", "lab_polling", "lib_polling"]] = df[["con_polling", "lab_polling", "lib_polling"]] / 100

    return df

