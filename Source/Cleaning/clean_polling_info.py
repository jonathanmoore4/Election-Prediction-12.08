from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLLING_FILE = PROJECT_ROOT / "Data" / "Raw" / "PollBase-Q1-2026.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "Data" / "Processed" / "cleaned_polling.csv"

polling_df = pd.read_excel(POLLING_FILE, sheet_name="Monthly average")


polling_df = polling_df[["Date", "Conservative", "Labour", "LD"]]


polling_df = polling_df[polling_df["Date"].shift(-1).eq("GE")]

polling_df["Date"] = polling_df["Date"].astype(str).str[:4]

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

polling_df["incumbent"] = polling_df["Date"].map(incumbent_by_year)
polling_df[["Conservative", "Labour", "LD"]] = polling_df[["Conservative", "Labour", "LD"]] / 100




OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
polling_df.to_csv(OUTPUT_FILE, index=False)
