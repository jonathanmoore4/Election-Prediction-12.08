# Build a 2001 "notional" result file on 2005 constituency boundaries.
#
# England did not change boundaries between 2001 and 2005, so English rows can
# keep their actual 2001 constituency names. Scotland did change boundaries, so
# Scottish rows need to be renamed or removed using the 2005 boundary-change file.

import re

import pandas as pd


def normalise_constituency_name(constituency_name):
    # Standardise names before matching: trim whitespace, treat "&" as "and",
    # remove punctuation/parenthetical notes, collapse repeated spaces, and
    # ignore capitalisation.
    constituency_name = re.sub(r"\s*\([^)]*\)", "", str(constituency_name))
    constituency_name = constituency_name.strip().replace("&", "and").replace(",", "")
    return " ".join(constituency_name.split()).casefold()

def clean_2001_notional_fun(df, scottish_boundary_changes_df):
    # Drop the MP name column because this script only needs constituency mappings.
    scottish_boundary_changes_df = scottish_boundary_changes_df.drop(
        columns=["Member (as at 2001)"]
    )

    # Build a dictionary from 2005 Scottish constituency name to its 2005
    # constituency ID, so renamed Scottish rows also move onto 2005 IDs.
    scottish_constituency_ids_2005 = {
        normalise_constituency_name(constituency_name): constituency_id
        for constituency_name, constituency_id in zip(
            df.loc[
                (df["country/region"] == "Scotland")
                & (df["election"] == "2005"),
                "constituency_name",
            ],
            df.loc[
                (df["country/region"] == "Scotland")
                & (df["election"] == "2005"),
                "constituency_id",
            ],
        )
    }

    # Keep only the 2001 general election rows.
    df = df[df["election"] == "2001"].copy()

    # Build a dictionary for quick lookup from old Scottish constituency name to
    # its 2005-boundary constituency name.
    scottish_boundary_mapping_2005 = {
        normalise_constituency_name(old_constituency): str(new_constituency).strip()
        for old_constituency, new_constituency in zip(
            scottish_boundary_changes_df["Old Constituency"],
            scottish_boundary_changes_df["New Constituency"],
        )
    }

    # These names refer to the same 2001 constituencies, but the historical results
    # and boundary-change file use different word order.
    scottish_constituency_name_aliases = {
        "Central Fife": "Fife Central",
        "North East Fife": "Fife North East",
        "North Tayside": "Tayside North",
        "West Aberdeenshire & Kincardine": "Aberdeenshire West and Kincardine",
        "West Renfrewshire": "Renfrewshire West",
    }

    # Add extra lookup keys for constituency names where the election-results file
    # and the boundary-change file use different names for the same old seat.
    for election_results_name, boundary_changes_name in scottish_constituency_name_aliases.items():
        scottish_boundary_mapping_2005[
            normalise_constituency_name(election_results_name)
        ] = scottish_boundary_mapping_2005[normalise_constituency_name(boundary_changes_name)]

    # Identify Scottish rows in the 2001 results, then translate each old Scottish
    # constituency name to the matching 2005-boundary constituency name.
    scottish_2001_mask = df["country/region"] == "Scotland"
    mapped_scottish_constituencies = df.loc[
        scottish_2001_mask, "constituency_name"
    ].map(
        lambda constituency_name: scottish_boundary_mapping_2005.get(
            normalise_constituency_name(constituency_name)
        )
    )

    # Find any Scottish constituencies that failed to map. These would indicate a
    # missing alias, typo, or unmatched source row.
    unmapped_scottish_constituencies = df.loc[
        scottish_2001_mask & mapped_scottish_constituencies.isna(), "constituency_name"
    ]

    # Stop immediately if any Scottish row cannot be mapped, rather than silently
    # producing incomplete notional results.
    if not unmapped_scottish_constituencies.empty:
        raise ValueError(
            "Missing Scottish boundary mappings for: "
            + ", ".join(sorted(unmapped_scottish_constituencies.unique()))
        )

    # Some 2001 Scottish constituencies have no direct 2005 equivalent. Remove those
    # rows because they cannot be represented as a 2005-boundary constituency result.
    disappearing_scottish_seat_mask = (
        scottish_2001_mask & mapped_scottish_constituencies.eq("Disappearing seat")
    )

    # Apply the removal to both the results dataframe and the mapped-name series so
    # their indexes still align for the assignment below.
    df = df[~disappearing_scottish_seat_mask].copy()
    mapped_scottish_constituencies = mapped_scottish_constituencies[
        ~disappearing_scottish_seat_mask
    ]
    mapped_scottish_constituency_ids = mapped_scottish_constituencies.map(
        lambda constituency_name: scottish_constituency_ids_2005.get(
            normalise_constituency_name(constituency_name)
        )
    )

    missing_scottish_constituency_ids = mapped_scottish_constituencies[
        mapped_scottish_constituency_ids.isna()
    ]

    if not missing_scottish_constituency_ids.empty:
        raise ValueError(
            "Missing 2005 Scottish constituency IDs for: "
            + ", ".join(sorted(missing_scottish_constituency_ids.unique()))
        )

    # Replace Scottish constituency names and IDs with their 2005-boundary equivalents.
    # Non-Scottish rows keep their existing 2001 names.
    df.loc[
        df["country/region"] == "Scotland", "constituency_name"
    ] = mapped_scottish_constituencies
    df.loc[
        df["country/region"] == "Scotland", "constituency_id"
    ] = mapped_scottish_constituency_ids

    # Mark every remaining row as a 2001 result expressed on 2005 boundaries.
    df["boundary_set"] = "2005"
    df["election_type"] = "notional"
    df["election"] = "2001_notional"

    return df
