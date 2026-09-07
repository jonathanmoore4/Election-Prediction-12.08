from io import BytesIO
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


REQUEST_HEADERS = {
    "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream,*/*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Connection": "keep-alive",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


RAW_SOURCES = {
    "historical_results": {
        "url": "https://researchbriefings.files.parliament.uk/documents/CBP-8647/1918-2019election_results.csv",
        "reader": "csv",
        "kwargs": {"dtype": {"election": str, "constituency_id": str}, "encoding": "cp1252"},
        "headers": {
            "Referer": "https://commonslibrary.parliament.uk/research-briefings/CBP-8647/",
        },
        # Checks the download is the right dataset.
        "expected_columns": ["constituency_name", "country/region", "election"],
    },
    "results_2024": {
        "url": "https://researchbriefings.files.parliament.uk/documents/CBP-10009/HoC-GE2024-results-by-constituency.csv",
        "reader": "csv",
        "kwargs": {},
        "headers": {
            "Referer": "https://commonslibrary.parliament.uk/research-briefings/CBP-10009/",
        },
        # Checks the download is the right dataset.
        "expected_columns": ["ONS ID", "Constituency name", "First party"],
    },
    "notional_2005": {
        "url": "https://electionresults.parliament.uk/general-elections/7/candidacies.csv",
        "reader": "csv",
        "kwargs": {},
        # Checks the download is the right dataset.
        "expected_columns": ["Constituency name", "Main party abbreviation", "Candidate vote share"],
    },
    "notional_2019": {
        "url": "https://electionresults.parliament.uk/general-elections/5/candidacies.csv",
        "reader": "csv",
        "kwargs": {},
        # Checks the download is the right dataset.
        "expected_columns": ["Constituency name", "Main party abbreviation", "Candidate vote share"],
    },
    "polling": {
        "url": "https://www.markpack.org.uk/files/2026/07/PollBase-Q2-2026.xlsx",
        "reader": "excel",
        "kwargs": {"sheet_name": "Monthly average"},
        "verify_ssl": False,
        # Checks the download is the right dataset.
        "expected_columns": ["Date", "Conservative", "Labour", "LD"],
    },
    "scottish_boundary_changes_2005": {
        "url": "https://www.electoralcalculus.co.uk/scottish_oldnewties.html",
        "reader": "html_table",
        "kwargs": {},
        "verify_ssl": False,
        # Checks the download is the right dataset.
        "expected_columns": ["Old Constituency", "New Constituency", "Member (as at 2001)"],
    },
}


def read_raw_data(sources=None):
    # Reads every source listed in RAW_SOURCES into a dictionary of dataframes.
    sources = sources or RAW_SOURCES

    raw_data = {}
    for name, source in sources.items():
        try:
            raw_data[name] = _read_one_source(name, source)
        except (HTTPError, URLError, ValueError) as exc:
            raise RuntimeError(f"Failed to read raw source '{name}'") from exc

    return raw_data


def _read_one_source(name, source):
    # Reads one dataset from its internet URL.
    url = source.get("url")
    if not url:
        raise ValueError(f"No internet URL has been set for {name}")

    return _read_and_validate_dataframe(url, source)


def _read_and_validate_dataframe(path_or_url, source):
    # Uses the correct pandas reader, then checks the expected columns exist.
    data = _download_if_url(
        path_or_url,
        headers=source.get("headers"),
        verify_ssl=source.get("verify_ssl", True),
    )

    if source["reader"] == "excel":
        df = pd.read_excel(data, **source["kwargs"])
    elif source["reader"] == "html_table":
        df = _read_matching_html_table(data, source)
    else:
        df = pd.read_csv(data, **source["kwargs"])

    df.columns = _clean_column_names(df.columns)
    missing_columns = set(source.get("expected_columns", [])) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns from {path_or_url}: {missing_columns}")

    return df


def _download_if_url(path_or_url, headers=None, verify_ssl=True):
    # Some source websites reject Python's default urllib user agent.
    if not str(path_or_url).startswith(("http://", "https://")):
        return path_or_url

    request_headers = REQUEST_HEADERS | (headers or {})
    request = Request(path_or_url, headers=request_headers)
    context = None if verify_ssl else ssl._create_unverified_context()
    with urlopen(request, context=context) as response:
        return BytesIO(response.read())


def _read_matching_html_table(path_or_url, source):
    # Finds the webpage table that contains the expected columns.
    tables = pd.read_html(path_or_url, **source["kwargs"])
    expected_columns = set(source.get("expected_columns", []))

    for table in tables:
        table.columns = _clean_column_names(table.columns)
        if expected_columns.issubset(set(table.columns)):
            return table

    if len(tables) == 1 and len(source.get("expected_columns", [])) <= len(tables[0].columns):
        table = tables[0]
        table.columns = list(source["expected_columns"]) + list(table.columns[len(source["expected_columns"]):])
        return table

    raise ValueError(f"No matching table found at {path_or_url}")


def _clean_column_names(columns):
    return [str(column).strip() for column in columns]
