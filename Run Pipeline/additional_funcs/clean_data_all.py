from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_DATA_PREPARATION_ROOT = PROJECT_ROOT / "Current Data Preparation"

if str(CURRENT_DATA_PREPARATION_ROOT) not in sys.path:
    sys.path.append(str(CURRENT_DATA_PREPARATION_ROOT))

from clean_2001_notional import clean_2001_notional_fun
from clean_2005_notional import clean_2005_notional_fun
from clean_2019_notional import clean_2019_notional_fun
from clean_2024_results import clean_2024_results_fun
from clean_historical_results import clean_historical_fun
from clean_polling import clean_polling_fun


def clean_all_data(raw_data):
    cleaned_historical = clean_historical_fun(raw_data["historical_results"])
    cleaned_2001_notional = clean_2001_notional_fun(
        cleaned_historical,
        raw_data["scottish_boundary_changes_2005"],
    )
    cleaned_2005_notional = clean_2005_notional_fun(
        raw_data["notional_2005"],
        cleaned_historical,
    )
    cleaned_2019_notional = clean_2019_notional_fun(raw_data["notional_2019"])
    cleaned_2024 = clean_2024_results_fun(raw_data["results_2024"])
    cleaned_polling = clean_polling_fun(raw_data["polling"])

    return {
        "historical": cleaned_historical,
        "notional_2001": cleaned_2001_notional,
        "notional_2005": cleaned_2005_notional,
        "notional_2019": cleaned_2019_notional,
        "results_2024": cleaned_2024,
        "polling": cleaned_polling,
    }
