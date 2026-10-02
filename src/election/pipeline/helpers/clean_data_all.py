import election.preparation.clean_1992_notional as clean_1992_notional_module
import election.preparation.clean_2001_notional as clean_2001_notional_module
import election.preparation.clean_2005_notional as clean_2005_notional_module
import election.preparation.clean_2019_notional as clean_2019_notional_module
import election.preparation.clean_2024_results as clean_2024_results_module
import election.preparation.clean_historical_results as clean_historical_results_module
import election.preparation.clean_polling as clean_polling_module


def clean_all_data(raw_data):
    cleaned_historical = clean_historical_results_module.clean_historical_fun(raw_data["historical_results"])
    cleaned_1992_notional = clean_1992_notional_module.clean_1992_notional_fun(raw_data["results_1997_local"])
    cleaned_2001_notional = clean_2001_notional_module.clean_2001_notional_fun(
        cleaned_historical,
        raw_data["scottish_boundary_changes_2005"],
    )
    cleaned_2005_notional = clean_2005_notional_module.clean_2005_notional_fun(
        raw_data["notional_2005"],
        cleaned_historical,
    )
    cleaned_2019_notional = clean_2019_notional_module.clean_2019_notional_fun(raw_data["notional_2019"])
    cleaned_2024 = clean_2024_results_module.clean_2024_results_fun(raw_data["results_2024"])
    cleaned_polling = clean_polling_module.clean_polling_fun(raw_data["polling"])

    return {
        "historical": cleaned_historical,
        "notional_1992": cleaned_1992_notional,
        "notional_2001": cleaned_2001_notional,
        "notional_2005": cleaned_2005_notional,
        "notional_2019": cleaned_2019_notional,
        "results_2024": cleaned_2024,
        "polling": cleaned_polling,
    }
