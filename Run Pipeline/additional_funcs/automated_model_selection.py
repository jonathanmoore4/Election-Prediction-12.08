"""Select a candidate by five outer elections, then retune on forecast history."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Models.candidates import default_candidates
from Models.evaluation import HistoricalEvaluator, OUTER_ELECTIONS, INNER_ELECTIONS


def automated_model_selection(data, scores_path=None, *, candidates=None,
                              forecast_election=2024, outer_elections=OUTER_ELECTIONS,
                              inner_elections=INNER_ELECTIONS, verbose=True):
    """Return SelectionResult(model, report); also supports model, name unpacking.

    Registry order breaks exact ties. Input must exclude the forecast election.
    No data files are read here. A failed fit aborts comparison and saves a report
    marked failed, rather than selecting from a reduced set of elections/models.
    """
    evaluator = HistoricalEvaluator(
        default_candidates() if candidates is None else candidates,
        outer_elections=outer_elections, inner_elections=inner_elections,
        scores_path=scores_path, verbose=verbose)
    return evaluator.evaluate(data, forecast_election=forecast_election)
