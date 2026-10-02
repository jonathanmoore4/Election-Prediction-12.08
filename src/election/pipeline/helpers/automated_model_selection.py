"""Select a candidate by five outer elections, then retune on forecast history."""
import election.models.candidates as candidates_module
import election.models.evaluation as evaluation_module


def automated_model_selection(data, scores_path=None, *, candidates=None,
                              forecast_election=2024, outer_elections=evaluation_module.OUTER_ELECTIONS,
                              inner_elections=evaluation_module.INNER_ELECTIONS, verbose=True):
    """Return SelectionResult(model, report); also supports model, name unpacking.

    Registry order breaks exact ties. Input must exclude the forecast election.
    No data files are read here. A failed fit aborts comparison and saves a report
    marked failed, rather than selecting from a reduced set of elections/models.
    """
    evaluator = evaluation_module.HistoricalEvaluator(
        candidates_module.default_candidates() if candidates is None else candidates,
        outer_elections=outer_elections, inner_elections=inner_elections,
        scores_path=scores_path, verbose=verbose)
    return evaluator.evaluate(data, forecast_election=forecast_election)
