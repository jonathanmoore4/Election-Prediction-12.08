"""Select using persisted outer summaries, refit, and evaluate final outcomes."""
import election.models.candidates as candidates_module
import election.models.evaluation as evaluation_module


def automated_model_selection(data, *, test_data, output_dir=None, candidates=None,
                              forecast_election=2024, outer_elections=evaluation_module.OUTER_ELECTIONS,
                              inner_elections=evaluation_module.INNER_ELECTIONS, verbose=True,
                              tracking=None, predictor_guide=None):
    evaluator = evaluation_module.HistoricalEvaluator(
        candidates_module.default_candidates() if candidates is None else candidates,
        outer_elections=outer_elections, inner_elections=inner_elections, verbose=verbose)
    return evaluator.evaluate(data, forecast_election=forecast_election, test_data=test_data,
                              output_dir=output_dir, tracking=tracking, predictor_guide=predictor_guide)
