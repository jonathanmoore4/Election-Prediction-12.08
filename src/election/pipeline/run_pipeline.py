from pathlib import Path
from election import paths
from election.pipeline.helpers import automated_model_selection, clean_data_all, read_in_raw, predictor_guide
from election.sql import apply_sql_queries

PROJECT_ROOT = paths.project_root()


def run_pipeline(output_dir: str | Path | None = None, *, candidates=None, forecast_election=2024):
    """Run preparation, tracked selection/refitting, and final evaluation.

    Defaults to PROJECT_ROOT/notebooks/outputs. output_dir overrides the
    destination for datasets, the predictor guide and final prediction/matrix outputs.
    Relative paths are resolved from the caller's working directory.
    Returns SelectionResult with the fitted model and MLflow-backed evaluation report.
    Legacy model, model_name unpacking is supported. candidates overrides the registry.
    """
    output_dir = (Path(output_dir) if output_dir is not None
                  else PROJECT_ROOT / "notebooks" / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    from election.models.candidates import default_candidates
    from election.models.evaluation import OUTER_ELECTIONS, INNER_ELECTIONS
    from election.models.custom_model import PARTIES
    from election.models.tracking import ExecutionTracking, execution_metadata
    registry = tuple(default_candidates() if candidates is None else candidates)
    tracking = ExecutionTracking()
    metadata = execution_metadata(inner_elections=INNER_ELECTIONS, outer_elections=OUTER_ELECTIONS,
        classes=PARTIES, forecast_election=forecast_election,
        predictor_guide=output_dir / 'predictor_descriptions.md')
    with tracking.execution(registry, OUTER_ELECTIONS, metadata):
        # Read the configured internet sources and checksum-verified local inputs.
        raw_data = read_in_raw.read_raw_data()

        # Clean each raw dataframe using the Python cleaning files.
        cleaned_data = clean_data_all.clean_all_data(raw_data)

        # Run the SQL feature-building files and write the train/test CSV files.
        train_test_data = apply_sql_queries.apply_sql_queries(cleaned_data)

        train_test_data["train"].to_csv(output_dir / "train.csv", index=False)
        train_test_data["test"].to_csv(output_dir / "test.csv", index=False)
        predictor_guide.write_predictor_guide(train_test_data, output_dir / "predictor_descriptions.md")

        trained_model = automated_model_selection.automated_model_selection(
            train_test_data["train"], test_data=train_test_data["test"], output_dir=output_dir,
            predictor_guide=output_dir / "predictor_descriptions.md",
            candidates=registry, forecast_election=forecast_election, tracking=tracking,
        )

    return trained_model
