from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DATA_PREPARATION_ROOT = PROJECT_ROOT / "Current Data Preparation"
PIPELINE_HELPERS = Path(__file__).resolve().parent / "additional_funcs"

for import_path in (
    PROJECT_ROOT,
    CURRENT_DATA_PREPARATION_ROOT,
    PIPELINE_HELPERS,
):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from automated_model_selection import automated_model_selection
from clean_data_all import clean_all_data
from read_in_raw import read_raw_data
from predictor_guide import write_predictor_guide
from SQL.apply_SQL_queries import apply_sql_queries


def run_pipeline(output_dir: str | Path | None = None):
    """Run the workflow, saving train/test CSVs only in PROJECT_ROOT/TEST_TRAIN.

    output_dir controls model scores only (default: TEST_TRAIN).
    Relative score output paths are resolved from the caller's working directory.
    Returns [fitted_model, model_name].
    """
    train_test_dir = PROJECT_ROOT / "TEST_TRAIN"
    train_test_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir) if output_dir is not None else train_test_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the configured internet sources and checksum-verified local inputs.
    raw_data = read_raw_data()

    # Clean each raw dataframe using the Python cleaning files.
    cleaned_data = clean_all_data(raw_data)

    # Run the SQL feature-building files and write the train/test CSV files.
    train_test_data = apply_sql_queries(cleaned_data)

    train_test_data["train"].to_csv(train_test_dir / "train.csv", index=False)
    train_test_data["test"].to_csv(train_test_dir / "test.csv", index=False)
    write_predictor_guide(train_test_data, train_test_dir / "predictor_descriptions.md")

    trained_model = automated_model_selection(
        train_test_data["train"], scores_path=output_dir / "model_accuracies.csv",
    )

    return trained_model
