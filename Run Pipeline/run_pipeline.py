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
from SQL.apply_SQL_queries import apply_sql_queries


def run_pipeline():
    # Read the raw data from the internet sources.
    raw_data = read_raw_data()

    # Clean each raw dataframe using the Python cleaning files.
    cleaned_data = clean_all_data(raw_data)

    # Run the SQL feature-building files and write the train/test CSV files.
    train_test_data = apply_sql_queries(cleaned_data)

    output_dir = PROJECT_ROOT / "TEST_TRAIN"
    output_dir.mkdir(exist_ok=True)

    train_test_data["train"].to_csv(output_dir / "train.csv", index=False)
    train_test_data["test"].to_csv(output_dir / "test.csv", index=False)

    trained_model = automated_model_selection(train_test_data["train"])

    return trained_model


