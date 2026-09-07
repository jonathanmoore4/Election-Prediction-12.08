from pathlib import Path

import duckdb


SQL_DIR = Path(__file__).resolve().parent
SQL_FILES = [
    SQL_DIR / "1_reading_in_csv_files.sql",
    SQL_DIR / "2_adding_previous_elections.sql",
    SQL_DIR / "3_add_polling.sql",
    SQL_DIR / "4_projected_polling.sql",
    SQL_DIR / "5_reading_into_testtrain.sql",
]


def apply_sql_queries(cleaned_data):
    # This creates a temporary DuckDB database in memory.
    # Nothing is saved to disk, and the database disappears when this function ends.
    con = duckdb.connect(database=":memory:")

    try:
        # Register each pandas dataframe as an input table.
        # 1_reading_in_csv_files.sql then turns these into the table names used
        # by the rest of the SQL pipeline.
        for name, dataframe in cleaned_data.items():
            con.register(f"input_{name}", dataframe)

        # Run each SQL file in order. Later files depend on views created by earlier files.
        for sql_file in SQL_FILES:
            query = sql_file.read_text()
            con.execute(query)

        # Pull the final train and test views back out as pandas dataframes.
        return {
            "train": con.execute("SELECT * FROM train_data").df(),
            "test": con.execute("SELECT * FROM test_data").df(),
        }
    finally:
        # Always close the connection, even if the SQL fails.
        con.close()
