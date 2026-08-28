# Database

This folder is where the DuckDB database file should live.

Build or refresh the local database from the project root:

```powershell
C:\duckdb.exe "Database\election.duckdb" -c ".read SQL\reading_in_csv_files.sql"
C:\duckdb.exe "Database\election.duckdb" -c ".read SQL\playing.sql"
```

After running those commands, the database file is:

```text
Database/election.duckdb
```

Open that file in DuckDB/PyCharm instead of an unnamed in-memory DuckDB connection.
