# Manually supplied inputs

Keep these source files in Git alongside the code. They are inputs, not generated
outputs; do not remove them when rebuilding the pipeline.

## results97.xls

- Supplied by the project owner on 2026-09-11 as `results97 (2).xls`.
- Copied byte-for-byte to `src/election/data/results97.xls`, included in built distributions.
- Original publisher, acquisition date, source URL and licence: not yet recorded.
- SHA-256: e9e3605cc25492f2e3584db44cc9db669a595148f463445c8b4eecb80187ca7d
- Loader key: `results_1997_local` in `read_raw_data()`.
- Format: legacy Excel (.xls), read with xlrd. The default reads the first
  worksheet without assuming which row contains the column headings.

The loader verifies the bytes before reading. Missing or altered files stop the
run with an actionable error. For an intentional replacement, review and commit
the workbook, its SHA-256 in `read_in_raw.py`, and this provenance record together.

The pipeline uses this workbook to prepare the 1992 notional results through
`clean_1992_notional_fun`. The existing historical-results source supplies the
model's 1997 data.
