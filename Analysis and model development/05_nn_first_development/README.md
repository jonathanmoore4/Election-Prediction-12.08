# First neural-network development

This package contains the report and reproducible evidence for seven evaluation elections, 1997–2019. The retained input includes earlier training elections from 1987. No later election is read by this package's runner or notebooks.

- [Report](REPORT_all_elections_progress.md) and [standalone printable HTML](REPORT_all_elections_progress.html).
- [00 — architecture study](00_architecture_study.ipynb): executed comparison and audit tables.
- [01 — NN02](01_NN02_64_32_validation_selected.ipynb): executable copy of the 64/32, eight-rate model configuration.
- [02 — NN03](02_NN03_16_fixed_0_3.ipynb): executable copy of the 16-unit, fixed-0.3 configuration.
- `data/train.csv`: hash-verified historical input snapshot.
- `results/`: filtered ensemble probabilities, per-seed diagnostics, derived tables, figures, configuration and numerical verification.
- `.nn_study_cache/`: retained per-trajectory caches, ignored by Git and regenerable. All 2,240 trajectories used by the report are retained here locally.

The pipeline implementations are [`NN01_model.py`](../../Models/NN01_model.py), [`NN02_model.py`](../../Models/NN02_model.py) and [`NN03_model.py`](../../Models/NN03_model.py). NN02 and NN03 use NN01's shared training code without changing NN01's default architecture or rate grid. All six pipeline candidates (three existing non-neural models and three neural variants) are evaluated by the existing selection rule. The notebook variant-definition cells copy their corresponding module definitions; set `TRAIN_MODEL=True` to execute a full example fit.

## Reproduction

From the repository root, use the repository Python environment:

```bash
.venv/bin/python "Analysis and model development/05_nn_first_development/run_study.py"
```

This checks data, training-code and numerical-library identities, resumes missing trajectories, regenerates both three- and ten-seed summaries, then builds and audits the report. `NN_STUDY_WORKERS` controls worker processes; `NN_STUDY_CACHE` optionally overrides the package cache root. Caches are an optimization, not a requirement for reproduction.

To rebuild only the analysis from saved probabilities and trajectory diagnostics:

```bash
.venv/bin/python "Analysis and model development/05_nn_first_development/summarize_results.py"
.venv/bin/python "Analysis and model development/05_nn_first_development/verify_results.py"
.venv/bin/python "Analysis and model development/05_nn_first_development/build_report.py"
```

`study.py` preserves the original fitting implementation byte for byte, allowing its SHA-256 to be verified against the retained experiment. Use `run_study.py` as the entry point; it overrides that implementation's original three-window defaults with this package's seven windows and retained input. `fast_batches.py` supplies the previously verified equivalent batch iterator.

## Provenance and limitations

Saved window-level results and predictions were retained from the completed study, restricted to evaluation years through 2019. Aggregates, sensitivity rankings, plots and report conclusions were recomputed from those retained windows. The source cache fingerprint identifies the historical run from which the identical retained trajectories originated; it is not a claim that this restricted report was independently selected or tested. The package configuration records the retained windows, input hash, exact training implementation and software versions.

The supporting files are consolidated here. Superseded neural-network notebooks, interim reports and unrelated experimental results have been removed from the parent analysis folder. Notebooks 00–04 are each in their own sibling directory for the pipeline, overview, baseline, XGBoost and logistic-regression work.
