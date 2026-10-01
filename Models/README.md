# Model fitting and election evaluation

`candidates.default_candidates()` registers the eight candidates and the search
spaces from `MODEL_DEVELOPMENT_PLAN.md`. `HistoricalEvaluator` owns the chronological
splits, configuration ranking and model selection. Individual adapters fit one
supplied configuration and never choose their own validation elections.

```python
from Models.candidates import default_candidates
from Models.evaluation import HistoricalEvaluator

# Call explicitly when ready for the full, expensive comparison.
result = HistoricalEvaluator(
    default_candidates(), scores_path="results/model_accuracies.csv"
).evaluate(training_data, forecast_election=2024)
model = result.model
report = result.report
predictions = model.predict(forecast_data)
```

The pipeline returns the same `SelectionResult`. Existing `model, name =
run_pipeline(...)` unpacking still works. Comparison scores now live in
`result.report`, rather than in `model.initial_accuracies` or similar attributes.
The common class order is `con, lab, lib, natSW, oth`.

## Directory layout

All concrete model implementations live in `model_candidates/`. Shared code
stays directly under `Models/`.

```text
Models/
├── model_candidates/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── xgboost_expanded_model.py
│   ├── conditional_xgboost_model.py
│   ├── NN01_model.py
│   ├── NN02_model.py
│   └── NN03_model.py
├── candidates.py
├── evaluation.py
├── custom_model.py
├── pipeline_model.py
├── boosting.py
├── missing_data.py
└── training_policy.py
```

The top-level modules provide the registry, evaluation, model contracts,
preprocessing and training policies. Import concrete models from the subfolder:

```python
from Models.model_candidates.logistic_regression import LogisticRegressionModel
```

## Adding a model

1. Add a module under `model_candidates/` implementing a `custom_model` subclass with `train(data, configuration,
   fit_context) -> FitRecord` and `predict_proba(data)`. The inherited `predict`
   chooses the most probable class. Fit and retain preprocessing using training
   data only. Return a detached fit record through `_record()`.
2. Declare the model's predictors and add a `CandidateSpec` to `default_candidates()`.
   Supply its factory, unique name, search space and fixed settings. The factory
   must return a fresh object with the same name as the specification.
3. Use `TrainingPolicy` for ordinary full-history fits, `NeuralTrainingPolicy` for
   per-seed checkpoint durations, or implement the two policy hooks for a model
   with different fitting requirements. The evaluator needs no model-type checks.
4. Verify chronology, aligned probabilities, missing inputs and fresh-instance
   behavior using small test data before running the full comparison.

For an adapter that has already been implemented, registration looks like:

```python
CandidateSpec(
    name="New classifier",
    factory=NewClassifier,
    search_space={"regularisation": [0.1, 1.0, 10.0]},
    fixed_settings={"random_state": 42},
    feature_columns=tuple(FEATURE_COLUMNS),
)
```

Names and code in this example are placeholders. A caller can also pass a custom
candidate list to `automated_model_selection()` or `run_pipeline()` without
editing either function.

## Fitting and refitting

Each inner fit gets a fresh model, preprocessing and estimator state. Neural fits
receive one whole earlier election for checkpoint selection. They monitor log
loss, restore the lowest-loss checkpoint per seed, and return best epochs. The
evaluator ranks configurations by equal-weight mean inner-election accuracy.

The neural policy derives each seed's final duration from the selected
configuration's median best epoch, rounding halves up. A fresh model then fits
all available history for those durations, with no validation monitoring.
`model.retrain(history)` is a compatibility entry point that reruns tuning and
**returns a new model**; callers must assign its return value. Custom models can
supply `spec=` explicitly. New code should use `evaluator.tune_and_refit()`.

Conditional XGBoost accepts both `change.*` and `challenger.*` settings in one
configuration. Its stages are selected jointly by winner accuracy. It reuses
unchanged stage fits within one fold and rejects reuse with different training
data. A fold without any changed-seat training rows fails explicitly.

Missing training classes get zero probability from ordinary classifiers; a
single observed class uses a constant classifier. Neural outputs use the fixed
five-party convention. Logistic training includes `oth` to support the common
comparison; the old filtering helper remains available for historical notebook
code. Convergence failures and non-finite neural losses abort the comparison.

## Results and reproducibility

Selection uses the mean accuracy on 2005, 2010, 2015, 2017 and 2019. Inner validation
starts at 1997 and requires two earlier training elections. Missing scheduled
folds fail before fitting. Exact score ties use registry order; exact inner ties
use configuration order. Changed-seat accuracy is diagnostic, not an extra term
in the selection score.

Reports contain:

- A scorecard with all five scores, their mean and range, latest-three mean,
  diagnostic averages, runtime and completion status.
- Per-election metrics, changed/retained-seat counts, per-party seat errors,
  previous-winner baseline and neural seed metrics. Missing previous winners
  count as incorrect baseline predictions on the common evaluation rows.
- Constituency predictions and party probabilities, selected configurations,
  training years, seeds, checkpoint epochs and refit durations.
- A JSON audit with candidate versions, source digest, package versions, data
  digest, leave-one-election-out winners and final-fit records.

Macro-F1 always uses the five-party class convention, with zero contribution
for an absent class. Log loss clips probabilities at machine epsilon. Empty
changed/retained slices are unavailable (`None`), not zero; diagnostic means
include the count of available elections.

The report JSON records `incomplete`, `failed` or `complete`. On a fitting error,
partial results remain available with the error and active fitting context; no
reduced-fold winner is returned. Check this status when reading saved CSVs.

Inner scores and fit records are cached in memory for identical candidate,
configuration, policy, classes and training/validation data. Caches start empty
on each evaluation and are not persisted between runs. This avoids repeating
identical historical fits without reusing later data in earlier forecasts.

The full comparison is deliberately not run by the test suite. Tests use small
synthetic datasets and reduced fitting budgets. Run them with `python -m pytest
Tests` after installing the project requirements and pytest.
