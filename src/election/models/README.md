# Model evaluation

Model modules expose an accuracy-returning function, a candidate dictionary and
`TRAINING_METADATA`. There are no candidate, evaluation-report or training-policy
classes. Fitted sklearn pipelines, neural networks and two-stage classifiers
remain implementation details and are never recorded in MLflow.

```python
import pandas as pd
from election.models.adapters.logistic_regression import (
    LogReg, logreghyperparameters, TRAINING_METADATA,
)
from election.models.evaluation import nested_cv
from election.pipeline.preparation import load_prepared

history, locations = load_prepared("outputs/latest_prepared.json")
result = nested_cv(
    LogReg, logreghyperparameters, history,
    model_id="logistic_regression",
    metadata={**TRAINING_METADATA, "data_id": locations["data_id"],
              "prepared_data": locations},
    n_trials=30, random_seed=42,
    tracking={},  # Existing MLflow server/environment. None means do not log.
)
print(result["mean_outer_accuracy"], result["run_id"])
```

`nested_cv` imports no concrete model implementations. It validates the schedule,
iterates the outer elections, tunes using earlier elections, refits and scores.
Each outer election starts a fresh Optuna study using `RandomSampler`. The shared
objective samples the model's candidate dictionary with `suggest_parameters`,
evaluates every inner election and returns their unweighted mean accuracy.
The default budget is 30 trials with seed 42; `n_trials` and `random_seed` can be
set in the Python API or with CLI `--n-trials` and `--random-seed` for `evaluate`
and `all`. Sampling is with replacement and does not guarantee every combination
is tried. Repeated configurations reuse cached inner scores/checkpoint records;
a fixed search space runs once. Exact inner score ties select the first sampled
configuration. `parameter_combinations` remains a utility for inspecting candidate
spaces, but is not used by tuning. Model ties follow the
requested model order (the pipeline's default order preserves the old order).
All scheduled elections must succeed. Accuracy excludes unknown winner labels;
missing predictors are handled by training-fitted preprocessing.

The two schedule dictionaries in `config.py` reproduce the existing elections.
Final evaluation reuses `nested_cv` with outer election 2024, inner elections
1997–2019 and a training cutoff of 2019. It records a separate run and reuses the selected historical run's candidate
space, trial budget and random seed. The evaluation protocol version distinguishes
these random-search evaluations from earlier exhaustive-search records.

## Neural early stopping is unchanged

The inner election being scored also selects each seed's lowest-loss checkpoint.
Patience remains 20; the maximum is 1,000 epochs; best weights are restored.
Learning rates are compared by mean inner-election accuracy. For outer/final
refits, fresh preprocessing and networks use all eligible history. Each seed
trains for its median best inner epoch, rounded halves up, with a minimum of one.
There is no outer/final validation monitoring or extra holdout. For example,
2017 outcomes never inform the model fitted to predict 2017.

Neural functions called with only three arguments perform an inner-style fit:
`test_data` selects checkpoints. The nested evaluator passes `fit_records` for
outer/final refits. This explicit optional argument preserves the existing
procedure; a bare neural call is not a substitute for the final-evaluation stage.

All built-in functions optionally accept `return_details=True`, `cache`,
`fit_records` and `output_dir`. Details and fit records are ordinary transient
dictionaries. Only the selected inner score summary and outer refit durations
reach MLflow. Cached scores/checkpoint records are local to one nested evaluation;
conditional stage fits are cached only within the same training partition.
No fitted model appears in a returned evaluation dictionary.

## Adding a model

Expose a three-argument accuracy function and a candidate dictionary. Supply
model identity and fixed training/feature metadata when calling `nested_cv`.
An ordinary function returning a float needs no inheritance or optional keywords.
For inclusion in the CLI, add its module/function/candidate names to the small
`MODEL_MODULES` mapping in `pipeline/run_pipeline.py`.

Built-ins validate supported hyperparameter names explicitly. Model-specific
fitting, features and fixed settings remain in their existing adapter modules.
Shared fitting/scoring helpers strip test labels before prediction, enforce the
five-party probability order and optionally export final constituency reports
locally. `custom_model.py` and `pipeline_model.py` retain only fitted-model
implementation helpers; fit contexts and fit records are dictionaries.

See the [recording, comparison and stage commands](../../../README.md).
