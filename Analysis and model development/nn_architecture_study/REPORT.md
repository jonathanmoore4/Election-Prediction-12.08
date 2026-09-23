# Neural network architecture, learning-rate and early-stopping study

## Recommendation

**Keep the existing 32/16 ReLU architecture and cross-entropy, with ten fixed seeds and whole-election early stopping. Keep patience at 20 and restore each seed's lowest validation-loss checkpoint. For an automatic learning-rate search, a defensible smaller candidate set is 0.1, 0.2, 0.3 and 0.5, selected by ensemble validation log loss.** This smaller set reproduces the full eight-rate selector's results for the 32/16 network at all three tested patience values in all three windows.

This is a conservative, reproducible candidate configuration, **not a finding that it is the universally best forecaster**. Fixed 0.3 is a useful simpler alternative: its next-election accuracy was slightly higher than the full search's. Fixed 0.5 is also a credible development candidate, with small accuracy gains on all three evaluation elections relative to fixed 0.3 at patience 20, but no clear average log-loss benefit. These results do not compel automatic search over a fixed rate. If retaining search, do so to define an adaptable validation-based procedure, not because it won this backtest.

Most importantly, **the neural network has not established consistent value over predicting the previous winner**. Architecture and learning-rate tuning have not resolved the weak 2015 and 2005 forecasts. It is reasonable to retain it as a model candidate; these results do not justify forcing it to be the pipeline's selected forecaster. No pipeline files were changed.

## What was tested

- Train through 2015, validate on 2017, evaluate on 2019.
- Train through 2005, validate on 2010, evaluate on 2015.
- Train through 1997 (including 1987 and 1992), validate on 2001, evaluate on 2005.
- Hidden layers: 16, 32, 32/16 and 64/32. ReLU, five output classes, cross-entropy.
- Learning rates: 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5 and 1.0.
- Patience: 10, 20 and 50. Minimum improvement zero; maximum 1,000 epochs.
- SGD without momentum or weight decay; batch size 64. No projection, feature, loss or optimiser changes.
- Three-seed screen of every architecture/rate/window, then ten-seed confirmation of the existing 32/16 and strongest validation-selected challenger (16).

There were **624 unique training trajectories**: 288 in screening and 336 additional confirmation runs. Each trajectory supports three exact patience comparisons, equivalent to 1,872 distinct seed/window/architecture/rate/patience fits. Screening and confirmation produce 288 and 144 ensemble configurations respectively; the confirmation reuses the first three seeds. Whole elections stay together; preprocessing uses weight-update rows only. All labelled evaluation rows are retained, with missing features imputed from training data. No 2024 data was read.

The challenger was chosen from screening by lowest mean locally selected ensemble validation log loss at patience 20, not by evaluation-election accuracy. The 32 and 64/32 networks were screened only; they were not confirmed with ten seeds. This is a bounded architecture comparison, not an exhaustive architecture search.

## 1. Architecture: lower validation loss did not guarantee a better next-election forecast

Three-seed screening, with local learning-rate selection and patience 20:

| architecture | mean_validation_log_loss | mean_accuracy |
| --- | --- | --- |
| 32_16 | 0.2525 | 82.93% |
| 64_32 | 0.2525 | 82.66% |
| 16 | 0.2427 | 81.23% |
| 32 | 0.2519 | 80.44% |

Ten-seed confirmation under the same procedure:

| architecture | mean_validation_log_loss | mean_accuracy | mean_changed_accuracy | mean_log_loss |
| --- | --- | --- | --- | --- |
| 32_16 | 0.2523 | 82.60% | 36.59% | 0.5014 |
| 16 | 0.2409 | 82.08% | 32.21% | 0.5172 |

The single 16-unit layer consistently has the lower average validation log loss in these comparisons, but the 32/16 network has the better average next-election accuracy and log loss under the full-search policy. At fixed rate 0.3 and patience 20, their mean accuracies are almost identical (16 units: 82.98%; 32/16: 82.92%). There is no decisive architecture winner across procedures and metrics. The larger 64/32 network showed no clear screen-level advantage, so there is no evidence here to pay for a larger architecture. Retaining the existing 32/16 is a defensible choice; claiming it is proven optimal would not be.

## 2. Learning rates: 0.3 is a strong validation choice, not the retrospective accuracy maximum

Ten-seed 32/16 results at patience 20, averaging each election equally:

| learning_rate | mean_validation_log_loss | mean_next_election_accuracy | mean_next_election_log_loss |
| --- | --- | --- | --- |
| 0.0100 | 0.2695 | 80.97% | 0.4890 |
| 0.0300 | 0.2646 | 80.86% | 0.4975 |
| 0.0500 | 0.2624 | 81.29% | 0.4930 |
| 0.1000 | 0.2574 | 81.23% | 0.4985 |
| 0.2000 | 0.2556 | 82.44% | 0.4976 |
| 0.3000 | 0.2524 | 82.92% | 0.4950 |
| 0.5000 | 0.2604 | 83.40% | 0.4969 |
| 1.0000 | 0.2673 | 83.77% | 0.5316 |

The full validation selector chooses 0.3 using 2017, 0.3 using 2010, and 0.2 using 2001. The smaller set {0.1, 0.2, 0.3, 0.5} retains these choices. The compact set {0.1, 0.3} chooses 0.3 in all three windows at patience 20, so it behaves identically to fixed 0.3 here; there is no demonstrated benefit from that compact search over its fixed comparator.

Policy comparison for 32/16, patience 20:

| policy | mean_accuracy | worst_accuracy | mean_changed_accuracy | mean_log_loss | mean_brier |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | 83.77% | 78.80% | 43.34% | 0.5316 | 0.2380 |
| fixed_0.5 | 83.40% | 78.96% | 37.91% | 0.4969 | 0.2418 |
| validation_compact | 82.92% | 78.80% | 36.59% | 0.4950 | 0.2474 |
| fixed_0.3 | 82.92% | 78.80% | 36.59% | 0.4950 | 0.2474 |
| validation_all_rates | 82.60% | 78.80% | 36.59% | 0.5014 | 0.2517 |
| validation_midrange | 82.60% | 78.80% | 36.59% | 0.5014 | 0.2517 |

Fixed 0.5 improves accuracy relative to fixed 0.3 on 2019 (90.66% vs 90.19%), 2015 (78.96% vs 78.80%) and 2005 (80.57% vs 79.78%). Its mean log loss is very slightly worse (0.4969 vs 0.4950). That makes it a reasonable alternative for a winner-accuracy objective, but these differences are small and were inspected retrospectively.

Fixed 1.0 has the highest retrospective mean accuracy among the confirmed policies (83.77% with 32/16 and patience 20), but substantially worse average log loss than fixed 0.3. On 2015 its log loss worsens from 0.7897 to 0.9421 without improving accuracy. It is not selected by validation in any of the 32/16 windows. Promoting it solely because it heads the development accuracy table would ignore the probability-quality trade-off and selection uncertainty.

The 16-unit network with fixed 0.5 and patience 10 is another competitive development result: 83.72% average accuracy and 0.4777 log loss, with worst-election accuracy of 78.32%. This is evidence against claiming a unique winner, not an untouched confirmation that this retrospectively identified combination will generalise best.

## 3. Early stopping: no case for increasing patience to 50

Ten-seed 32/16 results with full validation-based learning-rate selection:

| patience | mean_accuracy | mean_changed_accuracy | mean_log_loss |
| --- | --- | --- | --- |
| 10 | 82.55% | 37.16% | 0.5098 |
| 20 | 82.60% | 36.59% | 0.5014 |
| 50 | 82.60% | 36.59% | 0.5014 |

Patience 20 and 50 give identical selected-ensemble metrics in all three windows, for both confirmed architectures. Across all 480 confirmed trajectories, increasing patience from 10 to 20 changes 95 best checkpoints; increasing 20 to 50 changes only 36. Those changes do not translate into an advantage for the full-search policy at patience 50. No run reaches the 1,000-epoch cap.

Patience 10 has some favourable fixed-rate results, so 20 is not a proven unique optimum. However, the full-search 32/16 comparison offers no compelling reason to replace the existing 20. Selecting both patience and rate on the validation election yields 82.55% mean next-election accuracy for 32/16, compared with 82.60% for fixed patience 20: the extra selection step adds no demonstrated benefit.

Patience measures how long to continue without improvement; it is not a fixed training duration. Every model restores its own best checkpoint. The shared-trajectory implementation was checked against three separate reference fits at patience 10/20/50: probabilities and stopping epochs matched exactly.

## 4. Baselines and errors: the bigger issue is historical transfer

Full learning-rate search, 32/16, patience 20:

| evaluation_year | learning_rate | evaluation_accuracy | evaluation_changed_seat_accuracy | previous_winner_accuracy |
| --- | --- | --- | --- | --- |
| 2019 | 0.3000 | 90.19% | 35.53% | 87.97% |
| 2015 | 0.3000 | 78.80% | 12.84% | 82.75% |
| 2005 | 0.2000 | 78.82% | 61.40% | 90.92% |

The previous-winner rule averages 87.22% across these elections, versus 82.60% for this neural procedure. Its changed-seat accuracy is zero by definition, so it does not solve forecasting changes; it nevertheless exposes the cost of incorrect changes introduced by the network. None of the confirmed policies beats its three-election average accuracy.

The per-party analysis makes the weaknesses concrete. For the full-search 32/16 model at patience 20:

- In 2015 it predicts **50 Liberal Democrat seats versus 8 actual**, and **7 nationalist seats versus 59 actual**. Both confirmed architectures have these same two predicted totals under their selected settings. The `natSW` class combines SNP and Plaid Cymru; these totals must not be described as SNP alone.
- In 2005 it predicts **312 Conservative seats versus 198 actual**, and **270 Labour seats versus 355 actual**.
- In 2019 it predicts only **1 Liberal Democrat seat versus 11 actual**, despite 90.19% overall accuracy.
- Neither confirmed full-search model at patience 20 predicts the `oth` class as a winner in these evaluation elections.

These observations suggest that further gains require attention beyond hidden-layer size and learning rate. They do not by themselves identify the cause: this study did not test revised projection features, regional polling, different targets or losses. Those changes should be evaluated as separate hypotheses.

## 5. What this does and does not establish

These are development results, not independent tests. The elections have already been inspected; earlier evaluation elections appear in later training windows; and the architecture shortlist uses validation results across periods. Each window's checkpoint and rate selection excludes its evaluation outcomes, but choosing an overall recipe after reading this report is still retrospective. Ten seeds measure optimisation variability, not ten independent future elections. With only three correlated periods, small accuracy differences do not establish statistical superiority. See [scikit-learn's explanation of model-selection bias](https://scikit-learn.org/1.6/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

A reasonable next step is to freeze the small cross-entropy candidate procedure described above, rather than continue searching until a favourable headline appears. Keep its baseline comparison visible when it is eventually integrated. The 2019 pipeline comparison is now development evidence, not an untouched election. Feature improvements can be evaluated later under the same declared procedure.

## Reproduction and files

Open `../07_nn_architecture_learning_rate_study.IPYNB` for the executed tables, plots and per-party results. By default it reads saved results; set `RUN_STUDY = True` to rerun/resume the full study. Alternatively run `python "Analysis and model development/nn_architecture_study/study.py"` from the repository root.

- `configuration.json`: data/code hashes, versions, seeds, splits and selection protocol.
- `screening_*` and `confirmation_*`: every configuration and policy, per-seed diagnostics and compressed ensemble probabilities.
- `per_party_recall.csv`: actual/predicted seat counts, precision and recall for the full-search patience-20 policies.
- `baselines.csv`: previous-winner and training-majority comparators on the same evaluation rows.
- `verify_results.py` and `verification.txt`: audit code and results.

The audit verifies ensemble/run counts, temporal ordering, checkpoint ordering, local policy selection, valid probability distributions and accuracy reconstructed from saved predictions. All 24 original ten-seed 32/16 patience-20 validation results reproduce notebook 06. Individual-run caches under `/tmp` support resuming on this machine; they can be regenerated from the code and data. No fitted pipeline or pipeline-selection code was changed.
