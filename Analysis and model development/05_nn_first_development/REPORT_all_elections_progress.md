# First neural-network development: elections through 2019


**Completed scope: seven evaluation elections—1997, 2001, 2005, 2010, 2015, 2017 and 2019.** Every table, figure, average and ranking in this package is restricted to that period. Earlier elections from 1987 supply training and validation history.


## Findings and candidate models


With validation-selected rates, **64/32 has the highest mean overall accuracy (83.80%)**. At fixed rate 0.3, **16 has the highest mean equally weighted overall/changed-seat score (59.35%)**. These rankings were recomputed from the seven retained elections.

The two candidate procedures are now implemented alongside NN01 in the pipeline. They remain retrospective development candidates: resizing a network does not consistently fix missed changes, false changes or biased party totals. The objective matters—overall accuracy, changed-seat recall and probability quality do not always favour the same procedure.


| Pipeline model | Hidden layers | Learning-rate policy | Notebook |
| --- | --- | --- | --- |
| NN01 | 32/16 | Validation-selected from 0.1, 0.2, 0.3, 0.5 | Existing reference implementation |
| NN02 | 64/32 | Validation-selected from 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 | [NN02 notebook](01_NN02_64_32_validation_selected.ipynb) |
| NN03 | 16 | Fixed 0.3 | [NN03 notebook](02_NN03_16_fixed_0_3.ipynb) |

All use ten seeded networks, SGD, cross-entropy, patience 20 and restoration of each seed’s best validation checkpoint. The latest supplied whole election is held out for validation. NN02 selects the rate with the lowest **ensemble** validation log loss; NN03 holds its rate fixed while retaining validation-based early stopping. The study’s full-grid 32/16 comparator is distinct from NN01’s smaller four-rate grid.


## Experiment and interpretation


Each window trains through a cutoff, validates on the next whole election and evaluates on the following one. Preprocessing is fitted on training rows only. Four architectures, eight rates and ten seeds give **2,240 trajectories**, supporting patience 10/20/50, **6,720 seed/patience records**, and **672 ten-seed ensembles**. The three-seed screening summaries reuse the first three seeds and are not independent replications.

These are already-inspected historical development results. Restricting the reporting period does not create an independent test. Elections overlap across training windows; ten seeds measure optimisation variation, not ten independent elections. The upstream features and boundary mappings have not received a fresh historical-availability audit.


## Architecture comparisons


Each election receives equal weight. The overall/changed score averages overall accuracy and accuracy on changed seats, matching the pipeline selector up to a factor of two. Higher accuracy/score is better; lower log loss and absolute party-seat error is better. The fixed-rate comparison holds the rate constant, whereas selected-rate comparisons include tuning effects.



### Validation-selected rates


| architecture | mean_accuracy | mean_changed_accuracy | mean_score | mean_log_loss | mean_absolute_seat_error |
| --- | --- | --- | --- | --- | --- |
| 16 | 83.40% | 29.53% | 56.47% | 0.519 | 176.000 |
| 32 | 82.81% | 30.45% | 56.63% | 0.535 | 186.857 |
| 32_16 | 83.74% | 32.17% | 57.96% | 0.528 | 172.571 |
| 64_32 | 83.80% | 30.42% | 57.11% | 0.533 | 169.429 |


### Fixed learning rate 0.3


| architecture | mean_accuracy | mean_changed_accuracy | mean_score | mean_log_loss | mean_absolute_seat_error |
| --- | --- | --- | --- | --- | --- |
| 16 | 83.65% | 35.05% | 59.35% | 0.532 | 175.429 |
| 32 | 83.19% | 34.82% | 59.01% | 0.538 | 181.714 |
| 32_16 | 83.27% | 33.02% | 58.15% | 0.541 | 181.429 |
| 64_32 | 83.04% | 33.00% | 58.02% | 0.544 | 183.429 |


## Every election

Overall accuracy with validation-selected rates:


| evaluation_year | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- |
| 1997 | 68.64% | 68.95% | 67.86% | 68.49% |
| 2001 | 94.07% | 94.23% | 94.38% | 94.54% |
| 2005 | 80.57% | 80.57% | 78.82% | 79.14% |
| 2010 | 85.60% | 84.81% | 87.34% | 89.24% |
| 2015 | 77.37% | 76.11% | 78.80% | 75.95% |
| 2017 | 89.24% | 87.82% | 88.77% | 89.56% |
| 2019 | 88.29% | 87.18% | 90.19% | 89.72% |


Correct new party among seats whose known previous winner changes:


| evaluation_year | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- |
| 1997 | 5.62% | 6.25% | 5.00% | 5.62% |
| 2001 | 12.50% | 12.50% | 16.67% | 4.17% |
| 2005 | 61.40% | 61.40% | 61.40% | 63.16% |
| 2010 | 83.04% | 86.61% | 84.82% | 83.04% |
| 2015 | 12.84% | 13.76% | 12.84% | 13.76% |
| 2017 | 8.96% | 8.96% | 8.96% | 8.96% |
| 2019 | 22.37% | 23.68% | 35.53% | 34.21% |


![Architecture comparisons](results/architecture_by_election.png)



## The two pipeline candidates


These rows reproduce NN02’s and NN03’s respective study procedures. They show why one candidate should not be declared universally best.


| evaluation_year | architecture | policy | learning_rate | accuracy | changed_accuracy | overall_changed_score | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1997 | 16 | fixed_0.3 | 0.300 | 68.49% | 5.00% | 36.74% | 1.338 |
| 1997 | 64_32 | validation_all_rates | 1.000 | 68.49% | 5.62% | 37.06% | 1.172 |
| 2001 | 16 | fixed_0.3 | 0.300 | 94.54% | 16.67% | 55.60% | 0.206 |
| 2001 | 64_32 | validation_all_rates | 1.000 | 94.54% | 4.17% | 49.35% | 0.198 |
| 2005 | 16 | fixed_0.3 | 0.300 | 80.25% | 63.16% | 71.71% | 0.444 |
| 2005 | 64_32 | validation_all_rates | 0.200 | 79.14% | 63.16% | 71.15% | 0.460 |
| 2010 | 16 | fixed_0.3 | 0.300 | 83.70% | 87.50% | 85.60% | 0.403 |
| 2010 | 64_32 | validation_all_rates | 1.000 | 89.24% | 83.04% | 86.14% | 0.305 |
| 2015 | 16 | fixed_0.3 | 0.300 | 77.53% | 14.68% | 46.11% | 0.828 |
| 2015 | 64_32 | validation_all_rates | 1.000 | 75.95% | 13.76% | 44.86% | 0.955 |
| 2017 | 16 | fixed_0.3 | 0.300 | 89.87% | 14.93% | 52.40% | 0.284 |
| 2017 | 64_32 | validation_all_rates | 1.000 | 89.56% | 8.96% | 49.26% | 0.404 |
| 2019 | 16 | fixed_0.3 | 0.300 | 91.14% | 43.42% | 67.28% | 0.223 |
| 2019 | 64_32 | validation_all_rates | 0.300 | 89.72% | 34.21% | 61.96% | 0.236 |


## Comparison with predicting the previous winner


The baseline and network are compared on exactly the same known-history rows. The 91 missing-history rows in 1997 remain in all-seat accuracy but are excluded from this comparison and from change/hold diagnostics. Correct changes minus false changes on held seats equals the net additional correct predictions against persistence.


| architecture | mean_known_accuracy | mean_baseline | mean_gain_pp | elections_beating_baseline |
| --- | --- | --- | --- | --- |
| 16 | 83.85% | 85.78% | -1.932 | 3 |
| 32 | 83.27% | 85.78% | -2.513 | 2 |
| 32_16 | 84.17% | 85.78% | -1.612 | 2 |
| 64_32 | 84.25% | 85.78% | -1.531 | 4 |


## Election-specific findings and party totals



### 1997


Training uses only 1987: national poll columns are constant, there are no positive other-party examples, and 91 evaluation rows lack previous winners. All architectures miss most changed seats and overpredict Conservatives.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 166 | 363 | 362 | 364 | 363 |
| lab | 418 | 268 | 266 | 271 | 269 |
| lib | 45 | 6 | 6 | 6 | 6 |
| natSW | 10 | 4 | 7 | 0 | 3 |
| oth | 2 | 0 | 0 | 0 | 0 |


### 2001


Only 24 seats change winner. All selected architectures score highly overall but underperform predicting the previous winner.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 166 | 187 | 186 | 186 | 178 |
| lab | 412 | 418 | 418 | 419 | 419 |
| lib | 52 | 26 | 27 | 28 | 34 |
| natSW | 9 | 10 | 10 | 8 | 10 |
| oth | 2 | 0 | 0 | 0 | 0 |


### 2005


Many changes are correctly identified, but false changes on held seats outweigh that benefit. Conservative totals are overpredicted and Labour totals underpredicted.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 198 | 301 | 301 | 312 | 313 |
| lab | 355 | 279 | 281 | 270 | 271 |
| lib | 62 | 38 | 36 | 36 | 35 |
| natSW | 9 | 10 | 10 | 10 | 9 |
| oth | 4 | 0 | 0 | 0 | 0 |


### 2010


The 64/32 architecture improves overall accuracy to 89.24%, versus 87.34% for 32/16. All four architectures nevertheless overpredict Conservative totals.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 306 | 375 | 391 | 369 | 353 |
| lab | 258 | 208 | 204 | 224 | 234 |
| lib | 57 | 43 | 32 | 32 | 38 |
| natSW | 9 | 6 | 5 | 7 | 7 |
| oth | 2 | 0 | 0 | 0 | 0 |


### 2015


The Liberal Democrat and combined nationalist totals show large shared errors across architectures. Architecture changes alone do not resolve this transfer failure.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 330 | 254 | 246 | 263 | 244 |
| lab | 231 | 321 | 332 | 312 | 334 |
| lib | 8 | 50 | 49 | 50 | 48 |
| natSW | 59 | 7 | 5 | 7 | 6 |
| oth | 4 | 0 | 0 | 0 | 0 |


### 2017


High overall accuracy masks weak changed-seat recall. The transition tables distinguish no-change predictions from changes to the wrong destination party.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 319 | 339 | 348 | 338 | 335 |
| lab | 261 | 231 | 222 | 229 | 234 |
| lib | 13 | 2 | 2 | 2 | 2 |
| natSW | 37 | 60 | 60 | 63 | 61 |
| oth | 2 | 0 | 0 | 0 | 0 |


### 2019


The full-grid 32/16 procedure reaches 90.19% accuracy, but predicts only one Liberal Democrat seat against eleven actual. Single-layer architectures improve at fixed rate 0.3 compared with their validation-selected rates.

| party | actual | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- | --- |
| con | 365 | 350 | 353 | 351 | 353 |
| lab | 202 | 245 | 250 | 235 | 233 |
| lib | 11 | 5 | 5 | 1 | 2 |
| natSW | 52 | 32 | 24 | 45 | 44 |
| oth | 2 | 0 | 0 | 0 | 0 |


Counts cover supplied Great Britain rows, not the entire UK Parliament. `natSW` combines SNP and Plaid Cymru, while `oth` pools other parties. Predicted totals use argmax winners and do not provide a parliamentary-majority probability.



## Sensitivity to individual elections


Each row counts first-place rankings after leaving out one of the seven evaluation elections. Models are not retrained and local rate selections remain fixed. Ties are counted for each tied architecture. This is descriptive sensitivity, not independent testing.


| policy | architecture | accuracy_first | score_first | log_loss_first |
| --- | --- | --- | --- | --- |
| validation_all_rates | 16 | 0 | 0 | 5 |
| fixed_0.3 | 16 | 7 | 6 | 6 |
| validation_all_rates | 32 | 0 | 0 | 0 |
| fixed_0.3 | 32 | 0 | 1 | 1 |
| validation_all_rates | 32_16 | 3 | 6 | 1 |
| fixed_0.3 | 32_16 | 0 | 0 | 0 |
| validation_all_rates | 64_32 | 4 | 1 | 1 |
| fixed_0.3 | 64_32 | 0 | 0 | 0 |


## Rates and patience

Selected rates at patience 20:


| evaluation_year | 16 | 32 | 32_16 | 64_32 |
| --- | --- | --- | --- | --- |
| 1997 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2001 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2005 | 0.200 | 0.500 | 0.200 | 0.200 |
| 2010 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2015 | 0.500 | 0.300 | 0.300 | 1.000 |
| 2017 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2019 | 0.050 | 0.050 | 0.300 | 0.300 |


For 32/16, the smaller grid uses {0.1, 0.2, 0.3, 0.5}. Positive accuracy deltas favour the smaller grid; positive log-loss deltas favour the full grid.


| evaluation_year | learning_rate_full | learning_rate_midrange | same_rate | midrange_accuracy_delta_pp | midrange_log_loss_delta |
| --- | --- | --- | --- | --- | --- |
| 1997 | 1.000 | 0.300 | False | 0.000 | 0.124 |
| 2001 | 1.000 | 0.500 | False | -0.468 | -0.004 |
| 2005 | 0.200 | 0.200 | True | 0.000 | 0.000 |
| 2010 | 1.000 | 0.500 | False | -1.266 | 0.006 |
| 2015 | 0.300 | 0.300 | True | 0.000 | 0.000 |
| 2017 | 1.000 | 0.500 | False | 0.949 | -0.035 |
| 2019 | 0.300 | 0.300 | True | 0.000 | 0.000 |


Patience comparisons with validation-selected rates:


| architecture | patience | mean_accuracy | mean_changed_accuracy | mean_log_loss |
| --- | --- | --- | --- | --- |
| 16 | 10 | 83.08% | 30.42% | 0.523 |
| 16 | 20 | 83.40% | 29.53% | 0.519 |
| 16 | 50 | 83.44% | 29.66% | 0.521 |
| 32 | 10 | 83.06% | 30.45% | 0.529 |
| 32 | 20 | 82.81% | 30.45% | 0.535 |
| 32 | 50 | 82.49% | 31.34% | 0.543 |
| 32_16 | 10 | 83.81% | 32.42% | 0.526 |
| 32_16 | 20 | 83.74% | 32.17% | 0.528 |
| 32_16 | 50 | 83.74% | 32.17% | 0.528 |
| 64_32 | 10 | 83.89% | 30.17% | 0.534 |
| 64_32 | 20 | 83.80% | 30.42% | 0.533 |
| 64_32 | 50 | 83.74% | 30.42% | 0.535 |


## Verification and reproduction


screening: 672 ensembles, 2016 patience records, policy selection, source-row alignment, accuracy, changed-seat accuracy, log loss and Brier score verified
confirmation: 672 ensembles, 6720 patience records, policy selection, source-row alignment, accuracy, changed-seat accuracy, log loss and Brier score verified
Election diagnostics and party totals verified, including missing-history denominators and persistence-gain identity.



[Executed study notebook](00_architecture_study.ipynb) · [Full numerical results](results/RESULTS.md) · [Reproduction instructions](README.md)

The retained data, probabilities, diagnostics and scripts are in this directory. The new pipeline classes share NN01’s training implementation, with independent architecture/rate settings; the model notebooks use those same classes to avoid diverging implementations.

