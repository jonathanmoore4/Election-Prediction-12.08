# Proposed work

### 1. Error-analysis notebook using unsupervised learning

Use historical predictions for elections excluded from model training. Explore both clustering incorrect predictions and clustering all observations before measuring error rates within each cluster. The latter distinguishes unusually error-prone groups from groups that merely contain many observations.

Start with scaled numeric predictors and carefully weighted categorical encodings. Exclude actual winners and correctness from clustering inputs; use them to describe clusters afterward. Investigate previous margins, previous winners, missing predictors and polling conditions. Check whether clusters mainly identify election years because national conditions are shared within an election.

Treat discovered patterns as hypotheses and check them in subsequent elections. Keep exploratory analysis of an evaluation election separate from claims of independent confirmation on that same election.

### 2. Probability averaging and stacking

First measure error overlap between models and compare a simple average of party probabilities with individual models. Stacking is most promising when models make complementary errors.

Add consistent `predict_proba` and class ordering to the shared model contract. A first stack could use a regularised logistic-regression meta-model over probabilities from logistic regression, XGBoost, a neural network and conditional XGBoost.

Generate meta-model training inputs by fitting base models on earlier elections and predicting later ones. Base-model tuning must also exclude each predicted election. Evaluate the complete stack on a further election that neither layer used.

Sklearn's `StackingClassifier` uses `cross_val_predict`, which requires test folds to partition the input rows. Expanding-window splits leave the initial training rows without held-out predictions, so replacing its `cv` with a standard temporal splitter is not sufficient. A custom temporal prediction and stacking loop is a suitable project approach.

### 3. Explore sklearn's automated-search building blocks

Scikit-learn provides tools for automated model selection rather than a built-in end-to-end AutoML system. Explore `Pipeline`, `ColumnTransformer`, `RandomizedSearchCV`, `GridSearchCV`, multiple scoring metrics and custom refit rules. Its related-projects page lists separate AutoML tools.

Start with randomised search and a fixed number of trials, using the election splits described above. Later, consider **successive halving**: try many configurations with a small computing budget, then give more computation to the promising ones. Where the model supports it, increase training iterations rather than randomly reducing the number of rows. Removing rows can change the mix of elections and leave too few examples of rare parties.

### 4. Bounded neural architecture search

The existing neural study is already a small manual architecture search. Formalise the search space, search strategy and performance-estimation procedure.

Explore modest networks with one to three hidden layers, layer widths, activation, weight decay, dropout, learning rate and batch size. This combines architecture and training-hyperparameter search. Screen configurations with fewer seeds, then confirm finalists with the existing ten-seed approach across historical elections. Keep outer evaluation elections outside architecture selection.

Prioritise reliable evaluation over a very large search: many constituency observations represent relatively few independent election environments.

### 5. Improve slice-based evaluation

Use a shared report covering election, country/region, previous winning party, previous margin bands, changed/retained seats, missing/complete predictors and prediction-confidence bands.

Report sample counts, errors, accuracy and, where meaningful, party recall and probability loss. Show election-specific results before pooling. Tiny slices and dependence within elections limit strong statistical conclusions.

Distinguish detecting that a seat changes from identifying its correct new winner. Current changed-seat accuracy measures correct winning-party prediction on actually changed seats; a model can detect a change but choose the wrong challenger. Also report false change predictions on retained seats.

Use a small predefined slice set for comparison. Verify newly discovered slices in subsequent elections rather than repeatedly optimising against the same discovered weaknesses.

### 6. Strengthen automated model selection

The shared expanding-election evaluator now implements nested historical tuning, common evaluation rows and selection by unweighted mean accuracy across five outer elections. Extend its diagnostics and saved predictions. Include simple baselines such as predicting the previous winner and logistic regression; define a polling-based baseline if useful.

Record overall accuracy, changed-seat accuracy, macro-F1, log loss and party seat-total errors. Decide in advance which metric or rule selects the winner, rather than choosing the objective after inspecting results. Save runtime, failures, configurations, data identity and seeds.

The current selection score is mean overall accuracy across 2005, 2010, 2015, 2017 and 2019. Changed-seat accuracy does not contribute to selection. Earlier proposals to add overall and changed-seat accuracy would count changed seats in both terms; any alternative objective needs an explicit decision.

If the intention is to give changed and retained seats equal importance, calculate accuracy for each group separately and average the two. This is an alternative to consider, not an agreed replacement.

Since 2024 results have already informed development, further comparisons on 2024 should be described as retrospective evaluation rather than an untouched final test.

### 7. Move model-specific preprocessing before each nested-CV iteration

Perform model-specific preprocessing once before the algorithm fits in each nested-CV iteration, rather than repeating it inside the algorithm for multiple fits. Make this an explicit step for each inner training/validation split, outer refit and final refit. Reuse prepared features across compatible hyperparameter candidates and neural seeds within the same iteration. Currently, preprocessing is performed inside model-fitting implementations.

Fit learned transformations only on that iteration’s training rows, then apply them to its held-out rows. Keep encoders, scalers, imputers and conditional role handling specific to the model and fold. Reuse transformed data across hyperparameter candidates only when preprocessing is independent of those parameters and the training/validation split is identical. Never fit preprocessing on the full dataset before splitting into folds. Preserve row membership, party ordering and the neural checkpoint/refit rules.

This is proposed work; the current pipeline still preprocesses inside model fits.

## Recommended implementation order

1. Shared historical evaluation and saved probabilities.
2. Slice reports and an error-analysis notebook.
3. Probability averaging and temporal stacking.
4. Broader automated hyperparameter search.
5. Bounded neural architecture search.

The first concrete deliverable is a reusable historical prediction dataset, which supports the other analyses. No implementation order or metric change has yet been approved as a coding task.

## References

- [Current model comparison](../src/election/models/compare_models.py)
- [Current XGBoost implementation](../src/election/models/adapters/xgboost_model.py)
- [Shared model contract](../src/election/models/custom_model.py)
- [Sklearn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [Sklearn cross-validation predictions](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
- [Sklearn hyperparameter search](https://scikit-learn.org/stable/modules/grid_search.html)
- [Sklearn related projects](https://scikit-learn.org/stable/related_projects.html)
- [Neural Architecture Search: A Survey](https://www.jmlr.org/papers/v20/18-598.html)
