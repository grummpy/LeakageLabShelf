# Copied rows

Train/test contamination from duplicate rows.

The table is 1,200 synthetic claims. Columns `x1`–`x3` carry a smooth signal; `x4`–`x6` are noise. `row_id` identifies the claim and is not a feature.

The leaky pipeline stacks a second copy of every row, with the same `row_id` and the same features, and then calls a stratified row split (`test_size=0.30`). About 70% of the leaky test rows have their twin in training. The clean pipeline splits the original table. Each `row_id` appears once, so the two sides share none.

## What is held fixed

Same forest on both paths: 200 trees, `min_samples_leaf=1`, `max_features="sqrt"`, `n_jobs=1`, `random_state` set to the shelf seed. Bootstrap means a given training row is missing from some trees. The copies are exact, so the trees that did see the twin still vote its label. On the copied test rows the forest's accuracy is 1.0000.

## Seed 42

Seed 42: leaky ROC-AUC 0.9617, clean ROC-AUC 0.6978.

502 of 720 leaky test rows share a `row_id` with training. Accuracy on those copied rows is 1.0000. Accuracy on the leaky test rows that are new is 0.6422, in the same neighborhood as the clean forest's accuracy of 0.6583. The headline 0.9617 is that mixture.

A logistic regression on the same two splits scores 0.7377 contaminated and 0.7350 clean. It cannot recall a twin, and the duplicated rows do not hand it new signal. The features support a linear ROC-AUC around 0.74. The forest on the clean split is a bit weaker than that, which is an ordinary random forest on a smooth boundary with noise columns. The forest on the contaminated split is the one that has seen the answer already.

## Where this shows up

An extract is concatenated with last month's extract, the key is not dropped, and `train_test_split` runs on the stack. The metric is then a blend of "rows the model memorized" and "rows it had to generalize to." Reporting only the blend hides which is which. This card reports both.
