# Target encoding

Preprocessing fit on the full table before the split.

Each of the 2,000 synthetic intake rows has its own `member_id`. The clinical columns are age, systolic blood pressure, and one lab. The lab carries most of the honest signal.

## The leaky call

```python
encoder = TargetEncoder(smooth="auto", target_type="binary", random_state=seed)
encoder.fit(frame[["member_id"]], frame["high_risk"])
encoded = encoder.transform(frame[["member_id"]])
# the split happens after this
```

`fit` stores a target mean for every category it saw. `transform` reads that table back. A category with one row has a mean equal to that row's label, including under `smooth="auto"` on this data: the stored value is 0 or 1, and it matches `high_risk` on every row. The logistic regression that trains after the split is separating on a column that already is the label. Seed 42: leaky ROC-AUC 1.0000, clean ROC-AUC 0.7531. Leaky accuracy is 1.0000.

That perfect score is the failure. There is nothing left for the clinical columns to explain.

## The clean call

Split first. The encoder lives inside a `Pipeline`, beside a `StandardScaler` on the clinical columns, and `fit` runs on the training rows only. Every test `member_id` is new. `TargetEncoder` fills an unseen category with the training prevalence, so the test encoding is constant (standard deviation 0.0000 on seed 42). The clean ROC-AUC matches a model that sees only age, systolic, and the lab, within 0.001.

## `fit_transform` is a different call

scikit-learn's `TargetEncoder.fit_transform` cross-fits, so a row does not receive an encoding built from its own label. On these unique ids, `fit_transform` of the full table correlates with the label at about 0 (seed 42: -0.0025). It does not reproduce the leaky score.

Cross-fitting the full table is still the wrong order when the same category appears on both sides of a later split: held-out folds can contain rows you meant to score. The leaky notebook in this card does not use that call. It fits on every row it then transforms. The shelf keeps the two calls separate so the perfect score is traceable to `fit` then `transform`, which is the one that pastes the label back onto a one-row id.

## Where this shows up

A high-cardinality key — member, claim, device — gets target-encoded in a cell above the split, because the encoder was easier to fit once on the whole frame. If the key is unique, the encoded column is the label. If the key is only rare, the same call leaks a smaller piece of the label. This card is the unique case, where the arithmetic is visible.
