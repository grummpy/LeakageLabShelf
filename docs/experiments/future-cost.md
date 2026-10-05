# Future cost

Target leakage via a feature from the future.

The label is 30-day readmission. At discharge the model may see age, prior admissions, length of stay, and one lab. The leaky pipeline also sees `cost_through_day_60`. That total is still accumulating during the 30-day window. In the generator it is

```
1800 + 90 * length_of_stay + 3200 * readmitted_30d + Normal(0, 1300)
```

Length of stay is already a discharge-time column, and it moves the cost a little. The 3200 charge is the readmission the model is supposed to predict. The noise keeps the column from being a pure copy of the label. On the seed-42 test rows the correlation with the label is 0.7644.

## What is held fixed

Same 2,400 synthetic discharges. Same stratified 25% holdout (600 test rows). Same logistic regression. The scaler is fit on the training rows in both paths. The future cost is the only column that appears in one path and not the other.

## Seed 42

Seed 42: leaky ROC-AUC 0.9712, clean ROC-AUC 0.7325.

Accuracy at a probability cutoff of 0.5 is 0.8983 leaky and 0.6650 clean. The base rate is about 0.43, so accuracy sits lower than ROC-AUC on the clean path. The ranking gap is the one to quote.

The clean score is a real model of the discharge-time fields. It is ordinary. The leaky score is what you get when the bill for the event you are predicting is sitting in `X`.

## Where this shows up

A notebook adds "total cost" or "final claim amount" because it is on the extract, and the extract was pulled after the outcome window closed. The column looks like a covariate. Part of it is the label, delayed.
