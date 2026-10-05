![Leakage Lab Shelf cover](docs/cover.jpg)

# Leakage Lab Shelf

A local gallery of planted machine-learning data leaks. Each experiment builds a synthetic table from a fixed seed, fits a leaky pipeline and a clean control, and writes down what the score is measuring.

The leaky number is the failed audit. The clean number is what the features support.

## Experiments

Seed **42**. ROC-AUC on the held-out rows. Accuracy, average precision, and the row counts are in the shelf output.

| Experiment | What was planted | Leaky ROC-AUC | Clean ROC-AUC |
| --- | --- | ---: | ---: |
| [Future cost](docs/experiments/future-cost.md) | A day-60 cost that includes the readmission | 0.9712 | 0.7325 |
| [Copied rows](docs/experiments/copied-rows.md) | Every claim pasted twice, then split by row | 0.9617 | 0.6978 |
| [Target encoding](docs/experiments/target-encoding.md) | `TargetEncoder.fit` on the whole table, then the split | 1.0000 | 0.7531 |

On the copied-rows card the forest is the model that memorizes a twin (accuracy 1.0000 on the copied test rows). A logistic regression on those same splits scores 0.7377 contaminated and 0.7350 clean. The dirty split flatters the forest and leaves the linear model where it was.

## Install

Python 3.11 or newer. The dependency pins are exact so a second machine prints the same four-decimal scores.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m leakage_lab shelf --seed 42
python -m leakage_lab shelf --seed 42 --html docs/shelf.html
pytest
```

`python -m leakage_lab` with no arguments runs the shelf at seed 42. `leakage-lab serve` writes the same HTML and serves it at `http://127.0.0.1:8000/`.

`--output reports/shelf.md` writes a Markdown report. `reports/` is gitignored. The HTML shelf committed at [docs/shelf.html](docs/shelf.html) is the seed-42 run.

A second run with seed 42 matches the first, including the scores above. Other seeds keep the same shape: the leaky path stays too good, and the clean path stays in the ordinary range. That is what `pytest` locks in.

## How a card is scored

Both paths use the positive-class probability. ROC-AUC is the ranking metric. Average precision is the area under the precision-recall curve. Accuracy uses a cutoff of 0.5, so it moves with the base rate (about 0.43 to 0.51 on these tables). The gap to read is the ROC-AUC gap.

The data generator is `numpy.random.Generator` (PCG64) seeded with the same integer as the split and the estimator. Nothing here is a benchmark on real records.

Measured with the pins in `pyproject.toml`: NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.14.1.

## Related

[SplitCheck](https://github.com/grummpy/SplitCheck) audits a train file and a test file you already have for copied rows, near-copies, and shared group ids. This shelf builds the synthetic cases and keeps the leaky and clean pipelines side by side. Either project runs without the other installed.

scikit-learn's own notes on this family of bugs: [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

## Layout

```
src/leakage_lab/                 shelf, scoring, text and HTML reports
src/leakage_lab/experiments/     one module per planted leak
tests/                           leaky vs clean, plus the seed-42 write-up
docs/cover.jpg                   project cover
docs/experiments/                what each card is doing
docs/shelf.html                  seed-42 shelf, open in a browser
```

## License

MIT. See [LICENSE](LICENSE).
