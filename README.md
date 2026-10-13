![Leakage Lab Shelf cover](docs/cover.jpg)

# Leakage Lab Shelf

A local gallery of planted machine-learning data leaks. Each experiment builds a synthetic table from a fixed seed, fits a leaky pipeline and a clean control, and writes down what the score is measuring.

The leaky number is the failed audit. The clean number is what the features support.

## Experiments

Seed **42**. ROC-AUC on the held-out rows. Accuracy, average precision, and the row counts are in the shelf output.

| Experiment | What was planted | Leaky ROC-AUC | Clean ROC-AUC |
| --- | --- | ---: | ---: |
| [Future cost](docs/experiments/future-cost.md) | A day-60 cost that includes the readmission | 0.9712 | 0.7325 |
| [Copied rows](docs/experiments/copied-rows.md) | Every claim pasted twice, then split by row | 0.9618 | 0.6980 |
| [Matched holdout copies](docs/experiments/matched-holdout.md) | Same holdout, with copies inserted into only one training path | 0.9089 | 0.7015 |
| [Target encoding](docs/experiments/target-encoding.md) | `TargetEncoder.fit` on the whole table, then the split | 1.0000 | 0.7531 |
| [Group-aware encoding](docs/experiments/group-aware-encoding.md) | Repeated members: train-only encoder with random rows versus group holdout | 0.9993 | 0.7440 |

On the copied-rows card the forest is the model that memorizes a twin (accuracy 1.0000 on the copied test rows). A logistic regression on those same splits scores 0.7377 contaminated and 0.7350 clean. The dirty split flatters the forest and leaves the linear model where it was.

## Install

Python 3.11 or newer. The dependency pins make the numerical stack reviewable,
but they do not turn a captured score table into a cross-platform benchmark.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m leakage_lab shelf --seed 42
python -m leakage_lab shelf --seed 42 --html docs/shelf.html
python -m leakage_lab shelf --seed 42 --manifest reports/shelf-manifest.json
python -m leakage_lab shelf --seed 42 --multi-seed-summary
pytest
```

`python -m leakage_lab` with no arguments runs the shelf at seed 42. `leakage-lab serve` writes the same HTML and serves it at `http://127.0.0.1:8000/`.

`--output reports/shelf.md` writes a Markdown report. `reports/` is gitignored.
The HTML shelf committed at [docs/shelf.html](docs/shelf.html) is an illustrative
seed-42 capture, not an assertion that every supported Python/platform runtime
will print byte-identical scores. Its checked-in capture contract is
[docs/shelf-capture.json](docs/shelf-capture.json): CI compares the complete
HTML only on that declared runtime, while every other runtime checks the
capture's provenance rather than borrowing its numbers as current output.

A second run with seed 42 matches the first **in the same runtime**. Use
`--manifest` whenever retaining or comparing a run: it records the synthetic
input, split row counts, seed, source revision, and runtime versions. Combined
with `--multi-seed-summary`, it also retains every paired leaky/clean ROC-AUC
result for the predeclared set `0, 7, 42`. The summary reports the paired-gap
mean and range as descriptive synthetic output—not a confidence interval or
real-world benchmark—and never selects favorable seeds after seeing results.

`--splitcheck-export DIR` writes optional synthetic CSV pairs for a separate split-audit tool. Leakage Lab Shelf does not import or require that tool. Student prompts and a separate answer key are in [docs/student-prompts.md](docs/student-prompts.md) and [docs/answer-key.md](docs/answer-key.md).

## How a card is scored

Both paths use the positive-class probability. ROC-AUC is the ranking metric. Average precision is the area under the precision-recall curve. Accuracy uses a cutoff of 0.5, so it moves with the base rate (about 0.43 to 0.51 on these tables). The gap to read is the ROC-AUC gap.

The data generator is `numpy.random.Generator` (PCG64) seeded with the same integer as the split and the estimator. Nothing here is a benchmark on real records.

The declared stack is NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, and SciPy
1.14.1. A saved manifest, rather than the table alone, is the evidence for a
specific captured output.

## Related

[SplitCheck](https://github.com/grummpy/SplitCheck) audits a train file and a test file you already have for copied rows, near-copies, and shared group ids. This shelf builds the synthetic cases and keeps the leaky and clean pipelines side by side. Either project runs without the other installed.

scikit-learn's own notes on this family of bugs: [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

## Layout

```
src/leakage_lab/                 shelf, scoring, text and HTML reports
src/leakage_lab/experiments/     one module per planted leak
src/leakage_lab/provenance.py    versioned run manifest and predeclared seed set
src/leakage_lab/exports.py       optional standalone synthetic CSV exports
tests/                           leaky vs clean, plus the seed-42 write-up
docs/cover.jpg                   project cover
docs/experiments/                what each card is doing
docs/shelf.html                  seed-42 shelf, open in a browser
docs/student-prompts.md          predict, trace, and repair exercises
docs/answer-key.md               separate instructor answers
```

## License

MIT. See [LICENSE](LICENSE).
