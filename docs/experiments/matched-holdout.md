# Matched holdout copies

This card is deliberately separate from the existing copied-rows demo. The original demo begins with a doubled table and then splits it, which correctly demonstrates a common contamination pattern but gives the clean comparison a different test table.

The matched card fixes one synthetic holdout table. Both models score every one of those rows. The contaminated training frame receives copies of half the holdout claims; the equal-size control receives copies drawn only from the original training side. Its purpose is not realism—it isolates copied holdout rows as the only explanatory difference.

Run `python -m leakage_lab shelf --seed 42` for the current deterministic scores. The card reports copied-versus-untouched holdout accuracy in addition to the two ROC-AUC values.
