# Leakage Lab Shelf: answer key

1. A forest can memorize copied training twins. The mixed copied-rows demo reports the contaminated subset separately; the matched-holdout card removes changing-test-population as an explanation.
2. The matched card generates synthetic claims, fixes one heldout set, and scores that exact set twice. The contaminated training frame receives copies of half the holdout rows. The clean frame receives an equal number of duplicates from its own training rows, so it contains no holdout row.
3. Split by `member_id` before fitting. Keep `TargetEncoder` inside the model pipeline so it fits only on training members. A test member missing from that fit receives the training prevalence, not a mean built from its own outcome.
4. Future cost is future information. The original and matched copied-row cards are train/test contamination. The unique-id encoding card is preprocessing fit on the full table. The repeated-entity card is group leakage even though its encoder is fit only on train rows in the random-split path.
5. The export writer produces ordinary synthetic CSVs and a JSON note; it does not import SplitCheck. An audit should find copied rows in the first pair and shared `member_id` values in the repeated-entity random pair.
