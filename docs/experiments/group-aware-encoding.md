# Group-aware repeated-entity encoding

Train-only fitting is necessary but not sufficient when one entity has repeated rows. A random row split puts most members on both sides. A target encoder fit only on training rows can then legitimately transform a test row with that member's training-row mean—yet the evaluation is still contaminated by group overlap.

The clean path splits by `member_id` first and keeps `TargetEncoder` inside the training-only pipeline. Test members are unseen and receive the training prevalence. This is a separate lesson from the unique-id full-table encoder card.
