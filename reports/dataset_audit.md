# Private Datagen Audit

**Generated:** 2026-08-20T09:56:44.515923+00:00

This report contains aggregate counts and hashes only. It does not contain training prompts.

## Findings

- The primary dataset contains 86,967 unique valid prompts across 29 categories.
- 42,759 borderline rows include an unused adjacent audit-suggested label.
- The four multi-turn files contain 8,282 unique prompts in total, of which only 455 are novel relative to the primary dataset.
- Those four files are one metadata/schema family, not four independent datasets.
- Cross-file multi-turn label conflicts: 0.

## Decision

Keep the 455 novel multi-turn prompts isolated as an external diagnostic slice. Do not merge the metadata variants or count them repeatedly. Treat audit-suggested labels as experimental ambiguity evidence and promote them only if untouched evaluation improves.

## Limitation

All available prompts are synthetic and largely generator-related. Production approval still requires response-level outcomes on representative, time- or customer-separated workload data.
