# Complexity router artifact

`complexity_router_v1.npz` is a five-class weighted multinomial Naive Bayes model trained from the audited `routing_dataset_100k_valid_only.jsonl` snapshot.

Artifact SHA-256: `6c96a674f590571b258bf5d798888a8670a8f93b21a2a121e51bc40a3bac0b89`

## Provenance

- Dataset rows: 86,967
- Dataset SHA-256: `bca11574006fac48d17f1efb67cd471822911e366b9a580755b68964499aa30e`
- Train/validation/test split: deterministic normalized-prompt hash, 80/10/10
- Train rows: 69,841
- Validation rows: 8,604
- Test rows: 8,522
- Features: hashed word unigrams, word bigrams, and structural conversation features
- Feature dimension: 32,768
- Audit weighting: `appropriate=1.0`; `borderline=0.65`; multiplied by audit confidence
- Class prior: uniform
- Probability calibration: temperature selected on the validation split

## Intended use

Use the artifact as a prompt-complexity signal behind the custom router's capability, budget, latency, context, health, and governance gates. `hybrid` mode combines the learned estimate with deterministic risk and capability features.

## Results

- Hash-held-out exact five-level accuracy: 90.93%
- Hash-held-out macro F1: 0.909
- Hash-held-out collapsed-tier under-route rate: 3.64%
- Novel, non-overlapping multi-turn slice accuracy: 57.14%
- Novel multi-turn collapsed-tier under-route rate: 18.68%

The distribution-shift result is a release blocker for using this artifact as the sole selector. It is suitable for shadow-mode integration and evaluation, not an unsupervised production rollout.

## Limitations

- Labels represent synthetic, LLM-audited complexity rather than measured candidate-model outcomes.
- Prompts were generated to fit the label rubric and can contain learnable stylistic cues.
- The model has not been evaluated on representative production traffic.
- The model does not know current model price, health, capability, tenancy, or compliance policy; those remain runtime router responsibilities.
