# Contributing

Use Python 3.12 or newer and keep changes small enough to evaluate independently.

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
python -m build
```

Changes to the model catalogue must include an authoritative source, verification date, and evidence status. Changes to model selection, thresholds, features, labels, or training data must include reproducible before/after evaluation by workload slice—not only an aggregate score.

Never commit provider credentials, production prompts, provider responses, live evaluation results containing content, or customer identifiers. Use synthetic fixtures in tests.

Do not change `deployment_mode` to `enforce`, mark evidence `workload_measured`, or mark a release gate complete without attaching the reviewed source report and exact artifact/configuration versions.
