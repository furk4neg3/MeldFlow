# Contributing to MeldFlow

Thanks for taking the time to contribute! This is a personal project with a professional bar: small PRs, clear tests, friendly tone. By participating, you agree to our [Code of Conduct](./CODE_OF_CONDUCT.md).

---

## TL;DR checklist

* Open an issue first for non‑trivial changes.
* Use Docker for dev if possible (fast, reproducible).
* Run **`make test`** locally before opening a PR.
* Keep PRs focused; add/adjust docs as needed.
* Prefer Conventional Commits in messages (see below).

---

## Getting started

### 1) Fork & clone

```bash
git clone https://github.com/furk4neg3/MeldFlow.git
cd meldflow
```

### 2) Recommended: Docker workflow

Build once:

```bash
docker build -t mm-analytics:latest .
```

Generate data:

```bash
docker run --rm -v "$PWD":/app mm-analytics \
  python scripts/generate_synth_dataset.py --out_dir data/synth --num_samples 600 --image_size 64
```

Train:

```bash
docker run --rm --shm-size=1g \
  -v "$PWD":/app -v hf_cache:/root/.cache/huggingface \
  mm-analytics \
  python scripts/train.py --config config/default.yaml
```

Serve:

```bash
docker run --rm -p 8000:8000 -v "$PWD":/app \
  -e MM_CKPT=artifacts/best_model.pt \
  mm-analytics
```

Quick API smoke:

```bash
curl -X POST http://127.0.0.1:8000/predict-multipart \
  -F text="red square low value" \
  -F 'tabular_json={"num_a":0.1,"num_b":2.0,"cat_x":"A"}'
```

### 3) Local (no Docker)

* Python 3.11 recommended
* Install deps and run tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test  # or: pytest -q
```

---

## Dev flow

### Branches

Use short, descriptive names:

* `feat/<thing>` new feature
* `fix/<bug>` bug fix
* `docs/<topic>` docs only
* `chore/<task>` CI/build/tooling
* `test/<area>` tests

### Commit messages (Conventional Commits)

```
<type>(optional-scope): short summary

body (optional)
```

**Types:** `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, `build`, `ci`
Examples: `feat(text): add BOW fallback`, `fix(api): handle empty tabular payload`

### Code style

* Keep functions small and typed where practical.
* Prefer straightforward naming over cleverness.
* If configured, run formatters/linters before committing:

```bash
make format  # optional target if available
# or manually
python -m black .
python -m ruff check . --fix || true
```

### Tests

* Add/adjust tests for your changes. The suite includes unit tests + a smoke train→infer run.
* Run locally:

```bash
make test
# or
pytest -q
```

### Pull requests

* Keep PRs under \~400 lines of diff when possible.
* Fill a short description with **what/why**, screenshots/logs if relevant.
* Link the related issue.

---

## Proposing significant changes

Open an issue labeled **proposal** with a short design note:

* Problem, rough approach, alternatives
* Impact on config/API/backward compatibility

---

## Security

Please avoid filing public issues for sensitive bugs. Email **[nizamfurkanegecan@gmail.com](mailto:nizamfurkanegecan@gmail.com)** and we’ll coordinate a fix.

---

## License & attribution

By contributing, you agree your contributions are licensed under the project’s license (MIT). You confirm you have the right to contribute the content.

---

Thanks again for helping make MeldFlow better! 🙌
