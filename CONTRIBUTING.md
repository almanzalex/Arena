# Contributing to Arena

Thanks for considering a contribution. Arena is a **protocol and tooling** layer:
changes should preserve artifact identity, fail loud on unknown kinds, and attach
qualification evidence before expanding “supported” claims.

## Development setup

```bash
git clone https://github.com/almanzalex/Arena.git
cd Arena
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

## Checks before opening a PR

```bash
ruff check .
python scripts/check_release_truth.py
pytest -q
```

For hermetic / clean-room changes also run:

```bash
pytest -m slow -q
```

## Design rules

1. **Registries over special cases.** New action, store, provider, packager, or
   trainer kinds go through `arena/plugins/` and fail with an extension recipe
   when unknown.
2. **No silent coercion.** Incomplete contracts refuse publication; errors name
   the repair.
3. **Evidence before marketing.** Do not claim an adapter is supported without
   `arena adapter qualify` (or the relevant qualify surface) and docs updates.
4. **Identity is sacred.** Digests must not change across identity-preserving
   mirrors. Simulated store paths (`?simulate=`) never satisfy live release gates.
5. **Keep the core small.** Optional integrations belong in extras.

## Documentation

- User-facing behavior → update the relevant guide under `docs/` and
  [docs/README.md](docs/README.md).
- Product boundary / stage claims → update RFCs and milestone evidence docs;
  do not quietly rewrite sealed “complete” records—add a newer evidence note.
- Public README stays short; deep material stays in `docs/`.

## Pull requests

- Prefer small, reviewable PRs with a clear “why.”
- Include tests for both the happy path and the fail-loud path.
- Do not commit secrets, live credentials, or generated `eval-run/` / `dist/` trees.

## Security

See [SECURITY.md](SECURITY.md).
