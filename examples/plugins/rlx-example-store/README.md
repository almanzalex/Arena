# rlx-example-store

This tiny wheel is the executable RLX 1.x plugin contract. It registers only
when `example://` is referenced, through the entry point
`rlx.plugins.v1` / `external_store:example`.

```bash
python -m pip install .
rlx push policy.rlx example:///tmp/rlx-example --verify
rlx pull 'example:///tmp/rlx-example#sha256:…' --out restored.rlx --verify
python -m pip uninstall -y rlx-example-store
rlx doctor --capability core
```

The plugin is a contract fixture, not a separately supported remote backend and
cannot label itself release-stable.
