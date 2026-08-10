# arena-example-store

This tiny wheel is the executable Arena 1.x plugin contract. It registers only
when `example://` is referenced, through the entry point
`arena.plugins.v1` / `external_store:example`.

```bash
python -m pip install .
arena store qualify examples/eval/demo/rock.arena example:///tmp/arena-example
arena push policy.arena example:///tmp/arena-example --verify
arena pull 'example:///tmp/arena-example#sha256:…' --out restored.arena --verify
python -m pip uninstall -y arena-example-store
arena doctor --capability core
```

The plugin is a contract fixture, not a separately supported remote backend and
cannot label itself release-stable.
