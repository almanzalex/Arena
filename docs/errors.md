# RLX diagnostic contract

Expected CLI failures emit `rlx.diagnostic/v1` under `--json` and a compact
equivalent on stderr otherwise. JSON stdout contains no progress text.

| Exit | Category | Meaning | First repair |
|---:|---|---|---|
| 2 | `usage` | Invalid command grammar | Run the exact command with `--help` |
| 3 | `schema_compatibility` | Invalid manifest, unsupported kind, or incompatible composition | Correct the field or consult `rlx schema list` |
| 4 | `integrity_authenticity` | Hash, signature, conformance, or qualification failed | Do not consume the artifact; restore trusted bytes/key |
| 5 | `external` | Store, executable, service, or worker unavailable/timed out | Run `rlx doctor --capability NAME` |
| 6 | `incomplete_cancelled` | Declared attempts did not all complete | Inspect the attempt ledger and retry to a new output |
| 70 | `internal` | Unexpected RLX defect | Retry with `--debug` and report the redacted diagnostic |

Stable high-level codes include `SCHEMA_INVALID`, `COMPATIBILITY_FAILED`,
`INTEGRITY_FAILED`, `CONFORMANCE_FAILED`, `EXTERNAL_UNAVAILABLE`,
`EXECUTION_INCOMPLETE`, `PLUGIN_LOAD_FAILED`, `EVALUATION_INCOMPLETE`, and
`MATCH_INCOMPLETE`. More specific additive codes may be introduced within the
same category.

Messages and context redact credential-shaped keys, bearer/token text, and
secret query parameters. `--debug` never changes the exit category and does not
put a traceback on JSON stdout.
