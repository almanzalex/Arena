# RLX clean-room usability sign-off

This checklist is for a researcher who did not create the policy or training
repository. Give them only the built wheel, received `.rlx` bundles,
`match.yaml`, and `docs/clean-room.md`. Do not give them this repository,
README, checkpoint, trainer source, or author assistance.

For **evaluation / population** handoff (0.2), use the companion form
[eval-usability-signoff.md](eval-usability-signoff.md) with the checked-in
`examples/eval/demo/` pack (or an equivalent received pack).

## Record

- Reader / machine / OS / Python:
- Wheel filename and hashes:
- Bundle and match file hashes:
- Start time (UTC):
- End time (UTC):
- Every command attempted, in order, with stdout/stderr and exit code:
- Any command not explicitly documented:
- Any ambiguous instruction, hidden assumption, or author intervention:
- Did `check`, the seeded match, and trajectory inspection succeed from the
  supplied files alone? Yes / no:
- Friction score (1 effortless — 5 blocked):
- Concrete guide/CLI changes requested:

## Required release decision

Pass only if a reader follows the guide without source access or author
intervention and all documented success criteria hold. A passing automated
blind-reader harness proves the supplied commands and isolation properties; it
does **not** prove prose clarity, unfamiliar-OS behavior, or human confidence.
Record those judgments here rather than silently treating automation as a human
sign-off.
