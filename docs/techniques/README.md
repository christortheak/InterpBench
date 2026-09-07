# Maintained operation specifications

Each `operations/<id>.json` owns its catalog row and, for managed operations,
its interview, execution binding and input-role selection. `registry.json`
holds method categories, the shared interview answer envelope and named input
role vocabularies. Math stays in the existing Python/Swift owners.

Run `python scripts/ci/check-generated.py --write` from the checkout root.
This generates catalog/interview JSON, the lightweight Python binding module,
packaged/compiled copies and source identity in dependency order. `--list`
shows generator and audit discovery; `--audits` runs existing immutable-baseline
checks. Supply `--cli <source-built-helper>` for the Swift CLI reference too.

`order` and `interviewOrder` preserve the intentional presentation order.
Names and bodies must agree across spec filename, catalog and interview.
An ordinary binding names `module`, `config_class`, `function`, and `compute`.
A special binding names an existing special dispatch branch; declaring one
does not implement it. Non-managed operations have catalog rows only.

`inputRoleProfile: "managed-v1"` preserves the existing recursive reference
vocabulary. `inputRoles` supplies operation-specific key overrides. Recognized
roles are `artifact`, `artifacts`, `file`, `trees`, and `lens`; they describe
dependency closure, not train/selection/final-test scientific roles. Those
remain explicit in the interview and scientific owner. Test every new reference
through review, packaging and actual owner execution.

No installed-client scaffold is added. The disposable worked example already
exercises adding a spec and owner; an automatic scaffold is optional future
developer convenience, not a new researcher setup requirement.

See [the complete guide](../ADDING-A-TECHNIQUE.md) and
[the exercised example](../ADDING-A-TECHNIQUE-EXAMPLE.md).
