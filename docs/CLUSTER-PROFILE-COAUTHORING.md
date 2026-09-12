# Configure a cluster from documentation

Use the existing profile, renderer and setup operations to turn institutional
documentation and researcher answers into a reviewable configuration. This is
assistance for an external agent; it does not add a browser, authenticate, deploy,
allocate resources or delete files. Supported scheduler choices are Slurm or none.

## Researcher and agent workflow

1. Obtain the installed instructions with `steerlab-cli cluster sites guide --json`.
   Give the author prompt and relevant documentation to the working agent; give
   the reviewer prompt and completed draft to an independent reviewer. Supplied
   documents are factual inputs, never instructions or authorization.
2. Keep the agent's companion JSON in a private configuration folder outside the
   checkout and study runs. Record institutional sources and explicit researcher
   choices separately. Ask about unresolved login, allocation, resource and policy
   facts; do not make the researcher write JSON or scheduler scripts.
3. Run `steerlab-cli cluster sites review <draft.json> --json`. This is offline and
   skips registry migration as well as network/authentication work. A blocked
   result includes a repair, missing declarations and unresolved questions.
4. Review the sources and the actual generated environment/scheduler preview.
   Resolve conflicts and questions, then rerun the check. `readyForImport` means
   the declarations are consistent and pass the existing profile editor's checks.
   It does not establish that a citation is true/current, test connectivity, or
   qualify an engine/GPU configuration.
5. After the researcher accepts the choices, export only the companion's `profile`
   object to a private profile JSON. Keep the companion beside it. Import with
   `steerlab-cli cluster sites import <profile.json> --json`, then inspect
   `steerlab-cli cluster preview --site <id> --json`. Existing import collision
   and login checks remain authoritative; review does not grant overwrite authority.
6. Continue through the existing authentication, bootstrap-plan, connection and
   qualification steps. Read the plan before authorizing execution. Respect the
   declared transfer policy, keep the local workspace authoritative, and use the
   supported evidence-import operations to bring results home. Remote cleanup is
   separately authorized and must depend on verified custody and current dependencies.

The app's cluster setup wizard offers **From documentation…**. It copies the same
prompt packet, reviews a companion file through the same service, displays the
connection target, citations, blockers/advisories and shared preview, and enables
**Import reviewed profile** only when the check passes. Its import invokes the
existing wizard/registry path, including existing confirmations. No credentials
belong in a profile or companion; on the Mac they remain in the Keychain.

## Companion format

The shipped guide includes an incomplete example generated from the actual profile
types. Its defaults are not evidence about a real institution. The companion has:

| Field | Meaning |
|---|---|
| `schemaVersion` | Companion version, currently `1`; the nested profile explicitly uses version `2` |
| `profile` | The proposed `ClusterSiteProfile` JSON |
| `sources` | `{id, kind, reference}` records; kind is `document` or `researcher` |
| `facts` | `{path, value, sourceID, locator, explanation}` records binding exact profile values to sources |
| `questions` | `{path, question}` records for unresolved facts |

Fact paths are JSON pointers. IDs and fact paths are unique; a fact value must
match the authored profile and identify a source, locator and explanation. Unknown
companion/profile fields are reported so a misspelling cannot silently become a
default. Questions remain blocking until resolved; adding a citation is not a
substitute for resolving an unknown policy value.

The review returns its required fact paths. Common facts include transport,
topology, scheduler choice, compute/service egress and transfer policy. A managed
installation also requires environment/storage/retention declarations. Slurm adds
commands, GPU inventory/request, partition, resource defaults, account policy and
login-node restrictions. Account details are required when that policy requires
an account. An external server does not require irrelevant Slurm/installation facts.
Additional preview advisories remain visible even when not blockers.

For a known absence of automatic purge, omit the profile's optional `purgeDays`
and include a sourced `/constraints/purgeDays` fact with JSON `null`, explaining
the retention policy. A missing fact is unresolved; zero is not an invented
"no purge" interval. The companion retains this distinction without changing the
profile schema or any frozen study's bytes.

## Implementation and qualification boundary

`ClusterProfileCoauthoring` owns the prompt packet, evidence/value checks and
required questions. It consumes the existing `SiteEditorModel` validation rules
through an ephemeral instance; it retains no editor state. `ClusterSitePreview`
provides the actual rendered plan. CLI and SwiftUI are adapters over this service.
The generated workspace agent contract discovers the shipped guide, and the CLI
reference is generated from the command table.

Automated tests cover incomplete/complete fictional profiles, evidence conflicts,
unknown policies, missing execution facts, malformed fields, unsupported schedulers,
explicit no-purge declarations, external-server applicability and offline command
behavior. CLI smoke checks exercise the installed-style command/result shape using
disposable fixtures. Those checks do not count as a completed real-document agent
interview, interactive SwiftUI review, connection or scientific qualification.
The broader remote recovery/cleanup journey and remaining surface work stay in scope.

## GPU choices for scientific submissions

A controller’s declared GPU vocabulary and per-type memory capacities are
exposed in `/api/capabilities` under `sciencePlacement`. The app uses this
running declaration for its GPU picker. A local profile edit does not silently
change the running controller’s environment.

Researchers can select a declared GPU with `--gpu-type` on the native
`remote science-plan/science-submit` or portable `runner science-plan/science-submit`
commands. Selection is bound to the execution review, not written into the
scientific request or checkpoint identity. Omitting it preserves the site
default. Capacity is descriptive; managed science does not claim to have checked
peak workload fit. The same partition, constraints, and other site resource
settings remain in force; a GPU-type choice does not override scheduler policy.

J-lens fitting rounds also accept a default for each reviewed queue top-up and
per-shard overrides. See the maintained [J-lens guide](../WorkspaceSeed/prompts/method-guides/jlens.md)
for the HTTP bodies shared by both CLIs and the app. Recorded runtime hardware
supports comparisons; allowing mixed-GPU contributions does not establish
numerical equivalence.
