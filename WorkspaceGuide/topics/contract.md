# The machine contract

<!-- client: all -->

## Discovering the surface with `--help`

**`--help` is how you discover the surface**, at three levels:

```bash
{{cli}} --help                      # every family
{{cli}} experiment --help           # that family's verbs, one line each
{{cli}} experiment attach --help    # one verb's positionals and flags
```

It is a declared flag on **every** verb, it runs **nothing**, and it exits 0
in both modes — so it is always safe to ask, including on a verb that would
otherwise write a manifest. With `--json` the same page comes back as data in
`result`, so you never have to parse the columns. Each page ends with the
exit-code line, and a verb page names every flag it accepts with its purpose
and its argument's shape, including closed vocabularies.

<!-- client: mac -->

For example, `set-instruments --help` prints the legal instrument names.

<!-- client: all -->

Secondary: running a family with no verb (`{{cli}} experiment`) refuses
and lists every verb it accepts. That roster answers "which verbs exist";
`--help` answers "and what do they take", so reach for `--help` first.

## The envelope

**Pass `--json` on every command.** In JSON mode:

- **Exactly one JSON document on stdout.** Every diagnostic, progress line,
  warning, and human report goes to stderr. No ANSI.
- Keys are sorted, dates are ISO-8601, there is exactly one trailing newline.
- `--json` is honored even when argument parsing itself fails.

<!-- client: mac -->

- `--out <path>` also writes the document to a file. (`--json <path>` is the
  deprecated spelling on the one verb that had it; it warns on stderr.)

<!-- client: python -->

- `--out <path>` also writes the document to a file. Two verbs own `--out`
  for their own archive instead: `bundle package` and `runner evidence`. Read
  their document from stdout.

<!-- client: all -->

- **Hashes are full.** The human lines elide them; the document never does.
  `freezeHash`, `taskPromptsHash`, `judgeRubricHash` and friends are complete
  in `result`.

Document shape:

```jsonc
{
  "schemaVersion": 1,          // the ENVELOPE's version, never the payload's
  "verb": "experiment freeze",
  "engine": "…",               // which engine answered
  "state": "refused",          // AUTHORITATIVE
  "changed": false,            // did this mutate durable state
  "observedAt": "2026-01-01T00:00:00Z",
  "message": "…",              // one sentence for a human
  "workspace": "/abs/path",    // which data root answered
  "advisories": [ { "code": "…", "detail": "…" } ],   // omitted when empty
  "nextAction": { "verb": "experiment validate demo", "requiresHuman": false,
                  "missingPermissionFlags": [], "detail": null },  // successes
  "error": { "code": "freezeGateFailed", "gate": "validateEvidence",
             "gates": ["validateEvidence", "judgeValidity"],
             "reason": "…", "repairAction": "…" },
  "result": { /* per-verb payload */ }
}
```

`schemaVersion`, `verb`, `engine`, `state`, `changed`, `observedAt`, and
`message` are always present. `workspace`, `advisories`, `nextAction`, `error`,
and `result` appear only when they have something to say — a missing key is a
straight answer, not a null.

### State and exit codes

**The JSON `state` is authoritative; the exit code is a convenience.**

| `state` | Exit | Meaning |
|---|---:|---|
| `ready` | 0 | the requested target is reached |
| `planned` / `running` | 0 | work remains / in progress |
| `okWithAdvisories` | 0 | succeeded, and `advisories[]` is non-empty |
| `needsHumanAuthentication` | 10 | a person must authenticate at their own terminal |
| `needsApproval` | 11 | a mutation needs its explicit `--allow-…` flag |
| `pending` | 12 | valid asynchronous work is in flight; repeat the command |
| `degraded` | 13 | retryable: a layer could not be read |
| `blocked` | 64 | malformed invocation or unusable configuration |
| `refused` | 65 | a gate declined a well-formed request against a healthy system |
| `notFound` | 66 | the named experiment, run, or panel does not exist — or a named artifact could not be read at all |
| `failed` | 70 | non-retryable operational failure |

<!-- client: mac -->

Two live caveats. **These codes are the `--json`-mode codes.** Without
`--json`, most failures still exit `1`; two verbs differ, and both differ in
both modes — `data check` exits `65` for blockers, and `vectors compare`
exits `1` when it compared and diverged but `2` when it could not compare at
all. One more reason to always pass `--json`. And an undeclared flag is exit
`64` in **both** modes, before the verb does any work: flags are parsed
strictly against a per-verb table, so a typo cannot silently change what a
study means.

<!-- client: python -->

This client derives the exit code from `state` in both modes, with or without
`--json`. An undeclared flag is exit `64` before the verb does any work: flags
are parsed strictly against a per-verb table, so a typo cannot silently change
what a study means.

<!-- client: all -->

Refusals are typed everywhere, not only at freeze: a lifecycle refusal
carries `error.code == error.gate` from a second closed vocabulary
(`statusImmutable`, `pinDrift`, `emptyStudy`, `studyDeclaration`,
`missingPrerequisite`, `promotionEvidence`, …), while freeze refusals keep
`code: "freezeGateFailed"` with the gate id in `error.gate`. Either way,
`error.repairAction` is an executable command sequence — run it, then retry.

Three of these describe a study that does not verify. `pinDrift` means a
pinned file changed, is missing, or appeared. `emptyStudy` means nothing is
attached yet: attach a concept, attach an agent, or start from a template.
`studyDeclaration` means the study's own settings are incomplete or
inconsistent.

<!-- client: python -->

Every repair this client shows is spelled with this client's verbs. Where a
step needs a model, the repair names the runner route (`steerlab run <name>
--runner <url> --verb …`); `<url>` is the runner the researcher gave you. The
one other program a repair may name is `steerlab-server`, for an act only an
engine can do, such as probing a chat template. Never type a `steerlab-cli`
command here: that is the Mac command line.

<!-- client: all -->

### Advisories

`advisories[]` never changes the exit code. An advisory is something you should
know that did not stop the verb: a skipped freeze gate, a vacuous validation, a
one-judge panel, an empty analysis. **Read them.** Treating them as failures
will make you refuse to walk a legitimate lifecycle; ignoring them will make
you produce results that are stamped as not citable.

### Refusals

`state: "refused"` means a gate declined a well-formed request. It is not a
transient error. **`error.repairAction` is the field to read**: today
`nextAction` is emitted on *successes*, where it names the next lifecycle step,
and a refusal carries `error` without one. **Never retry a refusal without
performing the repair first.**
