import CryptoKit
import Foundation

// =============================================================================
// AGENTS.md — the contract every workspace carries (WP0 step 10)
//
// EDITS GO TO `docs/AGENTS-WORKSPACE-DRAFT.md` FIRST, then get mirrored here.
// That document is the human source of truth; this constant is the shipping
// copy, and `AgentContractTests.agentContractMatchesTheDraftDocument` asserts
// the two are byte-identical (modulo the generated header line below and the
// draft's own `<!-- … -->` markers, which are stripped). Editing only one of
// them fails that gate rather than drifting silently.
//
// Emitted in CODE, not as a seed-tree file, per WP0-AGENT-SURFACE-AUDIT §4.3.
// The original reason was positional — `CodeResources.Family.workspaceSeed`
// resolved to the CHECKOUT ROOT in developer mode, so a template at the
// family root would have been `<checkout>/AGENTS.md`, the INTERNAL
// engineering pointer, copied into every workspace. WP1 moved the family to
// the curated `WorkspaceSeed/` tree, so that hazard is gone; the reason that
// remains is the durable one: a generated string is the only form the drift
// gates above can check against the parser's own verb table.
// =============================================================================

/// The workspace-facing agent contract: what `AGENTS.md` says, and the file
/// name it is written under. `WorkspaceStore` writes it at creation, lazily on
/// open, and — since the file's header started carrying a hash of its own body
/// — refreshes it in place while that hash proves nobody has edited it. A file
/// the hash does not vouch for is never touched.
public enum AgentContract {

    /// The file, at the workspace root.
    public static let fileName = "AGENTS.md"

    // MARK: - The header line

    /// **The header format.** One line, first in the file, one HTML comment so
    /// it does not render as prose and so stripping comment lines recovers the
    /// draft exactly. Shape:
    ///
    /// ```
    /// <!-- …fixed prose… sha256:<64 lowercase hex> -->
    /// ```
    ///
    /// The hex is the SHA-256 of **the body bytes this writer emitted after
    /// the header**, in the same normalized form `classify` compares: the
    /// file's text after the header line's newline, with at most one leading
    /// blank line removed. So `contents()` — header, blank line, body — hashes
    /// exactly `body`, and a copy that lost the blank line still verifies.
    ///
    /// That hash is what turns the old header-intact HEURISTIC into a proof:
    /// a file whose header hash matches its body is one SteerLab wrote and
    /// nobody has edited, which is the only condition under which this build
    /// rewrites it.
    static let headerPrefix =
        "<!-- Written by SteerLab workspace seeding. SteerLab keeps this file "
        + "current for you while this line's hash still matches the text under "
        + "it, and never touches it once you edit that text. sha256:"

    /// Closes the HTML comment. Everything between prefix and suffix is hex.
    static let headerSuffix = " -->"

    /// The header builds before the hash existed: the same promise, no proof,
    /// and it named a manual repair (delete + reopen) because that was the
    /// only one there was. **Recognised forever, never written again.** A file
    /// carrying it is treated exactly as this build's predecessors treated it
    /// — heuristic classification, advisory only, never rewritten — and the
    /// one manual regeneration the advisory names is what graduates it into
    /// the hashed regime.
    static let legacyGeneratedHeader =
        "<!-- Written by SteerLab workspace seeding; safe to regenerate — "
        + "delete this file and reopen the workspace to get it back. SteerLab "
        + "never overwrites an existing AGENTS.md, so local edits survive. -->"

    /// The header line for a given body — the writer's half of the format
    /// above, and the seam a test uses to age a fixture backwards honestly
    /// (an older build's body under an older build's *correct* hash).
    public static func header(for body: String) -> String {
        headerPrefix + sha256Hex(body) + headerSuffix
    }

    /// The header line this build writes, over the body this build ships.
    public static var generatedHeader: String { header(for: body) }

    /// The hash a header line declares, or nil when the line is not one of
    /// ours in the hashed format (a legacy header, a tampered one, prose).
    static func declaredBodyHash(inHeaderLine line: String) -> String? {
        guard line.hasPrefix(headerPrefix), line.hasSuffix(headerSuffix),
            line.count >= headerPrefix.count + headerSuffix.count + 64
        else { return nil }
        let hex = line.dropFirst(headerPrefix.count).dropLast(headerSuffix.count)
        guard hex.count == 64, hex.allSatisfy(\.isHexDigit),
            hex.lowercased() == hex
        else { return nil }
        return String(hex)
    }

    /// Lowercase hex SHA-256 of a string's UTF-8 bytes.
    static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    /// The contract text, byte-identical to `docs/AGENTS-WORKSPACE-DRAFT.md`
    /// with its draft-only comment markers removed. Ends with exactly one
    /// newline.
    public static let body: String = literal + "\n"

    /// The bytes written into a workspace: header line, blank line, body.
    public static func contents() -> String {
        generatedHeader + "\n\n" + body
    }

    // MARK: - Staleness

    /// What a workspace's `AGENTS.md` is, relative to the contract THIS build
    /// ships.
    ///
    /// The contract is written at workspace creation or on the first open of
    /// an older workspace. A file that is **provably** still the machine's —
    /// its header hash matches its body — is refreshed in place when the
    /// shipped contract moves on; anything else is never overwritten. That
    /// asymmetry is the whole design: the contract is documentation, it alters
    /// no run, and a workspace made before a contract revision otherwise keeps
    /// the old text forever, silently, while its runner agent reads
    /// instructions that no longer describe the CLI.
    ///
    /// Nothing in *this* type writes: it is a cheap read, a classification and
    /// two sentences. `WorkspaceStore` owns the one write it enables.
    public enum Status: Sendable, Equatable {

        /// No `AGENTS.md` at the workspace root. Silent by design: the
        /// upkeep path regenerates an absent contract on its own, so there is
        /// nothing for a person to repair.
        case absent

        /// Byte-identical to what this build would write. (Also the answer for
        /// a legacy hashless header over the current body: there is nothing to
        /// do and nothing to say.)
        case current

        /// **Proven machine-owned and behind.** The header carries a hash, and
        /// that hash is the hash of this file's own body — so SteerLab wrote
        /// these exact bytes and nobody has changed them since — but the body
        /// is not the body this build ships. This is the one state that gets
        /// rewritten automatically.
        ///
        /// `linesBehind` counts lines of the SHIPPED body that this copy does
        /// not have (see `missingLineCount` for exactly what that means and
        /// what it deliberately is not).
        case staleProven(linesBehind: Int)

        /// **A legacy header, intact, over an older body.** Written by a build
        /// from before the hash: the header says SteerLab wrote the file and
        /// nobody has touched the line that says so, which is a heuristic and
        /// not a proof. Advisory only — this state is never rewritten, and the
        /// one manual regeneration the advisory names graduates the file into
        /// the hashed regime, after which it refreshes itself.
        case staleUnedited(linesBehind: Int)

        /// The header is gone, altered, or hashed-but-not-matching — or the
        /// file is there but unreadable as text. **The researcher owns this
        /// file now.** Silent on every surface and never written: they chose
        /// their text, and a tool that nags about a file it promised never to
        /// touch is a tool that will be worked around.
        case edited
    }

    /// **Proof where there used to be a heuristic.**
    ///
    /// The first line of the file decides everything, and there are three
    /// kinds of it:
    ///
    /// 1. **Hashed header** (`headerPrefix … sha256:<hex> -->`). Recompute the
    ///    hash over the body actually present. Match → SteerLab wrote these
    ///    exact bytes and nobody has edited them: `current` if the body is the
    ///    shipped body, else `staleProven` — the state that is safe to rewrite
    ///    without asking, because we can *show* no human text is at risk.
    ///    Mismatch → someone edited the body under our header: `edited`, hands
    ///    off, no notice.
    /// 2. **Legacy hashless header**, byte-for-byte. Exactly the pre-hash
    ///    behaviour, deliberately unchanged: `current` or `staleUnedited`,
    ///    advisory only, never written. The heuristic's one wrong direction —
    ///    a body edited under an intact header reads as `staleUnedited` — is
    ///    still wrong in the harmless direction *because nothing writes here*.
    /// 3. **Anything else**, including a tampered header and a file with no
    ///    newline at all: `edited`.
    ///
    /// A pure seam: text in, classification out, no filesystem.
    public static func classify(_ text: String) -> Status {
        guard let split = splitHeader(text) else { return .edited }
        let (headerLine, rest) = split

        if let declared = declaredBodyHash(inHeaderLine: headerLine) {
            guard declared == sha256Hex(rest) else { return .edited }
            if rest == body { return .current }
            return .staleProven(
                linesBehind: missingLineCount(shipped: body, workspace: rest))
        }

        guard headerLine == legacyGeneratedHeader else { return .edited }
        if rest == body { return .current }
        return .staleUnedited(
            linesBehind: missingLineCount(shipped: body, workspace: rest))
    }

    /// A contract file's first line and the body under it, in the ONE
    /// normalized form everything downstream uses: the text after the header
    /// line's newline, with at most one leading blank line removed
    /// (`contents()` writes that blank line; a copy that lost it is not an
    /// edit). Nil when there is no first line to speak of.
    ///
    /// The header's hash is computed over exactly this, which is what makes
    /// the proof survive the same normalization the comparison does.
    static func splitHeader(_ text: String) -> (header: String, body: String)? {
        guard let breakIndex = text.firstIndex(of: "\n") else { return nil }
        var rest = String(text[text.index(after: breakIndex)...])
        if rest.hasPrefix("\n") { rest.removeFirst() }
        return (String(text[text.startIndex..<breakIndex]), rest)
    }

    /// The body a contract file carries, normalized as above — the text a
    /// refresh is replacing, and therefore the text a "lines changed" count
    /// must be about. Empty for a file with no header line at all.
    static func bodyText(of fileText: String) -> String {
        splitHeader(fileText)?.body ?? ""
    }

    /// Classify the `AGENTS.md` at a workspace root. Never throws and never
    /// writes: an unreadable-but-present file is `edited` (hands off — we
    /// cannot see it, so we cannot claim it is ours), a missing one is
    /// `absent`.
    public static func status(at workspaceRoot: URL) -> Status {
        let url = workspaceRoot.appending(component: fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .absent
        }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return .edited
        }
        return classify(text)
    }

    /// How many lines the shipped body has that the workspace's copy does
    /// not, counting duplicates — a multiset difference, one pass, no
    /// allocation per line beyond the tally.
    ///
    /// **Honest about what it is not:** this is not an LCS diff and does not
    /// claim to be. It cannot tell a moved line from a deleted one, and it
    /// reports 0 for a copy that only ADDED lines. It answers exactly one
    /// question — "how much of the current contract is missing here" — which
    /// is the question the advisory asks, and it answers it in linear time
    /// over a file read once. `stalenessAdvisory` words the 0 case as
    /// "out of date with" rather than "0 lines behind", so the number is
    /// never asked to mean more than it does.
    static func missingLineCount(shipped: String, workspace: String) -> Int {
        var have: [Substring: Int] = [:]
        for line in workspace.split(
            separator: "\n", omittingEmptySubsequences: false)
        {
            have[line, default: 0] += 1
        }
        var missing = 0
        for line in shipped.split(separator: "\n", omittingEmptySubsequences: false) {
            if let count = have[line], count > 0 {
                have[line] = count - 1
            } else {
                missing += 1
            }
        }
        return missing
    }

    /// How many lines differ between two bodies, in both directions —
    /// `missingLineCount` symmetrized, so a refresh that only ADDS lines still
    /// reports a number rather than 0.
    ///
    /// Same honesty caveat as its half: a multiset difference, not an LCS
    /// diff. Two bodies with the same lines in a different order report 0, and
    /// `refreshNotice` words that case rather than printing "0 lines changed".
    static func changedLineCount(from old: String, to new: String) -> Int {
        missingLineCount(shipped: new, workspace: old)
            + missingLineCount(shipped: old, workspace: new)
    }

    /// The one advisory sentence, or nil when there is nothing to say.
    ///
    /// Fires for `staleUnedited` ONLY — the LEGACY hashless header, the one
    /// state we can neither vouch for nor rewrite. `current` has nothing to
    /// report, `absent` regenerates itself, `staleProven` is refreshed in
    /// place and speaks through `refreshNotice`, and `edited` is the
    /// researcher's file. Non-blocking everywhere it is used: it changes no
    /// exit code and gates nothing.
    ///
    /// Prefix-free on purpose — the CLI stamps its own `advisory: ` in front,
    /// the app's notice feed carries a severity instead, and neither surface
    /// has to own the other's punctuation.
    ///
    /// The repair it names is real on both surfaces — the app rewrites an
    /// absent contract on open (`WorkspaceStore.open`), and the CLI does the
    /// same once per invocation on its resolution path
    /// (`ExperimentCLIRunner.agentContractUpkeepLine`). Delete the file, run
    /// anything, get the current contract back — and the file that comes back
    /// carries a hashed header, so this is the last time the repair is
    /// manual.
    public static func stalenessAdvisory(at workspaceRoot: URL) -> String? {
        stalenessAdvisory(for: status(at: workspaceRoot), at: workspaceRoot)
    }

    /// The wording, from an already-computed status — so a caller that has to
    /// classify before acting (both surfaces do: they classify, then write)
    /// does not read the file twice.
    public static func stalenessAdvisory(
        for status: Status, at workspaceRoot: URL
    ) -> String? {
        guard case .staleUnedited(let linesBehind) = status else { return nil }
        let extent =
            linesBehind > 0
            ? "\(linesBehind) line\(linesBehind == 1 ? "" : "s") behind"
            : "out of date with"
        let path = workspaceRoot.appending(component: fileName).path
        return "this workspace's \(fileName) is \(extent) the agent contract "
            + "this build ships, and its machine header shows it unedited — "
            + "delete \(path) and reopen the workspace (or run any workspace "
            + "verb) to regenerate it. That one manual refresh is the last: "
            + "the regenerated file carries a hashed header, and SteerLab "
            + "refreshes a hashed, unedited contract for you from then on"
    }

    /// The one notice sentence for a contract this build just rewrote.
    ///
    /// Fires for `staleProven` and nothing else, once per open on each
    /// surface — the same discipline and the same channels as the advisory
    /// above (CLI stderr, the app's `"Workspace"` notice feed), and the same
    /// prefix-free wording for the same reason.
    ///
    /// It is a report, not a request: the work is already done and there is
    /// nothing for the reader to repair. It says so, and it says why the
    /// rewrite was safe — the header hashed the text it wrote, and that hash
    /// still matched.
    public static func refreshNotice(
        linesChanged: Int, at workspaceRoot: URL
    ) -> String {
        let extent =
            linesChanged > 0
            ? "\(linesChanged) line\(linesChanged == 1 ? "" : "s") changed"
            : "same lines, reordered"
        let path = workspaceRoot.appending(component: fileName).path
        return "refreshed \(path) to the agent contract this build ships "
            + "(\(extent)) — its machine header hashed the text it wrote and "
            + "that hash still matched, so nobody had edited it; nothing else "
            + "in the workspace was touched"
    }

    // Raw literal (`#"""`) on purpose: the contract is full of shell
    // continuations (`\` at end of line) and one escaped table pipe, none of
    // which may be interpreted as Swift escapes.
    private static let literal = #"""
# AGENTS.md

You are working inside a **SteerLab data workspace**. This file is the
contract: read it before running anything. It describes the folder, the
command surface, the machine-readable output contract, and the refusals you
will hit. Every command below is real; every file shape below is the one the
loaders actually parse.

This is *data*, not code. The SteerLab source tree lives elsewhere. Nothing
here is built or compiled.

---

## 1. What this folder is

A workspace is a plain folder that is its own git repository, created with an
initial commit. Layout:

```
prompts/          git-versioned inputs (see §3)
experiments/      experiment manifests — freezable recipes
runs/             immutable run outputs (gitignored)
catalog/          GENERATED navigation over runs/ (symlinks; gitignored)
adapters/         per-adapter training data and outputs
WORKSPACE.md      the marker file that makes this a workspace
.gitignore        runs/, catalog/, adapters/**/*.safetensors, .DS_Store
```

A folder counts as a workspace if it carries `WORKSPACE.md` or at least a
`prompts/` directory.

An experiment is a **recipe**, not results: it pins inputs by SHA-256 plus the
options used to derive vectors from them, and runs re-derive deterministically.
That pinning is the firewall — settings are chosen and frozen *before* behavior
is measured, so a result cannot be reverse-fitted to the settings that produced
it.

---

## 2. Pointing the CLI at this workspace

Resolution order, exactly: (1) `STEERLAB_WORKSPACE`; (2) `--workspace <dir>`,
or the app's in-process override; (3) the app's persisted choice, honored only
while that directory still exists; (4) a compiled-in development fallback.

**Set the environment variable once at the start of your session** rather than
passing `--workspace` on every call:

```bash
export STEERLAB_WORKSPACE=/abs/path/to/this/workspace
```

Every JSON response carries a top-level `workspace` field — a sibling of
`state`, never something under `result` — naming the root that answered, so you
can tell a wrong-workspace answer from a wrong answer. Check it on your first
command, and compare **resolved** paths: it echoes the path you configured with
symlinks intact, so `/tmp/ws` and `/private/tmp/ws` are one directory
disagreeing on paper. `realpath` both sides before concluding they differ.

If this folder is not a workspace yet — no `WORKSPACE.md`, no `prompts/` — run
`steerlab-cli workspace init <path>` first. `experiment create` will *not* do
this for you: it will build a half-workspace and say nothing.

---

## 3. Where things live

| Path | What goes there | Shape |
|---|---|---|
| `prompts/concepts/<name>/positive.jsonl` | contrastive stimuli, concept-present | `{"text": "…"}` per line |
| `prompts/concepts/<name>/negative.jsonl` | contrastive stimuli, matched control | `{"text": "…"}` per line |
| `prompts/concepts/<name>/validation.jsonl` | held-out probe (§4.4) | `{"text": "…", "expresses": true\|false}` per line |
| `prompts/concepts/<name>/markers.json` | optional marker word list | `{"words": ["…"]}` |
| `prompts/tasks/*.jsonl` | the measured task prompts | `{"id", "prompt", ["options", "target"]}` per line; ids unique |
| `prompts/rubrics/*.md` | judging instruments (Markdown) | prose rubric; pinned by file, not inline text |
| `prompts/batteries/*.jsonl` | capability probes | `{"prompt", "answer", "grading"}`; grading ∈ `exact_number`, `yes_no`, `token_exact`, `exact_normalized`, `regex` |
| `prompts/neutral/corpus.jsonl` | neutral corpus that denominates norm-unit α | `{"text": "…"}` per line |
| `prompts/templates/`, `probes/`, `readers/`, `dev/`, `panels/`, `parsers/`, `generation/`, `emotions/` | other pinned inputs | per their loaders |
| `experiments/<name>/experiment.json` | the manifest | written by the CLI; edit through verbs, not by hand |
| `experiments/<name>/pinned/` | freeze-time snapshot of every pinned input | written by `freeze`; read-only |
| `runs/<timestamp>-<slug>/` | one immutable run | see §7 |

A new workspace is born with generic INSTRUMENTS only — batteries, the neutral
corpus, dev prompts, the parser registry, judge-rubric and data templates, and
the dataset-generation prompts. It carries **no concepts**: `prompts/concepts/`
exists and is empty, and authoring or importing a concept is the first real
step. A worked example (stimuli, task prompts, a battery — a recipe, never
vectors or runs) ships separately as a sample workspace you open on purpose.
**None of the seeded content is study material — adapt it before any run you
intend to keep**, and every concept needs its own `validation.jsonl` or freeze
refuses (§4.4).

---

## 4. The lifecycle, in order

Every command takes `--json`. Use it (§5).

### 4.1 `workspace init`

```bash
steerlab-cli workspace init /abs/path/to/workspace
```

Creates directories, copies seed data, `git init`s, commits. Idempotence is
*not* offered: it refuses if the path is already a workspace.

### 4.2 `experiment create`

```bash
steerlab-cli experiment create <name> --model <model-id> \
  [--revision <commit>] [--description "…"]
```

`--model` is required. `--revision` pins the model commit; without it, freeze
demands one (or auto-pins from the local model cache). The manifest starts in
status **draft**, which is the only status any authoring verb accepts.

### 4.3 `experiment attach`

```bash
steerlab-cli experiment attach <name> <concept>… \
  [--method meanDifference|lat|emotionGrandMean|designatedReference] \
  [--pool-from K] [--reading-position '<label>'] \
  [--extraction-rendering '<json>'] \
  [--reference <concept>] [--corpus a,b,c]
```

`--reading-position` pins WHERE the residual stream is read, by name (`last
content token`, `content offset 2`, `mean content from token 0`, `offset from
end 3`, …). `--pool-from K` is the legacy spelling of one of them (`mean from
token K`); declaring both is refused, and a template-aware role declared under
a raw rendering is refused here rather than hours later on a GPU.

`--extraction-rendering` pins HOW those stimuli reach the model before they
are read: a JSON object (`{"mode": "raw"|"chatTemplate", "voice":
"user"|"assistant", "addGenerationPrompt": true|false}`), or the bare mode
word. Raw and chat-template renderings of the same stimuli give different
directions, so the rendering is recipe identity exactly as the reading
position is. **Absent ≡ raw ≡ today's bytes**, and an explicit `raw` — or an
explicit `voice: "user"` — canonicalizes away, so a silent legacy declaration
and a newly explicit one compare equal. `voice: "assistant"` renders the
stimulus as the model's OWN output, extracted by template subtraction;
`steerlab-cli` refuses that voice — a typed engine asymmetry, so run
assistant-voice cells on the server engine — and under it `addGenerationPrompt`
and `systemPrompt` reach nothing and are typed refusals at declaration time.
The template-aware reading positions above are the other side of this pairing:
under a raw rendering there is no template for them to find, which is the
refusal `attach` fires.

Pins each named concept's **current** stimulus hash plus its extraction
options. It also pins the neutral corpus when one exists — that corpus
denominates norm-unit α, so it is a pinned input, not a convenience.

Author the concept directory first. Both stimulus files must exist; if either
is missing, the refusal names *both* missing paths and the row shape in one
message. The row shape is:

```jsonl
{"text": "A sentence that expresses the concept."}
{"text": "Another one, same topic, same length, same register."}
```

`positive.jsonl` and `negative.jsonl` should be content-matched pair-for-pair:
the difference between the files should be the concept and nothing else.
Everything the extraction reads is these two files.

`--project-neutral K` exists and is **legacy, draft-only** — verified and
frozen manifests reject it. Do not use it.

`steerlab-cli experiment detach <name> <concept>…` is the inverse: it removes
each named concept's pin from a draft, all-or-nothing, and refuses
(`conceptInUse`) while any declaration still names one — an injection
condition's slot, a per-concept sweep-selection instrument, a variant
condition's `fromPromotion`, a perturbation policy. Remove or re-declare those
first. Re-pointing a draft at a different concept is `detach` then `attach`.

### 4.4 Author `validation.jsonl` — do this before you validate

`prompts/concepts/<name>/validation.jsonl` is the **held-out probe**: scenarios
that evoke the concept (or deliberately do not) *without using its
vocabulary*. It plays no role in extraction. It is the only evidence that the
extracted direction moves anything other than the words it was built from.

```jsonl
{"text": "A never-named scenario that should elicit the concept.", "expresses": true}
{"text": "A matched scenario that should not.", "expresses": false}
```

**Author it before `attach`.** `attach` pins this file's hash — or an explicit
"absent" when there is none — so a set that appears afterwards is a `verify()`
violation ("appeared after attach (pinned as absent) — re-attach to pin it")
that blocks `validate` and `freeze` alike. Working order: **author → attach →
validate → freeze**. Found out late: author the file, re-run `experiment attach
<name> <concept>…` for those concepts, then `validate`, then `freeze`. On a
frozen manifest there is no repair — duplicate first.

`validate` builds a per-concept **vacuity ledger**: every pinned concept owes a
scored held-out probe and is struck off only when one is actually scored. A
concept with no probe leaves the run **vacuous** — it exits 0 and looks
identical on the surface, but it carries the stamp, and `freeze` refuses it
under the `validateEvidence` gate, naming the missing file paths. So: no
`validation.jsonl` → `validate` "succeeds" → `freeze` refuses. In `--json` mode
`validate` reports `result.vacuous`, `result.vacuousConcepts[]`, and one
`vacuousValidation` advisory per concept. Read those, not the exit code.
Deleting the set after `validate` is a `verify()` pin violation, not a way out.

### 4.5 `pin-prompts` — the measured task

```bash
steerlab-cli experiment pin-prompts <name> prompts/tasks/<file>.jsonl
```

Pins `taskPromptsFile` + `taskPromptsHash` (SHA-256 of the raw bytes) and
parses the file with the run loop's own parser, so a file the run would refuse
is refused here instead of at generation time. Rows:

```jsonl
{"id": "item-01", "prompt": "…", "options": ["a", "b"], "target": "b"}
```

`options` + `target` are optional; when present the answer-token/logprob
instrument can score the item deterministically, which is the preferred
instrument for categorical outcomes — but the instrument is an explicit
declaration, never inferred from the items: run
**`set-instruments <name> answerTokenLogprob`** (draft-only) or the run
records prose and `parsedChoice` only. `pin-prompts` warns with a
`choiceItemsWithoutInstrument` advisory when items carry `options` and no
direct-scoring instrument is declared. Ids must be unique. `""` clears the
pin.

**`responseFormat` is optional, and absence is fine.** The instrument reads
any item whose `options` is non-empty; `target` is not consulted at dispatch
on either engine. The field only ever *subtracts*: an option-bearing item
that explicitly declares `"responseFormat": "json"` or `"freeText"` is
refused at run start under the `responseFormat` gate, and when the manifest
declares an `outcomeInstrumentScope` only rows whose declared format the
scope lists are measured — so in a mixed file `"label"` becomes required on
the rows you want scored, and only then. Do not add `"responseFormat":
"label"` to a file that already runs; nothing asks for it.

### 4.6 `pin-rubric` — the judging instrument

```bash
steerlab-cli experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md \
  --judges <name>:<kind>[:<model>[:<provider>]][,…]
```

Pins `judgeRubricFile` + `judgeRubricHash`, optionally replaces the judge
panel, and writes the explicit `evaluation` declaration the pin pair implies.
Kinds are `claude`, `local`, `openrouter`. A blank model field is *absent*, not
empty: a local judge then resolves to the study model at its pinned revision; a
`claude` judge to the default judge model. The fourth field pins a serving
provider and is only legal on `openrouter`.

The judge **name is a label, never a model id.** Declare **any number of
judges, including exactly one**: a single-coder design is a legal methodology
and freezes cleanly. What it costs is said rather than forbidden — a
`judgePanelTooSmall` advisory here and again at freeze, saying that no
inter-rater agreement statistics will exist for the study's codings, and the
coding report then records `fieldAgreement` as **absent with that reason**
rather than as an empty list. Zero judges is the state `judgeValidity` refuses:
a judged instrument with no judge codes nothing. Inline rubric text is
draft-only and cannot freeze — pin a file.

A panel of two or more must be **distinct**: identity resolves to (kind, model,
provider), so `--judges a:local,b:local` with both model fields blank resolves
twice to the study model at temperature 0 — one judge agreeing with itself by
construction — and refuses at freeze under `judgeValidity`. Vary the kind, the
model, or the provider.

A **local judge naming a model other than the study model** must pin the exact
bytes that will judge — `judges[].revision` and `judges[].dtype` — or freeze
refuses under `judgeValidity`. Declare them with `--judge-pin`, repeated per
judge and keyed by judge name:

```bash
steerlab-cli experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md \
  --judges strict:local:google/gemma-3-27b-it,lenient:claude \
  --judge-pin strict=<commit-hash>:bfloat16
```

Dtypes are `bfloat16`, `float16`, `float32` (aliases `bf16`/`fp16`/`fp32`,
stored canonically). The revision must be a commit hash: a branch or tag is
re-pointed by definition, so it cannot identify the weights a run used, and
that is refused at the declaration rather than at freeze. A pin naming no
declared judge, or a pin on a `claude`/`openrouter` judge (which carry no
revision or dtype), is a malformed invocation — never silently dropped.

`--judges` replaces the ROSTER, but the pins merge field by field beneath it: a
judge whose name survives with the same kind and model **keeps** the revision
and dtype it had, a judge whose model changed **drops** them (they identify the
old bytes), and either way the echo says which — under
`result.inheritedFromExistingDeclaration`, the same key the sweep-selection
merge uses. Before this, `pin-rubric --judges` wiped pins the app had written
and the study then refused at freeze for want of pins it used to have.

### 4.7 `declare-condition` — the arm

```bash
steerlab-cli experiment declare-condition <name> <condition> \
  --slots <concept>:<layer>:<alpha>[:add|ablate][,…] \
  [--band-width K] [--alpha-units norm|raw] \
  [--control randomMatchedNorm|randomDirectionAblation]

steerlab-cli experiment declare-condition <name> <condition> --baseline
```

**Without at least one condition, a concept study runs the implicit baseline
alone and measures nothing.** A multi-slot condition *is* the linear mix
`h + Σ αᵢ·vᵢ` and hashes as a single condition. Every named concept must
already be attached. `--baseline` and `--slots` are exclusive.

`--alpha-units norm` (the default) denominates α by the residual-stream norm at
that layer on the pinned neutral corpus — that is what makes α comparable
across concepts. Use `raw` only when you know why.

`--control` substitutes a deterministic random direction into the same slots,
giving you the matched-norm control arm.

### 4.8 `validate`

```bash
steerlab-cli experiment validate <name>
```

Extracts (or reuses) the vectors and scores the held-out probes; writes a run
directory. This is the evidence `freeze` looks for, and it must match the
manifest's *exact* pins — model + revision, concepts and their options, neutral
corpus, and the run substrate. Change any pin and the evidence stops matching;
re-validate.

Those pins *are* the evidence's key; the experiment's name is not among them,
so evidence is shared across the workspace — a `duplicate`, or any fresh
experiment with matching model, revision, concepts, options and neutral corpus,
freezes on validation it never ran. A passing `validateEvidence` gate is not by
itself proof that *this* experiment produced the evidence.

`extract` runs the derivation alone if you want it separately. Both load the
model.

### 4.9 `freeze` — one-way

```bash
steerlab-cli experiment freeze <name> [--force] [--run-substrate local|server]
```

Freeze verifies every pin, stamps the manifest's content hash and the
workspace git commit, snapshots every pinned input into
`experiments/<name>/pinned/` — concept stimulus directories, task prompts,
judge rubric, capability battery, reasoning-style taxonomy, human tables, and
the neutral corpus that denominates norm-unit α — writes the generated
settings summary beside the manifest (`experiments/<name>/preregistration.md`
when that path is free or holds a file that is *provably* a previous freeze's
own untouched output; a researcher-authored preregistration there is preserved
untouched, frozen — its SHA-256 is stamped into the manifest, it is
snapshotted into `pinned/`, `verify` re-hashes it from then on, and it travels
in run bundles — and the generated summary lands as
`preregistration-frozen-settings.md` instead), and makes the manifest
read-only. There is no unfreeze. **Iterate by `experiment duplicate <name>
<new-name>`, never by editing.**

A file at `preregistration.md` counts as the freeze's own only when the
manifest's stamped hash of what it last generated matches the bytes exactly,
or — for files predating those stamps — when the first line is the generated
header and the marker line `*Generated at freeze; do not edit.` is the last
non-empty line. Quoting that marker anywhere else in your own document is
safe: when in doubt the file is preserved, never overwritten.

The snapshot is taken at freeze time only: it is the no-git reproducibility
floor, not a live mirror. A study frozen before a pin joined the snapshot
keeps whatever its own freeze wrote.

`--run-substrate server` matches the evidence gates against evidence produced
on the *server* engine — the substrate the measured runs will execute on —
instead of this one.

Two classes of check, and the difference matters:

- **`verify()` pin integrity** — always runs, **never skippable**, owns no gate
  id. Drift in any pinned file's bytes is a violation here. `--force` does not
  reach it.
- **The seven gates below** — force-skippable, each with a stable id.

#### The seven freeze gates

| Gate id | What it demands | Repair |
|---|---|---|
| `revision` | a pinned, immutable model commit — not absent, not symbolic | `create --revision <commit>`, load the model once so it is cached, or pin the commit the symbolic ref resolves to |
| `measurementPins` | pins that determine *what is measured* are present and valid (e.g. a loadable study dtype) | repoint the invalid pin at a loadable value |
| `validateEvidence` | a `validate` run matching the exact pins on the run substrate, **and** that evidence is not vacuous (§4.4) | author the named `validation.jsonl` files, **re-attach** their concepts, then `experiment validate <name>` (§4.4; the refusal's own `repairAction` names this full sequence) |
| `variantValidity` | attached variants carry hashed adapter weights and a pinnable dataset manifest | re-save the variant with hashed weights and re-attach it |
| `batteryEvidence` | each variant condition has scope-matched capability-battery evidence | re-run `experiment validate <name>` (each variant condition runs the pinned battery) |
| `judgeValidity` | a rubric **file** and at least one judge the pipeline can actually run (a panel of two or more must be distinct) | `experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md --judges a:local[,b:claude]` |
| `gitClean` | every pinned input is committed in the workspace git repo | commit the pinned inputs (freeze auto-commits in most workspaces; this gate speaks when it could not) |

In `--json` mode a refusal gives you `error.gate` (the gate whose message is in
`error.reason`), `error.gates[]` (**every** gate that failed, not just the
first), and `error.repairAction`. Fix the gate; do not retry the same command.

**`validateEvidence` is keyed by PINS, not by the experiment's name — so a
duplicate inherits its donor's evidence.** The gate matches on a validation
scope hash built from the model id and pinned revision, the attached concepts
with their pins, the neutral-corpus hash, the grand-mean corpus, the
capability-battery hash (when variant conditions exist), and any declared
validation depths. Conditions, sampling settings and the name are deliberately
outside it. Practically: **do not re-run `validate` on a duplicate that
changed only measurement-side declarations** — a new rubric, a different judge
panel, new exclusion rules — because the donor's evidence already satisfies
the gate and re-validating spends GPU time for nothing. Change something the
scope covers (re-attach a concept, re-pin the revision, point at a different
battery) and the inheritance correctly stops; then validate.

**`--force` skips the seven gates and stamps the manifest** `freezeForced: true`
plus `forcedGatesSkipped: [<gate ids>]`, and emits one `freezeGateSkipped`
advisory per gate. A forced freeze is permanently non-citable — but checkably
so, by stamp. **Do not `--force` to get past a gate.** If a human explicitly
asks for it, do it and report exactly which gates were skipped.

### 4.10 `run`

```bash
steerlab-cli experiment run <name> [--prompts prompts/tasks/<file>.jsonl]
```

Generates under every declared condition and writes an immutable run directory
containing the manifest snapshot + content hash, `generations.jsonl`,
`battery.jsonl`, computed metrics, and a canonical `config.json`.

`run` refuses, before the model loads, a concept-bearing manifest with no
injection, variant, or SAE arm. That refusal is not an obstacle: it is the
firewall telling you the study would have measured nothing. Declare a condition
(§4.7) or promote an agent (§4.12), or declare the baseline-only study
explicitly if that is genuinely what you want.

### 4.11 `analyze`

```bash
steerlab-cli experiment analyze <name> [--allow-unverified-epoch]
```

Pure CPU, no model load. Paired-to-baseline effect sizes — bootstrap CIs and
Wilcoxon — over the newest completed run; writes `effect-sizes.csv` and folds
`effectSizes` into `report.json`.

Guarded by the **epoch guard**: the run's stamped experiment hash must equal
the live manifest's content hash, or the verb refuses. `--allow-unverified-epoch`
bypasses only *unstamped legacy* runs and stamps `epochUnverified` on the
result. The guard is per-engine — analyze a run on the engine that produced it.

Zero effect-size entries is reported as an `emptyAnalysis` advisory, not a
failure. It means the source run had no non-baseline condition. Check for it.

### 4.12 `evaluate`, `sweep`, `promote`, `confirm`

**`evaluate <name> [--run <dir>] [--allow-unverified-epoch]
[--sample-per-condition <n> --sample-seed <hex-or-int>]`** — paired-judge
evaluation of a completed run through the manifest's pinned rubric and judges,
writing a new evaluation directory beside the source run, which is never
mutated. Same epoch guard as `analyze`; defaults to the newest completed run.

**To code a preregistered SUBSAMPLE rather than the whole run**, pass
`--sample-per-condition <n>` together with `--sample-seed <hex-or-int>`. Both
or neither: a sample with no seed is one nobody can redraw, a seed with no
size is a stamp on a coding it did not shape, and either half alone refuses
at 64. The draw is stratified — within each condition, `floor(n / P)` records
per promptID with the remainder handed out in seeded order, and records inside
each cell chosen over `sampleIndex` — and it is the same draw on both engines
for the same seed. An `n` above a condition's population REFUSES; it never
clamps, because a clamped design is a different design than the one that was
preregistered. Per-response coding only: a paired rubric refuses, since a pair
is not a record. The result is stamped loudly — a `sampling` block in
`coding-report.json` and in the run's `config.json` carrying
`samplePerCondition`, `sampleSeed`, `sampledRecords`, `sourceRecords` and the
derivation `rule`, and every human line reading `coded N of M (seeded
subsample)`. **No `sampling` block means the full corpus was coded**; never
report a sampled coding as a census.

Under a per-response coding rubric it writes `coding-report.json`. Read its
`fieldAgreement` entries before the aggregates: each categorical entry carries
`percentAgreement`, `kappa`, and a `confusion` block where `confusion[a][b]`
is how many shared cells judgeA coded `a` while judgeB coded `b`, summing to
the entry's `n` — so you can say WHERE two coders part ways without
re-deriving anything. A single-coder run has **no** `fieldAgreement` key at
all; it carries `fieldAgreementAbsentReason` instead. Do not report that as
"agreement was measured and was zero" — report it as what it says.

**To re-measure an existing run with a NEW instrument, duplicate — never edit
the source study.** The epoch guard tolerates exactly the drift that cannot
have moved a byte of the source run's generations: `judges`, `evaluation`,
`pipeline`, `judgeRubricFile`, `judgeRubricHash`, `humanValidation`, and the
study's own `name` (identity, not a measurement setting — and the one field a
duplication must change). So the sanctioned path is:

```bash
steerlab-cli experiment duplicate <name> <name>-recoded
steerlab-cli experiment pin-rubric <name>-recoded prompts/rubrics/<new>.md \
  --judges a:local:<judge-model>,b:claude --judge-pin a=<commit-hash>:bfloat16
steerlab-cli experiment evaluate <name>-recoded --run runs/<original-run-dir>
```

The original run directory is read, never mutated; the evaluation writes
beside it as always. The tolerated fields are named in the output's
`measurementDrift` stamp with a warning on stderr, so a re-measurement is
never mistaken for the original measurement. Change any generation-side pin —
model, concepts, task prompts, sampling protocol — and the guard refuses, as
it should: those runs would have been different. **`promote` tolerates
nothing** and still refuses a renamed or re-judged manifest, because a
promotion binds a judged sweep's evidence.

**`sweep <name>`** — walks layer × α on a dev split and records a
recommendation per concept, selecting by the manifest's declared criterion
(`sweep.selection`: an objective, capability/coherence constraints, an optional
matched-norm-random control margin). Objectives are `markerDensity`,
`judgeScore`, `logprobShift`. Marker density measures surface vocabulary — it
is a manipulation check, not a selection objective for any study whose outcome
is a decision rather than prose. Declare the rule headlessly with
**`set-sweep-selection <name> --objective judgeScore|logprobShift|markerDensity …`**
(draft-only); with no declared rule the sweep defaults to `markerDensity` and
says so with a `sweepSelectionDefaulted` advisory — on a choice-task prompt
set, treat that advisory as a stop sign. The sweep's `--json` result carries
the run directory and each concept's winning cell, criterion, and metrics;
on a frozen manifest it records recommendations only
(`sweepRecommendationsOnly` advisory). Loads the model.

**`promote <name> <concept> [--agent-name N] [--cell L:α --reason "why"]`** —
mints a variant artifact (an "agent") from the sweep-selected cell with a
`promotion` birth certificate (`promotedBy: "criterion"`). `--cell` is the loud
manual override: it stamps `promotedBy: "manualOverride"`, warns, and still
**requires evidence that a sweep ran for the concept** — promotion with no
sweep at all is refused. Hand-created variants stay legal but surface as freeze
advisories. Pure CPU.

**`confirm <name> --agent <A> [--deltas 0.2,0.5] [--no-control]`** — declares a
perturbation policy around a promoted agent's anchor cell, which expands
mechanically into ordinary hashed conditions on the draft manifest. Pure CPU.

**`data check <name>`**, at any point, returns the full classified readiness
list: every requirement, its status, the **path you must author**, and the
rationale. Fastest way to find what is missing. Blockers are a refusal:
`state: "refused"`, exit **65 in both modes** (the one verb whose human exit
has migrated — §5).

### 4.13 The rest of the surface, and how to find it


**`--help` is how you discover the surface**, at three levels, on both CLIs:

```bash
steerlab-cli --help                      # every family
steerlab-cli experiment --help           # that family's verbs, one line each
steerlab-cli experiment attach --help    # one verb's positionals and flags
```

It is a declared flag on **every** verb, it runs **nothing**, and it exits 0
in both modes — so it is always safe to ask, including on a verb that would
otherwise write a manifest. With `--json` the same page comes back as data in
`result`, so you never have to parse the columns. Each page ends with the
exit-code line, and a verb page names every flag it accepts with its purpose
and its argument's shape, including closed vocabularies (`set-instruments`
prints the legal instrument names).

Secondary: running a family with no verb (`steerlab-cli experiment`) refuses
and lists every verb it accepts. That roster answers "which verbs exist";
`--help` answers "and what do they take", so reach for `--help` first.

**`list`** — every experiment here with its status, model, concepts, condition
count and `freezeHash`. The cheapest orientation command; run it first.

**`verify <name>`** — re-hashes every pinned input against the manifest and
refuses (`pinDrift`, one line per drifted pin) if a byte moved. What the other
verbs run for you, available alone, safe at any status.

**`set-sampling <name> [--temperature <t>] [--max-tokens <n>]
[--prompt-mode chatAssistant|rawCompletion] [--samples-per-item <n>]
[--seed-policy manifestSeeds|derivedSHA256]`** — declares the generation
protocol on a draft, merge-style: only the flags given move, `""` clears
`--prompt-mode`/`--seed-policy`, and `--samples-per-item 1` clears to the
deterministic default. A stochastic replication arm is `--samples-per-item 25
--temperature 0.7 --max-tokens 1024 --seed-policy derivedSHA256` — legal to
declare locally; §7's greedy-only rule then routes the run to the server
engine. The joint rules (`samplesPerItem > 1` needs `temperature > 0` and
seedPolicy `derivedSHA256`) surface at `verify`, not here, so the fields can
be declared one flag at a time.

**`set-exclusions <name> <rule>[,…] [--endpoint <key>] [--min <x>]
[--max <x>]`** — declares the record-exclusion rules analysis applies
(`failedAttentionCheck`, `unparseableEndpoint`, `outOfRange`) on a draft;
`""` clears the declaration. `--min`/`--max` bound the `outOfRange`
keep-window and `--endpoint` names the parsed-value key the endpoint rules
read (default `parsedMonths`). Exclusions apply at analysis time only and
are stamped honestly — records never leave `generations.jsonl`.

**`set-system-prompt <name> "<text>"`** — declares the study's system prompt
on a draft: the deployment frame every arm is read under. `""` clears it. The
text is stored inline (there is no file and no hash beside it), and what the
model receives is capability-dependent, decided by the renderer: a family
whose chat template has a system role gets a **genuine system turn**; a family
without one — Gemma — gets the **same text prepended to the first user turn**
(`system + "\n\n" + user`); `rawCompletion` prepends it to the prompt text.
Every route delivers it, so there is no prompt-mode gate. An arm carrying an
agent persona reads under *persona*, blank line, *this frame* — declaring one
never displaces an agent's identity. The one place a declared frame does not
apply is a pinned item whose scripted transcript opens with its own `system`
turn, which replaces it for that item; the verb counts those and says so
through the `systemPromptNotApplied` advisory rather than letting the
substitution be silent.

**`set-parser <name> <parser>`** — declares the numeric-endpoint parser from
the workspace registry (`prompts/parsers/parser-registry.json`) on a draft and
pins that registry's SHA-256 as `parserRegistryHash`; `""` clears both. The
hash is never an argument: the registry file is the authority on which parser
VERSION the study preregistered, so re-declaring the same name is also the
drift repair. Without a declaration a numeric study falls back to the
DEPRECATED implicit selection (`caseFamily: "sentencing"` → the built-in
duration parser), which every firing site now announces.

**`set-instrument-scope <name> <responseFormat>[,…]`** — declares which
response formats (`label`, `json`, `freeText`) the option-consuming
instruments apply to on a draft, pinning the row set they select
(`itemCount` + `itemIDsHash` computed from the study's own task prompts);
`""` clears the declaration. This is the NON-LOSSY repair the run-start
`responseFormat` refusal names: a mixed json+label file keeps
`answerTokenLogprob`/`ordinalScale` on its label rows instead of dropping the
instrument for `sampledText`. Declaring formats no pinned item carries is
refused — a scope selecting zero rows would produce zero records.

**`set-evaluation-sampling <name> <n> <seed>`** — declares the study's
EVALUATION SAMPLING DESIGN on a draft: how many records per condition the
judged coding preregistered, and the seed that draws them
(`evaluationSampling`); `<name> ""` clears it. Both halves or neither. The
draw RULE is the third field and is derived from the engine at the write —
never an argument, for the same reason `parserRegistryHash` is not one.

Declare it rather than typing the flags. A stamp records what HAPPENED;
"preregistered" is a claim about what was decided BEFORE anything ran, and
only the declaration puts that claim in the artifact chain — every run writes
the manifest snapshot into its own `experiment.json`, so the design travels
with the evidence. `evaluate` then draws it with no flags at all;
`--sample-per-condition`/`--sample-seed` may still be typed and become a
CROSS-CHECK, refusing on any inequality rather than overriding. Declaring is
measurement-side, so it never invalidates the run being coded — which is what
lets you duplicate a frozen study, declare the coding design on the duplicate,
and evaluate against the original's run. What the desk checks is a whole `n`
of at least 1, a seed that parses, and a rule this build derives; the
POPULATION check stays at `evaluate`, because at declaration time the source
run need not exist yet.

**`set-style-taxonomy <name> prompts/taxonomies/<file>.json`** — pins a
reasoning-style taxonomy (path + hash) on a draft. No pin, no reasoning-style
scoring; drift after pinning is a verify violation like any other.

**`rescore-style <name> [--run <dir>]`** — recomputes reasoning-style features
for a completed run through that taxonomy into a **new** run directory, never
touching the source. Epoch-guarded like `analyze`. Pure CPU.

**`vectors mirror-poles <runDir/name> --concept <newName>`** — mints the
**opposite pole** of a contrastive direction as an artifact of its own,
instead of leaving it as a negative α that every dose surface reads as "less
of the concept" and no artifact records. A bit-exact sign flip on the
`layer_<i>` tensors (never on `neutral_mean_layer_<i>`), under a **required
new concept name**, stamped `negatedFrom` + `polesSwappedFromSource`. Restricted
to paired, source-concept-bearing contrasts — the CAA family — because those
are the methods whose two poles ARE two authored stimulus files, so swapping
their roles is exactly what the negation means; every other method is refused
`unmirrorableMethod`, pointing at the negative α that *is* available. No model
is loaded. Three consequences you will meet downstream: the mirrored concept's
own directory must hold the source's two files with their roles swapped, and
`attach` proves it by hashing them **in the source's order**; `promote`
matches the recipe identity against the pin's `sourceStimulusSetHash`, the
hash the artifact actually stamps; and the birth certificate carries a
`poleProvenance` block, so an arm injecting a negated direction never looks
like one injecting the concept. Author the mirrored concept's
`validation.jsonl` yourself — the verb writes nothing into `prompts/`, because
an engine that invented held-out evidence would be manufacturing the thing
`validate` exists to demand.

### 4.14 Multi-agent studies: casting a panel

A **panel** is a scenario under `prompts/panels/` — roles, turns, visibility,
case materials. A *semantic* panel binds no model to any seat, which makes it
deliberately unrunnable: it has to be **cast** first, and casting is the step
that binds the study's model and sampling settings to one seat assignment.

```bash
steerlab-cli panel list
steerlab-cli panel check <path-or-name>
steerlab-cli panel compile <path-or-name> --experiment <name> \
  [--seat <seat>=<agent-artifact-path>]… \
  [--model <id>] [--temperature <t>] [--max-tokens <n>] [--file-slug <slug>]
```

`compile` writes the bound scenario to `prompts/panels/compiled/` and pins it
into the **draft** in one step — both the scenario pins and the provenance pair
recording which semantic panel it came from. It also declares the study
multi-agent, because a panel scenario is read only by that run path.

**Seats are keyed by the scenario's agent `id`**, not its display name — `list`
reports the ids, and an id the panel does not have refuses with the list of the
ones it has. Seats you do not name stay **baseline**, and an all-baseline
casting is the control composition, not an absence. A `--seat` value is an
agent artifact path under `runs/model-variants/`; its hash is read from the
file.

`--model`, `--temperature` and `--max-tokens` **default from the manifest** and,
when given, are written to it before the compile: the manifest stays the one
place those three are decided. `check` validates a *bound* panel and reports
its advisories; a semantic panel fails that check by design, and `compile` is
the answer.

### 4.15 The sweep workflow

A sweep is where a study stops being a design and starts costing money: it
generates at every layer × α cell, for every concept, and then a declared rule
picks one cell. Four things go wrong here often enough to be written down.

**(a) Swapping the concept into an inherited sweep.** `duplicate` is how a
frozen study is iterated, and it copies the whole sweep block — grid, selection
rule, instrument pins — along with the donor's **concepts**. Those concepts ride
along swept but uncited: nothing in the new study declares them, and a direction
selected for one of them cannot be cited by the study that swept it. Do the swap
in this order, and let `conceptInUse` order you:

```bash
steerlab-cli experiment duplicate <donor> <new>
steerlab-cli experiment attach <new> <concept>
# author prompts/tasks/<concept>-choices.jsonl (§4.15d), then:
steerlab-cli experiment set-sweep-selection <new> --objective logprobShift \
  --choice-prompts prompts/tasks/<concept>-choices.jsonl
steerlab-cli experiment detach <new> <donor-concept>…
```

Detach **last**. `set-sweep-selection` MERGES its axes, so re-declaring the
objective and its instrument here keeps the donor's capability tolerance,
coherence rule and matched-norm control, and the success line names whatever it
carried over (only `--objective ""` clears the block). The inherited selection
rule still names the donor's concepts
in its per-concept instrument map, and `detach` refuses (`conceptInUse`) while
it does — that refusal is the ordering, enforced. Never hand-edit
`experiment.json` to break the cycle; the refusal is telling you a declaration
would be left dangling, and the manifest is not where you resolve that.

**(b) The grid dialog.** The grid is a cost and a preregistration, so a human
decides it — but the human has to be shown what they are deciding.

*Inherited grid* (a duplicate): **show it before you touch it** — the depth
fractions the manifest stores AND the absolute layers they resolve to at this
model's depth — and ask whether to keep it. `set-sweep-grid`'s `--json` result
carries both (`layerFractions`, `resolvedLayers`, `layerCount`, `cellCount`),
and so does `experiment list`'s manifest. A grid inherited from a study on a
26-block model names different blocks on a 62-block one; that is the fractions
working, and it is still a change the human should see.

*No grid* (de novo): **propose one and say where the proposal comes from.** The
engine default is `0.5,0.7,0.85 × 0.05,0.08,0.1,0.13` — depth fractions and
residual-norm α, recalibrated on live testing because stronger α routinely
buys incoherence and the useful cells sit late in the network. That is the
provenance; say so, say it is a starting grid and not a finding, and ask.

Then write the answer:

```bash
steerlab-cli experiment set-sweep-grid <name> \
  --layer-fractions 0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13
```

Layers may be named absolutely (`--layers 13,18,28`) when something has already
been extracted for the model — that is what states its depth. Both axes must
ascend with no repeats, α is in residual-norm units above 0, and `0` is the
baseline cell every sweep runs anyway. `set-sweep-selection` owns the RULE;
this owns the grid, and typing one verb's flag at the other answers with a
pointer.

**(c) The de novo path, in order.** Nothing here is skippable and each step
refuses if the one before it did not happen:

```bash
steerlab-cli experiment create <name> --model <id> --revision <commit>
steerlab-cli experiment attach <name> <concept>          # after §4.15d
steerlab-cli experiment extract <name>
steerlab-cli experiment validate <name>                  # the held-out probe
# the grid dialog (b), then set-sweep-grid, then:
steerlab-cli experiment set-sweep-selection <name> --objective logprobShift …
steerlab-cli experiment sweep <name>
steerlab-cli experiment promote <name> <concept>
```

`validate` before `sweep`, always: sweeping a direction that scores at chance
on its own probe buys a confident setting for a vector that measures nothing.
And treat `promote` as two steps — read the sweep's recommendation, show the
human the winning cell and its margin over the control, and promote only after
they say so. It mints an artifact with a birth certificate; that certificate is
a claim, and a human should have made it.

**(d) The missing-data rule.** Most of the work above is blocked by data that
does not exist yet. For every missing prerequisite, in this order:

1. **Name it** — the exact path, from `steerlab-cli data check <name>`, which
   classifies every requirement and names the file you must author.
2. **State what it ought to be** — the row shape, the counts, and what the file
   has to be independent of. Not "some validation rows": *held-out, labelled,
   and using neither pole's vocabulary, because the extraction corpus is full
   of that vocabulary and a probe that reuses it tests a word detector.*
3. **Emit its generation prompt** —
   `steerlab-cli authoring prompt <kind>` renders the prompt for that kind of
   data with your study's seam substituted, its audit battery as NUMBERS, and
   two hashes stamped in the header: `promptSpecHash`, over the template and
   partials, which recovers the exact WORDING later, and
   `promptInstanceHash`, over the rendered body and the resolved arguments,
   which recovers WHICH EMISSION produced a given corpus. Two prompts for two
   concepts share the first and differ in the second. Kinds:
   `contrastive-pairs`, `choice-prompts`, `validation-set`, `reader-pairs`,
   `battery`. Hand it to an author unedited.
4. **Never install generated data on the generator's word.** The emitter is not
   the acceptor. A *second* reviewer — one who did not write the rows — re-runs
   the prompt's own audit battery against the delivery and reports the numbers.
   Only then does the file land in the workspace, and only then is it pinned.

That fourth step is the one that gets skipped, and it is the one that matters.
An author asked to audit their own output reports a pass; the numbers are cheap
to compute and expensive to fake, which is why they are numbers. A corpus that
was installed unaudited is not repairable later — it is pinned, frozen, and
cited.

---

## 5. The machine contract

**Pass `--json` on every command.** In JSON mode:

- **Exactly one JSON document on stdout.** Every diagnostic, progress line,
  warning, and human report goes to stderr. No ANSI.
- Keys are sorted, dates are ISO-8601, there is exactly one trailing newline.
- `--json` is honored even when argument parsing itself fails.
- `--out <path>` also writes the document to a file. (`--json <path>` is the
  deprecated spelling on the one verb that had it; it warns on stderr.)
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

Two live caveats. **These codes are the `--json`-mode codes.** Without
`--json`, most failures still exit `1`; two verbs differ, and both differ in
both modes — `data check` exits `65` for blockers, and `vectors compare`
exits `1` when it compared and diverged but `2` when it could not compare at
all. One more reason to always pass `--json`. And an undeclared flag is exit
`64` in **both** modes, before the verb does any work: flags are parsed
strictly against a per-verb table, so a typo cannot silently change what a
study means.

Refusals are typed everywhere, not only at freeze: a lifecycle refusal
carries `error.code == error.gate` from a second closed vocabulary
(`statusImmutable`, `pinDrift`, `missingPrerequisite`,
`promotionEvidence`, …), while freeze refusals keep
`code: "freezeGateFailed"` with the gate id in `error.gate`. Either way,
`error.repairAction` is an executable command sequence — run it, then retry.

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
---

## 6. Immutability

- **`runs/` is append-only.** Never edit, overwrite, or delete a run directory.
  A run carries enough to rebuild its tables without rerunning the model, and
  that is only true if nobody touches it. `runs/` is gitignored.
- **Three subtrees under `runs/` are deliberately mutable libraries**:
  `runs/model-variants/`, `runs/neutral-pcs/`, `runs/jlens-lenses/`. A promoted
  agent's artifact is editable in place there; frozen studies are protected by
  the manifest snapshot and the artifact hash, not by the directory.
- **Frozen manifests are read-only** — the verbs that WRITE the manifest
  (`attach`, `detach`, `pin-prompts`, `pin-rubric`, `declare-condition`,
  `set-style-taxonomy`, `confirm`) refuse on a frozen or complete one;
  `duplicate`, then edit the copy. The verbs that only read it stay legal,
  including two that surprise people: **`sweep`**, which records its
  recommendations in its own run directory rather than in the manifest, and
  **`promote`**, which mints its agent into the mutable `runs/model-variants/`
  library. Both are confirmation-stage gestures a frozen study must still be
  able to make, and neither can change what the study means.
- **`experiments/<name>/pinned/`** is the freeze-time snapshot of every pinned
  input — the reproducibility floor when git is unavailable. Do not edit it.
- Do not hand-edit `experiment.json`. Its bytes *are* the content hash; an edit
  that bypasses the verbs surfaces as a verify violation, which is the good
  outcome, or as a silently different study, which is not.

---

## 7. Which engine runs what

Two compute engines read and write the same artifacts. Measured runs on the
**local** engine are **greedy only**: the study runner requires
`temperature == 0` and rejects more than one seed, because the local generator
cannot pin a per-run sampling seed. `seeds` is recorded for provenance and does
not affect local generation — every local generation record stamps
`seedInert: true`. A manifest needing `temperature > 0` or
`samplesPerItem > 1` belongs on the **server** engine, which seeds per record
and writes one record per (condition, prompt, sampleIndex). For categorical
outcomes prefer the answer-token/logprob instrument over sampled prose on
either engine: deterministic and temperature-free.

Activations do not transfer between engines. Vectors must be **re-extracted and
re-validated on whichever engine a study runs on** — the parity claim is
structural, not byte identity. Stimulus and corpus SHA-256 hashes *are*
identical across engines, which is what makes cross-engine comparison possible
at all. `freeze --run-substrate` and the evidence gates enforce this: evidence
from the other engine will not satisfy them.

**Authoring is CLIENT-authority by design.** The authoring client's
workspace is the source of truth; the engine on compute hardware never
authors, and its workspace is a cache. The boundary is client versus running
hardware — not macOS versus everything else. `create`, `attach`, the
`pin-*`/`declare-*`/`set-*` verbs, `panel compile`, and `freeze` run on the
local CLI;
execution and analysis (`run`, `evaluate`, `analyze`, `sweep`) answer
identically-shaped envelopes on either engine, and the epoch guard keeps
`analyze` on the engine that produced the run. A server-side refusal whose
repair is an authoring act names the local verb on purpose — go author
there, then submit. Some verbs exist on one engine only; asking the server
for one of them is refused with `error.code: "macAuthorityVerb"` (no
`error.gate` — it describes the engine, not the study) and an
`error.repairAction` spelling the local command. **That code's name is
historical**: it is a stable machine code agents switch on, and it means
"this engine executes, it does not author", never "author on a Mac". Do not
emulate the verb; run the repair where it belongs.

**One verb runs on the compute engine ALONE and has no client spelling at
all.** `steerlab-server battery run <battery-file> --agent <ref>…` reads a
capability battery against one or more agents — `baseline`, a condition spec
`<concept>:<layer>:<alpha>`, or a promoted agent artifact — and writes its own
pinned run directory holding `battery.jsonl` and `battery-report.json`. It
loads models, so it is execution and cannot live on an authoring client; and
unlike every other model-loading verb it is not manifest-shaped, so no bundle
route carries it either. If you need a floor reading, take it where the models
are, then cite the report by its pins.

This is the **floor battery**, and it is a different artifact from the battery
a study pins. The pinned one (`batteryEvidence`, §4) is a per-condition control
inside a study's own run matrix. The floor one precedes any study: it asks
whether an agent is a working model at all, under a charter that is ex ante,
study-blind and fixed. Two consequences you will meet:

- **A floor battery declares `batteryFormat: 3` and CANNOT be pinned.** It
  carries a second operating regime — long-form generation at a positive
  temperature, several samples per item, read for generation health rather
  than graded — and scored per condition inside a run matrix that would be a
  second outcome measure wearing a control's name. Pin a `batteryFormat: 2`
  battery; run a `batteryFormat: 3` one.
- **Take a floor reading before you compare arms of different kinds.** Before
  claiming a prompted persona and an injected agent differ in behaviour, show
  they are capability-equivalent — otherwise the difference you measured is
  competence, and no analysis afterwards can separate the two. One
  `battery run` naming each arm as an `--agent` does it, and the report is
  keyed by pins so a later study can cite the same reading.

`steerlab-server battery generation-prompt` states the charter in full and is
what you hand an author who is drafting one.

---

## 8. Remote execution, and where the depth is

When this workspace's study runs on a cluster, the same contract holds — and
one rule outranks convenience: **never write a bare sbatch script.** Submit
through the engine's rendered path (`steerlab-server study submit …`, or the
app), which requests node-scratch via the site's gres and arms the cleanup
trap; a hand-rolled script silently gets neither, and stale node-scratch is
how clusters come to email their operators about you. If a submission need
seems to force a hand-roll (a dependency chain, a resume), that is a missing
verb to report, not a reason to bypass the renderer.

The cluster lifecycle has first-class verbs — prefer them to raw `ssh`:
`steerlab-cli cluster push` (deploys the engine AND re-stamps its build
identity — read §8.1 before you reach for it), `cluster ensure`, `cluster
tunnel open`, `cluster remote --site <id> …`, and `cluster import --site <id>`
(verified, never-purging run import). Site profiles live in the SteerLab
home's `Sites/cluster-sites/` registry — never invent one; ask the researcher
for theirs.

**Sharding a long run across GPUs, and the two ways it bites.** A measured
`run` partitions cleanly (every generation record is independent), so a Slurm
submission can fan out across N sibling jobs whose partials the server merges
back into one ordinary run directory:

```bash
steerlab-cli remote submit-bundle <server-bundle-path> --site <id> \
  --verb run --executor slurm --parallel 4 --json
```

The value goes on the wire only when `n > 1`, the executor is `slurm`, and the
verb shards (`run`, or a pipeline whose declared chain starts with `run`). The
envelope tells you which happened — `parallelJobsRequested`,
`parallelJobsEncoded`, `parallelJobsSuppressedBecause` — so **read the echo**
rather than assuming the request was honored; a suppressed one also warns on
stderr. Sharding is execution logistics and never enters the manifest or its
content hash, so a sharded run and a single-job run of the same frozen study
are the same measurement.

Two rules that are not optional. **Verify the shard jobs actually landed:** a
fan-out can PARTIALLY FAIL while the submit still exits **0**, because the
abort is reported through the parent job record rather than the process's exit
status — so check `steerlab-cli remote jobs` (or the scheduler queue at the
site) and count them, and never report a sharded submit as successful on the
exit code alone. **And stagger submissions where the site caps queued jobs per
user:** K shards are K independent scheduler submissions, so a fan-out that
crosses the cap has its later shards refused while the earlier ones run. The
site profile's `maxParallelGPUJobs` records that limit when the researcher has
declared one; ask rather than guess. The merge is performed by a **running**
`steerlab-server serve`, not by the submitting process.

The shared SSH master EXPIRES — routinely, daily. A `Permission denied
(publickey,keyboard-interactive)` from an otherwise-working site means
expired authentication, not a broken site or profile: run
`steerlab-cli cluster auth open --site <id>`, which spawns a Terminal
window for the HUMAN's password and multi-factor prompt and then persists
the master for hours; `cluster auth status` confirms, `cluster auth close`
ends it. You cannot answer that prompt yourself — say so and wait for the
researcher rather than retrying commands that can only refuse.

### 8.1 Updating the server engine

**`cluster push` deploys the payload baked into the app bundle**
(`SteerLab.app/Contents/Resources/ClusterPayload/`), **not your checkout.**
Pulling new commits and pushing deploys nothing — and every observation agrees
that this is fine: the push succeeds, and `cluster status` reports `payload:
current`, truthfully, because the app's payload and the cluster's really do
match. Build first:

```bash
scripts/build-app.sh --install --force        # regenerates ClusterPayload
steerlab-cli cluster push --site <id> --json
```

Prefer that form — it keeps the Mac CLI and the remote engine on one revision.
When the installed signed app must not be replaced, or the machine carries no
signing identity, build the payload alone and push that directory instead:

```bash
scripts/make-server-payload.sh --source <checkout> --output <dir> --force
steerlab-cli cluster push --site <id> --payload <dir> --json
```

Legal, and it leaves a Mac CLI talking to an engine built from different
source: a skew you then have to carry in your head.

**Read what is deployed; never infer it from `git log`.** Both identities are
the `sourceRevision` of a `deployment-manifest.json` — one in the app's
`ClusterPayload/`, one at the remote bundle root — and `cluster status` prints
them beside `payload:`, so `current` visibly means *these two agree*. It still
says nothing about your checkout.

**The staleness check compares against INTENT, not against your app.** A
successful `cluster push` records — per machine, never in the shared `Sites/`
registry — the revision it deployed, and `cluster status` compares the engine
against *that*, naming every identity involved: `current (deployed e9a93c9a =
last pushed; app bundle 5686c2ee)`. So the `--payload` route above reads
`current` even though the deployed engine is newer than the app you are
running. Only a site this machine has never pushed to falls back to comparing
against the app bundle alone. When the deployed engine matches neither, the
message says what a push would DO — *pushing will REPLACE deployed X with the
bundle's Y* — because that flag is a repair in one direction and a silent
rollback in the other.

**What `current` cannot promise: that YOUR build is running.** It says
deployed == last pushed, which is the anti-rollback answer and is correct as
far as it goes — but the app bundle can have moved on since that push. It once
read `current` all day while the deployed engine trailed the bundle by eight
commits of engine-side semantics, and six GPU sweeps ran on stale selection
logic under a clean status line. So when the deployed revision differs from
this build's payload, the detail now appends *"server-side changes since that
push are NOT running; push a fresh payload if the study needs them"* — an
advisory, never a state change, and never a rollback offer. `remote
submit-bundle --site` prints the same warning once on stderr before
submitting, computed from local records only (no SSH probe grows on the submit
path), so a `--url` invocation or a never-pushed site stays honestly silent.

**The rule for you: after engine-side changes land that a study depends on,
push a fresh payload — do not read `current` as "my code is running."** Build
the payload, push, then cycle the controller (below). If you see that warning
on a submit, say so and ask before spending GPU time on it.

If a payload gate ever refuses on a site you have
reason to believe is current, the granular verbs reach `connected` without
evaluating the payload at all:

```bash
steerlab-cli cluster controller start --site <id> --json
steerlab-cli cluster tunnel open --site <id> --json
```

`controller start` accepts `--allow-controller-start` and ignores it — typing
the granular verb is itself the authorization.

**A push does not restart the engine.** The running controller keeps the code
it loaded; a push only replaces files on disk. Cycle the controller and
re-open the tunnel before importing anything:

```bash
steerlab-cli cluster controller stop --site <id> --json
steerlab-cli cluster ensure --site <id> --target connected \
  --allow-controller-start --json
```

**The payload is server code only** — `Server/` and `prompts/fixtures/`. Every
Mac-side verb (`cluster import`, `experiment attach`, all the authoring verbs)
is updated by rebuilding the app, never by pushing, so a refusal or a defect in
one of them is never fixed by a push. When you are unsure which side a verb
lives on, check whether it appears under `Server/steerlab_server/cli.py`.

### 8.2 Where the depth is

This file is deliberately self-contained for study work, but it is not the
whole reference. The code checkout (normally a sibling of this workspace's
SteerLab home, e.g. `~/SteerLab/<checkout>/`) carries the depth:
`docs/CLI-REFERENCE.md` — every verb, flag, envelope, and refusal on BOTH
command lines, generated from the parsers, so it is never stale —
plus `docs/ONBOARDING.md` (§9 is specifically about driving SteerLab as an
agent), `docs/CONDUCTING-A-STUDY.md`, and `SECURITY.md`. When a verb here
seems to lack a flag you need, check CLI-REFERENCE before improvising.

There is also a cross-platform Python client, `steerlab` (CLI-REFERENCE
§1.4, `docs/PORTABILITY-CONTRACTS.md`): it authors, packages, and runs this
workspace's studies against any runner from any OS — `steerlab run <name>
--runner <url>` is the whole frozen-study-to-verified-evidence round trip.
On a Mac the Swift CLI above remains the primary instrument.

---

## 9. What not to do

- **Do not parse prose.** Use `--json` and read `state`, `error.code`,
  `error.gate`, `error.gates[]`, `advisories[].code`, and `result`.
- **Do not retry a `refused` (65) without performing the repair.** It will
  refuse identically. Read `error.repairAction`.
- **Do not `--force` a freeze to get past a gate.** It is stamped, loud, and
  permanently non-citable. Fix the gate instead. If a human explicitly asks for
  a forced freeze, report every id in `forcedGatesSkipped`.
- **Do not write into `runs/`**, and do not edit a frozen manifest or a
  `pinned/` snapshot.
- **Do not edit a manifest to iterate.** `duplicate`, then edit the copy.
- **Do not treat an advisory as a failure**, and do not ignore one.
- **Do not skip `validation.jsonl`.** A study whose vectors were never probed
  on held-out material measures its own stimulus vocabulary.
- **Do not select a steering cell on marker density** for any study whose
  outcome is a decision rather than prose. It is a manipulation check.
- **Do not cite seeded or sample content.** It is there to be modified.
- **Do not guess at a file shape.** The shapes in §3 are the ones the loaders
  parse; a wrong key is refused with the expected key named.
"""#
}
