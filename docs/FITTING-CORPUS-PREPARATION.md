# Prepare existing text for fitting

A fitting corpus is the text population used to estimate a model's J-lens.
It is not a positive/negative concept dataset. Choose that population with the
researcher: broad prose for a general instrument, domain text for a domain
instrument, or a documented mixture. Prepare assessment text separately,
preferably from a different source split. Fitting success does not establish
readout quality.

## In the app

Open **Fit a new lens → Data → Prepare corpus from existing data**.

1. Choose existing local files, or public Hugging Face dataset files. Local
   files are copied into the workspace; their originals are unchanged. The
   WikiText example fills a source and a length filter but starts no download.
2. Choose the text column, number of records, sampling policy, and scan limit.
   Keep whole records unless there is a reason to select character passages.
   An optional document ID column records identities for later overlap checks.
3. Preview. For a public dataset, this explicitly downloads the selected data
   files. Review example text, skipped records, and the sampling scope.
4. Save. The corpus and receipt fill the fitting form automatically. The
   fitting request retains its small pilot default; preparation does not run it.

An optional token preview uses the exact model's locally cached tokenizer.
It reports truncation and rows too short for the selected position settings.
No model weights or tokenizer are downloaded. If that tokenizer is unavailable,
preparation still works; the fitting pilot measures lengths on its own runtime.

## Both command lines

Use `steerlab science` on the Python client, or `steerlab-cli science` on the
Mac. The following two verbs have the same owner and arguments. Local source
files must first be placed inside the workspace (the app copies chosen files
for you). The specification file itself is a local client input.

Save a specification such as:

```json
{
  "source": {"kind": "local", "files": ["sources/documents.jsonl"]},
  "textColumn": "text",
  "documentColumn": null,
  "selection": "seeded",
  "seed": 0,
  "count": 1000,
  "minChars": 600,
  "passageChars": 0,
  "scanLimit": 100000
}
```

Then:

```sh
steerlab science corpus-preview spec.json --root <workspace> --json
steerlab science corpus-publish <previewID> --plan-sha256 <planSHA256> \
  --destination prompts/fitting/<new-name> --root <workspace> --json
```

The first result contains examples, counts, warnings, a `previewID`, and a
`planSHA256`. Review it before the second command. Preview captures candidate
bytes under `.steerlab/corpus-preparations/`; it is not a write-free plan.
Publication verifies those captured bytes and creates a new directory. A
source changed after preview is not silently read again or resampled. A changed
candidate is refused with a fresh-preview repair. Existing destinations are
never replaced.

Publication returns `fittingInputs.corpus` and `fittingInputs.corpusReceipt`, each
with a workspace-relative path and SHA-256. Put both in a `jlens-fit` config,
or give their paths as the corresponding `science draft` interview fields.
The managed input closure carries both; the fitting run captures the receipt
under `inputs/corpus-preparation.json`. Manually authored corpora can omit it.
The receipt is separate from corpus bytes and checkpoint numerical identity.

## Public Hugging Face sources

Replace `source` with, for example:

```json
{
  "kind": "huggingface",
  "dataset": "Salesforce/wikitext",
  "revision": "main",
  "files": ["wikitext-103-raw-v1/train-*.parquet"]
}
```

The [dataset repository](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-raw-v1)
shows the configuration and train, validation, and test file paths. Select the
intended split explicitly. `revision` can be a branch, tag, or commit; preview
resolves it to an immutable commit and records that commit, ordered source
paths, file sizes, and SHA-256 hashes. Reproduce from that recorded commit,
not the current `main`. Patterns expand in lexical order within each supplied
pattern; files matched twice are read once. Local files retain supplied order.

Only repository-hosted `.txt`, `.jsonl`, `.csv`, and `.parquet` are supported.
No dataset scripts, arbitrary source URLs, or stored Hugging Face credentials
are used. For gated datasets or other formats, obtain an authorized local
export first. The download bound is 2 GiB across selected files. Readers scan
records incrementally; selected Parquet shards are downloaded in full, not
streamed over the network record by record. Choose fewer shards for very large
datasets, and record that narrower sampling population.

`selection: "first"`, `minChars: 600`, and `passageChars: 0` implement the
reference-style policy of taking the first eligible long records, preserving
text and duplicates. This is distinct from a random sample and does not claim
that a changed dataset revision reproduces a historical fit. Seeded selection
removes exact duplicate source text and selects by a deterministic SHA-256
priority over seed, source path, record number, and text hash.

## Settings and provenance

| Setting | Meaning |
|---|---|
| `textColumn` | Source prose column; plain text files always use `text`. CSV needs a header. JSONL may contain other fields. |
| `documentColumn` | Optional string/integer document ID copied into the receipt. Does not group, split, or deduplicate documents. |
| `selection` | `seeded` sample of eligible scanned records, or `first` eligible records in source order. |
| `seed` | Nonnegative integer through UInt64 maximum. Controls sample priorities and optional character windows. |
| `count` | Requested records, 1–20,000. Fewer available records produce an explicit warning and actual count. |
| `minChars` | Minimum trimmed character length, checked on source text and selected passage. Saved text itself is not stripped. |
| `passageChars` | Zero keeps whole records; positive values take one seeded character window per record. Windows can cut words; no sentence segmentation is implied. |
| `scanLimit` | At most 1,000,000 source records; default 100,000. Sampling is over this scanned portion, not unseen records. |
| `tokenizer` | Optional object with `modelID`, exact `revision`, `maxSeqLen`, and `skipFirst`. Uses offline native tokenization without a chat template or forced BOS. |

The receipt records algorithm version, resolved source, input hashes, normalized
settings, counts, token review, and each selected source record's ordinal,
text hash, document ID when provided, and character offsets. Output rows contain
only a stable `id` and `text`, matching fitting's existing input contract.

A source record/text document is bounded to 8 MiB, selected files to 2 GiB, and
the published corpus and receipt to 64 MiB each. CSV uses the standard parser's
field bound; JSONL is preferable for long documents. Memory is bounded by the
candidate sample, a scan-limited duplicate index, and Parquet batches rather
than the entire dataset. Malformed records are reported; missing or non-text
values, short text, and duplicate text are counted explicitly.

Preparation does not perform automatic document-level train/assessment splits
or certify overlap absence. Use separate source splits and check document IDs
and text hashes in the receipts where the research design requires it. Candidate
scratch and app-imported source copies remain available; this slice adds no
managed cleanup or automatic deletion of old corpora.

## Workbench HTTP and agent contract

`POST /api/science/workspace/corpus-preview` takes exactly `workspaceRoot` and
`specText` (a JSON string containing the specification). `corpus-publish` takes
`workspaceRoot`, `previewID`, `planSHA256`, and `destination`. These use the same
portable owner as both CLIs and the app; the existing route authority keeps
workspace authorship on workbenches. A runner's execution role is unchanged.
Local paths must be within the serving workspace; an HTTP caller cannot name
an arbitrary filesystem source outside it.

Agents should explain the population and sampling choice before preparing
text. Offer existing files or an agreed public source, and ask about unresolved
choices. Do not generate research data, delegate authorship, download a dataset,
or start fitting merely because the researcher named a model or concept. Once
source preparation is explicitly requested, review actual examples and limits,
publish the selected bytes, and propose a small fitting pilot as the next step.

Client releases now include CPU Parquet, Hugging Face, and tokenizer dependencies
in their managed dependency lock. They still contain no torch or GPU runtime.
Rebuild the app and Python payload together before deploying these new actions;
update an older managed client environment through its normal reviewed setup
plan. The installed app and cluster were not changed by this branch.
