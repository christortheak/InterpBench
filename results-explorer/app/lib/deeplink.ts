// Permalinks: `?run=&view=&record=` (upgrade plan Phase 5, "Deep links").
//
// The native app already opened the explorer at `?run=<name>`; this widens
// that to a citable address for a SECTION and a RECORD, so analysis prose
// can point at the transcript it is about instead of describing where to
// click.
//
// Scope rule. A URL only identifies anything in EMBEDDED mode, where the
// host serves one known workspace over a stable origin. In the browser build
// the workspace is whatever folder the researcher granted for this session:
// there is no address for it, so link-copying is hidden and nothing is
// written back to the URL. Reading params is harmless either way.
//
// Everything here is pure except `updateDeepLink` / `deepLinkHref`, which
// touch `history` / `window`; the parsing and patching functions are the
// tested surface.

import type { View } from "./types";

/// Every view id, as DATA. The `Record<View, true>` makes the compiler
/// enforce that a new view is added here too — an unlisted view could never
/// be deep-linked, silently.
const VIEWS: Record<View, true> = {
  triage: true,
  overview: true,
  concepts: true,
  optimization: true,
  effects: true,
  battery: true,
  choice: true,
  judged: true,
  coding: true,
  panels: true,
  residuals: true,
  generations: true,
  provenance: true,
};

export const VIEW_IDS = Object.keys(VIEWS) as View[];

/// A `view=` param that is not a view id is IGNORED rather than matched to
/// the nearest name — a stale link should land on the run, not on a section
/// the reader did not ask for.
export const asView = (raw: string | null | undefined): View | null =>
  raw && Object.prototype.hasOwnProperty.call(VIEWS, raw) ? raw as View : null;

export type DeepLink = {
  /// The run DIRECTORY NAME to activate; "" when the link names none.
  run: string;
  view: View | null;
  /// Opaque per-view record key (see `joinRecordKey`); "" when absent.
  record: string;
  /// A `view=` value that was present but is not a view id — surfaced so a
  /// caller can say so rather than silently dropping it.
  unknownView: string;
};

export const parseDeepLink = (search: string): DeepLink => {
  const params = new URLSearchParams(search);
  const rawView = params.get("view");
  return {
    run: params.get("run") ?? "",
    view: asView(rawView),
    record: params.get("record") ?? "",
    unknownView: rawView && !asView(rawView) ? rawView : "",
  };
};

// ---------------------------------------------------------------------------
// Record keys
// ---------------------------------------------------------------------------

/// Record keys are `part|part|part`. The separator is a character no
/// condition name, prompt id, or judge name in either engine's artifacts
/// uses, and `encodeURIComponent` escapes it in the URL.
export const RECORD_SEPARATOR = "|";

export const joinRecordKey = (...parts: (string | number | null | undefined)[]) =>
  parts.map((part) => (part === null || part === undefined ? "" : String(part))).join(RECORD_SEPARATOR);

export const splitRecordKey = (key: string) => key.split(RECORD_SEPARATOR);

/// The three record browsers all address a response by the same triple the
/// engines join on: (condition, promptID, sampleIndex). Coding adds the
/// coder, since one response carries one row per coder.
export const responseRecordKey = (condition: string, promptID: string, sampleIndex: number) =>
  joinRecordKey(condition, promptID, sampleIndex);

export const codingRecordKey = (condition: string, promptID: string, sampleIndex: number, judge: string) =>
  joinRecordKey(condition, promptID, sampleIndex, judge);

// ---------------------------------------------------------------------------
// Writing the URL back
// ---------------------------------------------------------------------------

export type DeepLinkPatch = { run?: string; view?: View; record?: string | null };

/// Pure core of the URL update: the existing query string plus the patch.
/// A `record` of null or "" REMOVES the param (no record is selected), which
/// is different from carrying an empty one.
export const applyDeepLinkParams = (search: string, patch: DeepLinkPatch): string => {
  const params = new URLSearchParams(search);
  if (patch.run !== undefined) {
    if (patch.run) params.set("run", patch.run); else params.delete("run");
  }
  if (patch.view !== undefined) params.set("view", patch.view);
  if (patch.record !== undefined) {
    if (patch.record) params.set("record", patch.record); else params.delete("record");
  }
  const text = params.toString();
  return text ? `?${text}` : "";
};

const embedded = () =>
  typeof window !== "undefined"
  && new URLSearchParams(window.location.search).get("embedded") === "steerlab";

/// Keep the address bar in step with the selection. `replaceState`, never
/// `pushState`: selecting a record is not a navigation the back button
/// should have to walk through. No-op outside embedded mode, where the URL
/// identifies nothing.
export const updateDeepLink = (patch: DeepLinkPatch) => {
  if (typeof window === "undefined" || !embedded()) return;
  const search = applyDeepLinkParams(window.location.search, patch);
  window.history.replaceState(null, "", `${window.location.pathname}${search}${window.location.hash}`);
};

/// The absolute link to copy. Built from the CURRENT location plus the
/// patch, so it carries the host's own `embedded`/`workspace` params.
export const deepLinkHref = (patch: DeepLinkPatch): string => {
  if (typeof window === "undefined") return "";
  const search = applyDeepLinkParams(window.location.search, patch);
  return `${window.location.origin}${window.location.pathname}${search}`;
};

/// Whether a copy-link affordance can mean anything in this session.
export const deepLinksAvailable = () => embedded();

// ---------------------------------------------------------------------------
// The inbound record hand-off
// ---------------------------------------------------------------------------

// A one-shot module hand-off, deliberately the same shape as the choice →
// generations hand-off in lib/instruments.ts: the shell parses the URL once
// at load and names a record; the target view consumes it when it mounts.
// Not a store — nothing subscribes, and a consumed link must not re-select
// the record every time the reader navigates back.
let pending: { view: View; record: string } | null = null;

export const setPendingRecord = (view: View, record: string) => {
  pending = record ? { view, record } : null;
};

export const takePendingRecord = (view: View): string => {
  if (!pending || pending.view !== view) return "";
  const { record } = pending;
  pending = null;
  return record;
};
