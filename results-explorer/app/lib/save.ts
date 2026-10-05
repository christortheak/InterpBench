// Handing a file to the reader — the ONE place the explorer does it.
//
// Every export and download control calls `saveText` or `saveRunFile`, and
// nothing else in the app touches an object URL or a `download` anchor
// (test/save.test.tsx holds that line). There are two builds and they save
// differently:
//
// - STANDALONE (a browser): a Blob, an object URL, and an anchor with a
//   `download` attribute, which the browser turns into a download.
//
// - EMBEDDED (the Mac app's web view): a web view has no downloads folder and
//   ignores a `download` anchor, and a file there is a lazy stand-in rather
//   than a real `File` (see embedded-workspace.ts), so an object URL cannot be
//   made from it at all. Instead the page posts WHAT to save to the native
//   host (`ResultsExplorerSaveHandler` in the app), the host shows a save
//   panel, and the reader chooses WHERE. A table the page built travels as
//   text; a file from the run travels as its PATH, and the host copies the
//   bytes itself, so a very large generations file is never pulled into the
//   page just to be saved.
//
// The page never names a destination, and the host refuses any destination
// inside the workspace's runs folder: the explorer still changes nothing in
// a run.

import { isEmbedded } from "../embedded-workspace";
import type { LocalFileHandle } from "./types";

/// What happened to a save. "browser" means the file was handed to the
/// browser's own downloads; "panel" means the reader placed it with the
/// app's save panel and the host wrote it.
export type SaveOutcome =
  | { state: "saved"; name: string; via: "panel" | "browser" }
  | { state: "cancelled" }
  | { state: "failed"; message: string };

/// The message the embedded page posts. These two shapes are the whole
/// vocabulary; the host ignores anything else.
export type SaveMessage =
  | { kind: "text"; filename: string; text: string }
  | { kind: "runFile"; filename: string; path: string };

type NativeSaveBridge = { postMessage: (message: SaveMessage) => Promise<unknown> };

type BridgeWindow = { webkit?: { messageHandlers?: { steerlabSave?: NativeSaveBridge } } };

const nativeBridge = (): NativeSaveBridge | null =>
  typeof window === "undefined"
    ? null
    : (window as unknown as BridgeWindow).webkit?.messageHandlers?.steerlabSave ?? null;

export const SAVE_UNAVAILABLE =
  "This copy of the app cannot save files from the explorer. The run's files are still in the workspace's runs folder, where Finder can copy them.";

export const SAVE_PATH_UNKNOWN =
  "The explorer could not tell the app which file this is, so nothing was saved. Rescan the workspace and try again.";

const failed = (message: string): SaveOutcome => ({ state: "failed", message });

const errorText = (error: unknown) =>
  error instanceof Error && error.message ? error.message : typeof error === "string" && error ? error : "";

/// Read the host's reply. Anything that is not one of the three known
/// states is a failure, said plainly — never a silent success.
export const readSaveReply = (reply: unknown, filename: string): SaveOutcome => {
  const fields = reply && typeof reply === "object" ? reply as Record<string, unknown> : {};
  if (fields.state === "saved") return { state: "saved", name: typeof fields.name === "string" && fields.name ? fields.name : filename, via: "panel" };
  if (fields.state === "cancelled") return { state: "cancelled" };
  if (fields.state === "failed" && typeof fields.message === "string" && fields.message) return failed(fields.message);
  return failed("The app did not confirm that the file was saved. Check the place you chose before relying on it.");
};

const saveThroughHost = async (message: SaveMessage): Promise<SaveOutcome> => {
  const bridge = nativeBridge();
  if (!bridge) return failed(SAVE_UNAVAILABLE);
  try {
    return readSaveReply(await bridge.postMessage(message), message.filename);
  } catch (error) {
    return failed(errorText(error) || "The app could not save the file.");
  }
};

const saveThroughBrowser = (filename: string, blob: Blob): SaveOutcome => {
  if (typeof document === "undefined") return failed("There is no page to download from.");
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
  return { state: "saved", name: filename, via: "browser" };
};

/// Save text the page built (a table export).
export const saveText = async (filename: string, text: string, type = "text/csv;charset=utf-8"): Promise<SaveOutcome> => {
  if (isEmbedded()) return saveThroughHost({ kind: "text", filename, text });
  try {
    return saveThroughBrowser(filename, new Blob([text], { type }));
  } catch (error) {
    return failed(errorText(error) || "The browser could not start the download.");
  }
};

/// Save a copy of one file from the run, byte for byte.
export const saveRunFile = async (filename: string, handle: LocalFileHandle | null | undefined): Promise<SaveOutcome> => {
  if (!handle) return failed("This run has no such file to save.");
  if (isEmbedded()) {
    // The path the native host listed this file under. The host re-checks
    // it against the runs folder; the page only repeats it back.
    if (!handle.embeddedPath) return failed(SAVE_PATH_UNKNOWN);
    return saveThroughHost({ kind: "runFile", filename, path: handle.embeddedPath });
  }
  try {
    const file = await handle.getFile();
    if (!(file instanceof Blob)) return failed("This file cannot be downloaded from this page.");
    return saveThroughBrowser(filename, file);
  } catch (error) {
    return failed(errorText(error) || "The file could not be read, so it was not downloaded.");
  }
};

/// The sentence a control shows after a save. "" for a cancelled save: the
/// reader closed the panel on purpose and needs no message about it.
export const saveStatusText = (outcome: SaveOutcome | { state: "saving" } | null): string => {
  if (!outcome || outcome.state === "cancelled") return "";
  if (outcome.state === "saving") return "Saving…";
  if (outcome.state === "saved") return outcome.via === "panel" ? `Saved as ${outcome.name}` : `Sent to your browser's downloads as ${outcome.name}`;
  return `Not saved. ${outcome.message}`;
};
