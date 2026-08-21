// Embedded (native SteerLab app) ingestion.
//
// The macOS app presents this page in a WKWebView served over a custom URL
// scheme; the scheme handler also answers `/api/tree` and `/api/file`
// read-only from the active workspace's `runs/` directory. This adapter
// implements the SAME structural directory/file handle surface the File
// System Access path uses, so every parser downstream of run discovery is
// shared byte for byte between the browser and the native app.
//
// The handle contract mirrors ResultsExplorer's local types on purpose —
// TypeScript's structural typing makes an embedded handle a drop-in
// LocalDirectoryHandle.

type TreeEntry = {
  name: string;
  kind: "file" | "directory";
  size: number;
  modified: number;
};

export type EmbeddedFileHandle = {
  kind: "file";
  name: string;
  getFile: () => Promise<File>;
};

export type EmbeddedDirectoryHandle = {
  kind: "directory";
  name: string;
  values: () => AsyncIterableIterator<EmbeddedFileHandle | EmbeddedDirectoryHandle>;
  getDirectoryHandle: (name: string) => Promise<EmbeddedDirectoryHandle>;
  getFileHandle: (name: string) => Promise<EmbeddedFileHandle>;
};

/** True when the page is hosted by the native app's WKWebView pane. */
export const isEmbedded = () =>
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("embedded") === "steerlab";

/** The run the native Results tab deep-linked, if any. */
export const embeddedRunName = () =>
  typeof window === "undefined"
    ? null
    : new URLSearchParams(window.location.search).get("run");

/**
 * The section a `?view=` permalink names, VERBATIM (upgrade plan Phase 5).
 * Validation against the view vocabulary is `lib/deeplink.ts`'s job — this
 * adapter only reads the URL the host handed it.
 */
export const embeddedViewParam = () =>
  typeof window === "undefined"
    ? null
    : new URLSearchParams(window.location.search).get("view");

/** The record a `?record=` permalink names, verbatim (per-view opaque key). */
export const embeddedRecordParam = () =>
  typeof window === "undefined"
    ? null
    : new URLSearchParams(window.location.search).get("record");

/** The workspace display name the native host stamped into the URL. */
export const embeddedWorkspaceName = () =>
  (typeof window === "undefined"
    ? null
    : new URLSearchParams(window.location.search).get("workspace")) ||
  "workspace";

const treeURL = (path: string) => `/api/tree?path=${encodeURIComponent(path)}`;
const fileURL = (path: string) => `/api/file?path=${encodeURIComponent(path)}`;

const listTree = async (path: string): Promise<TreeEntry[]> => {
  const response = await fetch(treeURL(path));
  if (!response.ok) throw new Error(`tree '${path}': HTTP ${response.status}`);
  return (await response.json()) as TreeEntry[];
};

// Lazy File-like object: run DISCOVERY touches only name/size/lastModified,
// so listing a workspace must not download every artifact — bytes are
// fetched once, on first read, and cached for the object's lifetime. The
// cast is safe because readers only use the surface implemented here
// (text / slice().text / arrayBuffer / size / lastModified / name).
const lazyFile = (path: string, entry: TreeEntry): File => {
  let blob: Promise<Blob> | null = null;
  const load = () =>
    (blob ??= fetch(fileURL(path)).then(async (response) => {
      if (!response.ok) throw new Error(`file '${path}': HTTP ${response.status}`);
      return await response.blob();
    }));
  const like = {
    name: entry.name,
    size: entry.size,
    lastModified: entry.modified,
    type: "",
    text: async () => await (await load()).text(),
    arrayBuffer: async () => await (await load()).arrayBuffer(),
    // A slice is a BOUNDED host read (review 2026-08-03, P2): the native
    // bridge serves offset/length, so previewing the head of a huge
    // generations.jsonl never materializes the whole file — on either
    // side of the bridge.
    slice: (start?: number, end?: number) => ({
      text: async () => {
        const offset = Math.max(0, start ?? 0);
        const length = end == null ? null : Math.max(0, end - offset);
        const bounded =
          `${fileURL(path)}&offset=${offset}` +
          (length == null ? "" : `&length=${length}`);
        const response = await fetch(bounded);
        if (!response.ok) {
          throw new Error(`file '${path}': HTTP ${response.status}`);
        }
        return await response.text();
      },
    }),
  };
  return like as unknown as File;
};

const fileHandle = (path: string, entry: TreeEntry): EmbeddedFileHandle => ({
  kind: "file",
  name: entry.name,
  getFile: async () => lazyFile(path, entry),
});

const directoryHandle = (
  path: string,
  name: string,
): EmbeddedDirectoryHandle => ({
  kind: "directory",
  name,
  values: () => {
    const iterate = async function* () {
      for (const entry of await listTree(path)) {
        const childPath = path ? `${path}/${entry.name}` : entry.name;
        yield entry.kind === "file"
          ? fileHandle(childPath, entry)
          : directoryHandle(childPath, entry.name);
      }
    };
    return iterate();
  },
  getDirectoryHandle: async (child: string) => {
    const entries = await listTree(path);
    const match = entries.find(
      (entry) => entry.kind === "directory" && entry.name === child,
    );
    if (!match) throw new Error(`no directory '${child}' under '${path}'`);
    return directoryHandle(path ? `${path}/${child}` : child, child);
  },
  getFileHandle: async (child: string) => {
    const entries = await listTree(path);
    const match = entries.find(
      (entry) => entry.kind === "file" && entry.name === child,
    );
    if (!match) throw new Error(`no file '${child}' under '${path}'`);
    return fileHandle(path ? `${path}/${child}` : child, match);
  },
});

/** The workspace's runs/ directory, served by the native host. */
export const embeddedRunsDirectory = (): EmbeddedDirectoryHandle =>
  directoryHandle("", "runs");
