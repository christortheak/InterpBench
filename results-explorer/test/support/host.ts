// The two places the explorer runs, stood up as plain objects for the unit
// suite: the Mac app's web view (EMBEDDED) and a browser tab (STANDALONE).
// Neither is a real WebKit or a real browser — see support/capture.ts for
// what that leaves untested.

import { vi } from "vitest";
import type { SaveMessage } from "../../app/lib/save";

type Reply = (message: SaveMessage) => unknown | Promise<unknown>;

// The real constructor, kept so the stubs below can still build URLs after
// the global is replaced.
const OriginalURL = URL;

export type EmbeddedHost = {
  /// Every message the page posted to the native save handler, in order.
  posted: SaveMessage[];
  /// `URL.createObjectURL` — the call that threw inside the app. An
  /// embedded save must never reach it.
  createObjectURL: ReturnType<typeof vi.fn>;
};

/// The embedded app: `?embedded=steerlab` in the address, the native save
/// handler on `window.webkit`, no downloads, and an object-URL call that
/// throws on anything that is not a real Blob — as WebKit's does.
export const enterEmbedded = (options: { reply?: Reply; withoutSaveHandler?: boolean; search?: string } = {}): EmbeddedHost => {
  const posted: SaveMessage[] = [];
  const reply: Reply = options.reply ?? ((message) => ({ state: "saved", name: message.filename }));
  const steerlabSave = { postMessage: async (message: SaveMessage) => { posted.push(message); return reply(message); } };
  vi.stubGlobal("window", {
    location: { search: options.search ?? "?embedded=steerlab&workspace=study", origin: "steerlab-explorer://app", pathname: "/index.html", hash: "" },
    history: { replaceState() {} },
    scrollTo() {},
    webkit: { messageHandlers: options.withoutSaveHandler ? {} : { steerlabSave } },
  });
  const createObjectURL = vi.fn((value: unknown) => {
    if (!(value instanceof Blob)) throw new TypeError("Failed to execute 'createObjectURL' on 'URL': Overload resolution failed.");
    return "blob:embedded";
  });
  vi.stubGlobal("URL", Object.assign(function URLStub(this: unknown, ...args: ConstructorParameters<typeof URL>) { return new OriginalURL(...args); }, { createObjectURL, revokeObjectURL: vi.fn() }));
  return { posted, createObjectURL };
};

export type BrowserHost = {
  /// One entry per download the page started: the anchor's `download` name
  /// and the Blob behind its object URL.
  downloads: { filename: string; blob: Blob }[];
  revoked: string[];
};

/// A browser tab: no `embedded` flag, no native handler, and a document
/// whose anchors record what they were asked to download.
export const enterBrowser = (options: { search?: string } = {}): BrowserHost => {
  const downloads: { filename: string; blob: Blob }[] = [];
  const revoked: string[] = [];
  const blobs = new Map<string, Blob>();
  vi.stubGlobal("window", {
    location: { search: options.search ?? "", origin: "http://localhost:3000", pathname: "/", hash: "" },
    history: { replaceState() {} },
    scrollTo() {},
  });
  vi.stubGlobal("document", {
    createElement: (tag: string) => {
      const anchor = { tag, href: "", download: "", click() { downloads.push({ filename: anchor.download, blob: blobs.get(anchor.href)! }); } };
      return anchor;
    },
    querySelector: () => null,
  });
  let next = 0;
  vi.stubGlobal("URL", Object.assign(function URLStub(this: unknown, ...args: ConstructorParameters<typeof URL>) { return new OriginalURL(...args); }, {
    createObjectURL: (value: unknown) => {
      if (!(value instanceof Blob)) throw new TypeError("Failed to execute 'createObjectURL' on 'URL': Overload resolution failed.");
      const url = `blob:browser-${next += 1}`;
      blobs.set(url, value);
      return url;
    },
    revokeObjectURL: (url: string) => { revoked.push(url); },
  }));
  return { downloads, revoked };
};

/// Undo `enterEmbedded` / `enterBrowser`. Call from `afterEach`.
export const leaveHost = () => vi.unstubAllGlobals();

/// The native bridge's two read endpoints, answered from an in-memory tree:
/// `files` maps a runs-relative path to that file's text.
export const serveRuns = (files: Record<string, string>) => {
  const requested: string[] = [];
  const fetchStub = vi.fn(async (input: string) => {
    requested.push(input);
    const url = new OriginalURL(input, "steerlab-explorer://app");
    const path = url.searchParams.get("path") ?? "";
    if (url.pathname === "/api/tree") {
      const prefix = path ? `${path}/` : "";
      const names = new Map<string, { name: string; kind: "file" | "directory"; size: number; modified: number }>();
      for (const [file, text] of Object.entries(files)) {
        if (!file.startsWith(prefix)) continue;
        const [head, ...rest] = file.slice(prefix.length).split("/");
        names.set(head, rest.length ? { name: head, kind: "directory", size: 0, modified: 0 } : { name: head, kind: "file", size: new TextEncoder().encode(text).length, modified: 1 });
      }
      return new Response(JSON.stringify([...names.values()]), { status: 200 });
    }
    if (url.pathname === "/api/file" && path in files) {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const length = url.searchParams.get("length");
      const bytes = new TextEncoder().encode(files[path]);
      return new Response(bytes.slice(offset, length === null ? undefined : offset + Number(length)), { status: 200 });
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchStub);
  return { requested };
};
