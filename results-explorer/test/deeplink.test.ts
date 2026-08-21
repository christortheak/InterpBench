import { describe, expect, it } from "vitest";
import {
  applyDeepLinkParams,
  asView,
  codingRecordKey,
  joinRecordKey,
  parseDeepLink,
  responseRecordKey,
  setPendingRecord,
  splitRecordKey,
  takePendingRecord,
  VIEW_IDS,
} from "../app/lib/deeplink";

// `?run=&view=&record=` permalinks (upgrade plan Phase 5). The parsing and
// patching are pure; the two functions that touch `history` / `window` are
// thin wrappers over `applyDeepLinkParams`, which is what these pin.

describe("asView", () => {
  it("accepts every view the shell can show", () => {
    for (const view of VIEW_IDS) expect(asView(view)).toBe(view);
    expect(VIEW_IDS).toContain("judged");
    expect(VIEW_IDS).toContain("battery");
    expect(VIEW_IDS).toContain("residuals");
  });

  it("ignores a value that is not a view rather than guessing at one", () => {
    expect(asView("judgements")).toBeNull();
    expect(asView("")).toBeNull();
    expect(asView(null)).toBeNull();
    expect(asView(undefined)).toBeNull();
  });

  it("does not accept inherited Object properties as views", () => {
    expect(asView("toString")).toBeNull();
    expect(asView("constructor")).toBeNull();
  });
});

describe("parseDeepLink", () => {
  it("reads run, view, and record", () => {
    const link = parseDeepLink("?embedded=steerlab&run=20260805T004016927-exp-x-evaluate&view=judged&record=baseline%7Citem-3%7C0");
    expect(link.run).toBe("20260805T004016927-exp-x-evaluate");
    expect(link.view).toBe("judged");
    expect(link.record).toBe("baseline|item-3|0");
    expect(link.unknownView).toBe("");
  });

  it("keeps an unrecognised view out of the reading but reports it", () => {
    const link = parseDeepLink("?run=r&view=nowhere&record=k");
    expect(link.view).toBeNull();
    expect(link.unknownView).toBe("nowhere");
    // The run is still honoured: a stale link lands where it can.
    expect(link.run).toBe("r");
  });

  it("reads absence as absence", () => {
    const link = parseDeepLink("");
    expect(link).toEqual({ run: "", view: null, record: "", unknownView: "" });
  });
});

describe("record keys", () => {
  it("addresses a response by the triple the engines join on", () => {
    expect(responseRecordKey("baseline", "loan-legal-ab", 0)).toBe("baseline|loan-legal-ab|0");
    expect(splitRecordKey(responseRecordKey("fear-agent", "item-3", 2))).toEqual(["fear-agent", "item-3", "2"]);
  });

  it("adds the coder for a coding, since one response has one row per coder", () => {
    expect(codingRecordKey("baseline", "item-3", 1, "judge-b")).toBe("baseline|item-3|1|judge-b");
  });

  it("writes an absent part as an empty segment rather than dropping it", () => {
    expect(joinRecordKey("a", null, 3)).toBe("a||3");
    expect(splitRecordKey("a||3")).toEqual(["a", "", "3"]);
  });
});

describe("applyDeepLinkParams", () => {
  it("sets the view and keeps the host's own params", () => {
    const next = applyDeepLinkParams("?embedded=steerlab&workspace=Study%201&run=r", { view: "coding" });
    const params = new URLSearchParams(next);
    expect(params.get("embedded")).toBe("steerlab");
    expect(params.get("workspace")).toBe("Study 1");
    expect(params.get("run")).toBe("r");
    expect(params.get("view")).toBe("coding");
  });

  it("REMOVES the record param when the patch clears it — no record selected is not an empty record", () => {
    expect(new URLSearchParams(applyDeepLinkParams("?run=r&record=a%7Cb%7C0", { record: null })).has("record")).toBe(false);
    expect(new URLSearchParams(applyDeepLinkParams("?run=r&record=a%7Cb%7C0", { record: "" })).has("record")).toBe(false);
  });

  it("round-trips a record key through the query string", () => {
    const key = responseRecordKey("optimize-x-fear-agent", "loan-legal-ab", 3);
    const search = applyDeepLinkParams("?embedded=steerlab", { view: "generations", record: key });
    expect(parseDeepLink(search).record).toBe(key);
    expect(parseDeepLink(search).view).toBe("generations");
  });

  it("leaves untouched params alone and returns an empty string for an empty query", () => {
    expect(applyDeepLinkParams("?run=r", {})).toBe("?run=r");
    expect(applyDeepLinkParams("", {})).toBe("");
    expect(new URLSearchParams(applyDeepLinkParams("?run=r", { run: "" })).has("run")).toBe(false);
  });
});

describe("the inbound record hand-off", () => {
  it("delivers the record to the named view exactly once", () => {
    setPendingRecord("judged", "baseline|item-3|0");
    expect(takePendingRecord("coding")).toBe("");
    expect(takePendingRecord("judged")).toBe("baseline|item-3|0");
    expect(takePendingRecord("judged")).toBe("");
  });

  it("holds nothing for an empty record", () => {
    setPendingRecord("generations", "");
    expect(takePendingRecord("generations")).toBe("");
  });
});
