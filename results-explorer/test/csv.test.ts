import { describe, expect, it } from "vitest";
import { splitCSV, strictNumber } from "../app/lib/csv";

describe("strictNumber", () => {
  // The review-2026-08-03 P0 contract: a blank cell is MISSING, never 0.
  // `Number("")` is 0, which turned an absent adjusted p into q = 0 —
  // falsely significant. These cases pin that behaviour.
  it("reads a blank cell as null, not zero", () => {
    expect(strictNumber("")).toBeNull();
    expect(strictNumber("   ")).toBeNull();
    expect(strictNumber(null)).toBeNull();
    expect(strictNumber(undefined)).toBeNull();
  });

  it("reads an explicit zero as zero", () => {
    expect(strictNumber("0")).toBe(0);
    expect(strictNumber(" 0 ")).toBe(0);
    expect(strictNumber("0.0")).toBe(0);
  });

  it("reads ordinary finite numbers", () => {
    expect(strictNumber("2.5")).toBe(2.5);
    expect(strictNumber("-1.25")).toBe(-1.25);
    expect(strictNumber("1e-3")).toBe(0.001);
  });

  it("reads garbage and non-finite values as null", () => {
    expect(strictNumber("n/a")).toBeNull();
    expect(strictNumber("NaN")).toBeNull();
    expect(strictNumber("Infinity")).toBeNull();
    expect(strictNumber("12 months")).toBeNull();
  });
});

describe("splitCSV", () => {
  it("splits a plain row", () => {
    expect(splitCSV("endpoint,deltaMean,ciLower")).toEqual(["endpoint", "deltaMean", "ciLower"]);
  });

  it("keeps commas inside quotes in one cell", () => {
    expect(splitCSV('a,"b,c",d')).toEqual(["a", "b,c", "d"]);
  });

  it("unescapes doubled quotes inside a quoted cell", () => {
    expect(splitCSV('a,"he said ""yes""",c')).toEqual(["a", 'he said "yes"', "c"]);
  });

  it("preserves empty cells, including trailing ones", () => {
    expect(splitCSV("a,,c,")).toEqual(["a", "", "c", ""]);
    expect(splitCSV("")).toEqual([""]);
  });

  it("keeps a quoted empty cell empty", () => {
    expect(splitCSV('a,"",c')).toEqual(["a", "", "c"]);
  });
});
