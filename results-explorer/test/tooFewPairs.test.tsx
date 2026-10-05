import { describe, expect, it } from "vitest";
import { ForestRow } from "../app/components/ui";
import { effectKey } from "../app/lib/effects";
import type { Effect } from "../app/lib/types";
import { render } from "./support/capture";

/// One or two independent pairs carry no interval and no test, on every
/// surface: the forest row draws no whisker and no significance dot. For
/// paired responses, what counts is the items they came from.
const effect = (overrides: Partial<Effect>): Effect => {
  const base = {
    condition: "steered", endpoint: "wordCount", short: "wordCount", estimate: 2, low: 1.5, high: 2.5,
    unit: "words", n: 8 as number | null, q: 0.01 as number | null, p: 0.01 as number | null, correction: "bh",
    direction: "positive" as const, stratifyBy: "pooled", stratum: "", pairedUnit: "", estimand: "", inference: "",
    effectUnit: { unit: "item", source: "engine_default" as const, pairedItems: 8 },
    ...overrides,
  };
  return { ...base, key: effectKey(base) } as Effect;
};

describe("too few independent pairs", () => {
  it("draws no whisker and no significance dot for responses from one item", () => {
    const row = effect({ n: 5, effectUnit: { unit: "response", source: "inferred_from_records", pairedItems: 1 } });
    const { html } = render(<ForestRow effect={row} />);
    expect(html).not.toContain("ci-line");
    expect(html).not.toContain("is-sig");
    expect(html).toContain("too few items for an interval");
  });

  it("still draws them for responses from enough items", () => {
    const row = effect({ n: 8, effectUnit: { unit: "response", source: "inferred_from_records", pairedItems: 4 } });
    const { html } = render(<ForestRow effect={row} />);
    expect(html).toContain("ci-line");
    expect(html).toContain("is-sig");
  });

  it("counts pairs, not items, for an item-level row", () => {
    const { html } = render(<ForestRow effect={effect({ n: 2, effectUnit: { unit: "item", source: "engine_default", pairedItems: 2 } })} />);
    expect(html).not.toContain("ci-line");
    expect(html).toContain("too few pairs for an interval");
  });
});
