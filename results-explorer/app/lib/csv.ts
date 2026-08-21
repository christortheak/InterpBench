// CSV cell splitting and the strict numeric reader every artifact parser
// shares.

export const splitCSV = (line: string) => {
  const values: string[] = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && line[index + 1] === '"' && quoted) { value += '"'; index += 1; }
    else if (character === '"') quoted = !quoted;
    else if (character === "," && !quoted) { values.push(value); value = ""; }
    else value += character;
  }
  values.push(value);
  return values;
};

// STRICT numeric parsing (review 2026-08-03, P0): `Number("") === 0`, so a
// blank CSV cell silently became a substantive zero — a missing adjusted
// p-value even read as q = 0, falsely significant. A cell is a number only
// when non-empty and finite; missing stays null.
export const strictNumber = (raw: unknown): number | null => {
  const text = String(raw ?? "").trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
};
