// Checks on the BUILT embedded bundle — what the Mac app ships. Run with
// `npm run test:embedded`, which builds it first. (This file replaces a test
// that asserted a starter template's loading page, which the explorer has not
// had for a long time.)
//
// The bundle is read from STEERLAB_EMBED_OUT_DIR when set, else from the
// default output, ../web/results-explorer/ (build output; never committed).

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const bundle = process.env.STEERLAB_EMBED_OUT_DIR
  || fileURLToPath(new URL("../../web/results-explorer/", import.meta.url));

const demoSource = await readFile(new URL("../app/demo/index.tsx", import.meta.url), "utf8");
const marker = /export const DEMO_MARKER = "([^"]+)"/.exec(demoSource)?.[1];
const listed = [.../export const DEMO_SENTINELS = \[([\s\S]*?)\];/.exec(demoSource)?.[1].matchAll(/"([^"]+)"/g) ?? []].map((match) => match[1]);
const sentinels = [marker, ...listed].filter(Boolean);

const files = async (directory) => (await readdir(directory, { withFileTypes: true }))
  .flatMap((entry) => entry.isDirectory() ? [] : [join(directory, entry.name)])
  .concat(...await Promise.all((await readdir(directory, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => files(join(directory, entry.name)))));

test("the bundle has a page and its assets, with relative paths", async () => {
  const html = await readFile(join(bundle, "index.html"), "utf8");
  assert.match(html, /<div id="root"><\/div>/);
  assert.match(html, /src="\.\/assets\/[^"]+\.js"/);
  assert.doesNotMatch(html, /(src|href)="\/(?!\/)/, "an absolute asset path would not resolve over the app's URL scheme");
});

test("no invented demo content is in the bundle", async () => {
  assert.ok(sentinels.length >= 2, "could not read DEMO_SENTINELS from app/demo/index.tsx");
  for (const path of await files(bundle)) {
    if (!/\.(js|css|html|json)$/.test(path)) continue;
    const text = await readFile(path, "utf8");
    for (const sentinel of sentinels) assert.ok(!text.includes(sentinel), `${path} contains demo content: "${sentinel}"`);
  }
});

test("the old fictional results are not in the bundle either", async () => {
  // Strings from the demo pages that used to ship, including the banner
  // that told a reader an invented run had been verified.
  const retired = ["Run epoch verified", "All checksums valid", "5 / 6 gates closed", "384 RECORDS", "Synthetic preview"];
  for (const path of await files(bundle)) {
    if (!/\.(js|html)$/.test(path)) continue;
    const text = await readFile(path, "utf8");
    for (const phrase of retired) assert.ok(!text.includes(phrase), `${path} contains "${phrase}"`);
  }
});
