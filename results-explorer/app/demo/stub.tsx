// What the EMBEDDED build gets instead of `./index.tsx`: the same names, with
// nothing behind them. No invented row, no invented check, no demo page.
//
// `vite.embed.config.ts` swaps this file in for every import of the demo
// module, so the app a researcher runs contains no demo content at all. Demo
// mode is off for good here, the lists are empty, and the pages render
// nothing — and since demo mode is off, nothing ever asks them to render.
//
// Keep the exports in step with `./index.tsx`; test/demo.test.tsx compares
// the two and fails when they drift.

import type { Effect, Generation, View } from "../lib/types";

export const DEMO_MARKER = "";
export const DEMO_SENTINELS: string[] = [];

export const demoPreviewEnabled = () => false;
export const demoLabel = () => "";

export const demoEffects: Effect[] = [];
export const demoGenerations: Generation[] = [];

export const demoCopy = {
  effectsEyebrow: "",
  effectsSource: "",
  effectsExperiment: "",
  effectsNote: "",
  generationsEyebrow: "",
  generationsModel: "",
  generationsFooter: "",
  downloadLabel: "",
};

export function DemoBanner() { return null; }
// The parameter is part of the shared signature; a page that draws nothing
// has no use for it.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function DemoOverview(_props: { onNavigate: (view: View) => void }) { return null; }
export function DemoProvenance() { return null; }
export function DemoEffectsLower() { return null; }
