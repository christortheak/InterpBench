// Rendering a view and pressing its buttons WITHOUT a browser.
//
// The unit suite runs in plain Node: there is no DOM library in the lockfile,
// and adding one for a handful of controls would widen what a cold clone has
// to install. So a "click" here is honest about what it is:
//
//   1. the component is rendered once with React's own server renderer, which
//      runs every component function (hooks included) and returns the HTML;
//   2. while it renders, every element React is asked to create is recorded,
//      which is how a `<button onClick={…}>`'s handler is reached;
//   3. `click(label)` finds the one button with that label and calls its
//      handler, exactly as React would on a real click.
//
// What this does NOT exercise: effects (`useEffect` never runs on the server
// renderer), state updates after the click (they are dropped), layout, and
// the real WebKit bridge. Tests built on it say what a control SENDS when
// pressed; what the app does with it is covered on the Swift side
// (ResultsExplorerBridgeTests) and still needs a look in the running app.
//
// A test file opts in with two lines, before its other imports:
//
//   vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
//   vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import type { ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

type Captured = { type: unknown; props: Record<string, unknown> };
type Factory = (type: unknown, props: Record<string, unknown>, ...rest: unknown[]) => unknown;

const created: Captured[] = [];

const recording = (factory: Factory): Factory => (type, props, ...rest) => {
  created.push({ type, props: props ?? {} });
  return factory(type, props, ...rest);
};

/// Wrap React's element factories so each created element is recorded. The
/// elements themselves are untouched.
export const wrapJSXRuntime = <Runtime extends object>(actual: Runtime): Runtime => {
  const source = actual as Record<string, unknown>;
  const wrapped: Record<string, unknown> = { ...source };
  for (const name of ["jsx", "jsxs", "jsxDEV"]) {
    if (typeof source[name] === "function") wrapped[name] = recording(source[name] as Factory);
  }
  return wrapped as Runtime;
};

/// The visible text of an element's children, for matching a button by label.
export const textOf = (node: unknown): string => {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (typeof node === "object" && "props" in node) return textOf((node as { props: { children?: unknown } }).props.children);
  return "";
};

export type RenderedButton = { label: string; disabled: boolean; title: string; props: Record<string, unknown> };

export type Rendered = {
  html: string;
  /// The page's text with tags removed and entities decoded.
  text: string;
  buttons: RenderedButton[];
  button: (label: string) => RenderedButton;
  /// Press the one enabled button whose label contains `label`, and wait
  /// for whatever its handler returns.
  click: (label: string) => Promise<void>;
};

// Emphasis tags sit INSIDE a sentence ("<strong>rule</strong>: what it
// does"), so they vanish; every other tag separates pieces of the page, so
// it becomes a space.
const decode = (html: string) => html
  .replace(/<\/?(?:strong|em|b|i|code|sub|sup)(?:\s[^>]*)?>/g, "")
  .replace(/<[^>]*>/g, " ")
  .replaceAll("&amp;", "&").replaceAll("&lt;", "<").replaceAll("&gt;", ">")
  .replaceAll("&quot;", "\"").replaceAll("&#x27;", "'")
  .replace(/\s+/g, " ").trim();

export const render = (element: ReactElement): Rendered => {
  created.length = 0;
  const html = renderToStaticMarkup(element);
  const buttons: RenderedButton[] = created
    .filter((entry) => entry.type === "button")
    .map((entry) => ({
      label: textOf(entry.props.children).replace(/\s+/g, " ").trim(),
      disabled: entry.props.disabled === true,
      title: typeof entry.props.title === "string" ? entry.props.title : "",
      props: entry.props,
    }));
  const button = (label: string) => {
    const matches = buttons.filter((candidate) => candidate.label.includes(label));
    if (matches.length !== 1) {
      throw new Error(`expected exactly one button labelled "${label}", found ${matches.length} among: ${buttons.map((candidate) => `"${candidate.label}"`).join(", ")}`);
    }
    return matches[0];
  };
  const click = async (label: string) => {
    const target = button(label);
    if (target.disabled) throw new Error(`the "${label}" button is disabled`);
    const handler = target.props.onClick;
    if (typeof handler !== "function") throw new Error(`the "${label}" button has no click handler`);
    await handler({ stopPropagation() {}, preventDefault() {} });
  };
  return { html, text: decode(html), buttons, button, click };
};
