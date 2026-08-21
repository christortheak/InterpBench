// The EMBEDDED entry: a plain static SPA (no worker, no RSC, no server)
// the native macOS app ships as a code resource and presents in a
// WKWebView. Same component, same parsers — only the ingestion transport
// differs (see app/embedded-workspace.ts).
import { createRoot } from "react-dom/client";
import ResultsExplorer from "../app/ResultsExplorer";
import "../app/globals.css";

const root = document.getElementById("root");
if (root) createRoot(root).render(<ResultsExplorer />);
