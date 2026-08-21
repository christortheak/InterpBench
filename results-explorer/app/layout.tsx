import type { Metadata } from "next";
import "./globals.css";

const title = "SteerLab Results Explorer";
const description = "A rigorous, readable explorer for activation-steering study results, generations, and provenance.";

export const metadata: Metadata = {
  metadataBase: new URL("http://localhost:3000"),
  title,
  description,
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: { title, description, type: "website", images: [{ url: "/og.png", width: 1200, height: 630, alt: title }] },
  twitter: { card: "summary_large_image", title, description, images: ["/og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
