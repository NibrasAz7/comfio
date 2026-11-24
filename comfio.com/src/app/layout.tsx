import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";
import { ThemeProvider } from "@/components/theme-provider";
import { CinematicScroll } from "@/components/cinematic-scroll";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "comfio — Multi-Domain IEQ & Performance Contract Framework",
  description:
    "A multi-domain IEQ (Indoor Environmental Quality) and performance contract framework for smart buildings. Includes pollutant IAQ, adaptive thermal comfort, sPMV, TSV augmentation, and personalised comfort models.",
  keywords: [
    "thermal comfort", "IEQ", "smart buildings", "PMV", "PPD",
    "ASHRAE 55", "ISO 7730", "performance contracts", "air quality",
  ],
  authors: [{ name: "comfio Contributors" }],
  openGraph: {
    title: "comfio — Multi-Domain IEQ Framework",
    description: "Unify Thermal, Visual, Acoustic, and IAQ metrics into a single Global IEQ Index.",
    type: "website",
    url: "https://github.com/NibrasAz7/comfio",
  },
  icons: {
    icon: [{ url: "/favicon.ico" }],
  },
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetbrains.variable} font-sans`}>
        <ThemeProvider>
          {/* Cinematic fixed layers */}
          <div id="scroll-progress" />
          <div id="ambient" />
          <div id="glow" />
          <div id="vignette" />
          <div id="grain" />
          <CinematicScroll />
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
