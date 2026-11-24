"use client";

import { useState } from "react";
import Link from "next/link";
import { Github, Thermometer, Menu, X } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";

export function Header() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header
      id="site-header"
      className="fixed top-0 left-0 right-0 z-50 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-950/80 backdrop-blur-md transition-transform duration-300"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold text-lg">
          <Thermometer className="w-5 h-5 text-brand-500" />
          comfio
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm">
          <Link href="/#features" className="hover:text-brand-500 transition-colors">Features</Link>
          <Link href="/#quickstart" className="hover:text-brand-500 transition-colors">Quick Start</Link>
          <Link href="/docs" className="hover:text-brand-500 transition-colors">Docs</Link>
          <Link href="/playground" className="hover:text-brand-500 transition-colors">Playground</Link>
          <a
            href="https://github.com/NibrasAz7/comfio"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 hover:text-brand-500 transition-colors"
          >
            <Github className="w-4 h-4" />
            GitHub
          </a>
        </nav>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <nav className="md:hidden border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 px-4 py-4 space-y-3 text-sm">
          <Link href="/#features" onClick={() => setMobileOpen(false)} className="block hover:text-brand-500">Features</Link>
          <Link href="/#quickstart" onClick={() => setMobileOpen(false)} className="block hover:text-brand-500">Quick Start</Link>
          <Link href="/docs" onClick={() => setMobileOpen(false)} className="block hover:text-brand-500">Docs</Link>
          <Link href="/playground" onClick={() => setMobileOpen(false)} className="block hover:text-brand-500">Playground</Link>
          <a
            href="https://github.com/NibrasAz7/comfio"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 hover:text-brand-500"
          >
            <Github className="w-4 h-4" /> GitHub
          </a>
        </nav>
      )}
    </header>
  );
}
