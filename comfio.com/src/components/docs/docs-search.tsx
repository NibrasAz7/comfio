"use client";

import { useState, useMemo } from "react";
import { Search, X } from "lucide-react";
import { docsNav } from "@/lib/docs";
import Link from "next/link";

interface SearchResult {
  title: string;
  desc: string;
  href: string;
}

export function DocsSearch() {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const results = useMemo<SearchResult[]>(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    const all: SearchResult[] = [];
    for (const section of docsNav) {
      for (const link of section.links) {
        if (
          link.title.toLowerCase().includes(q) ||
          link.desc.toLowerCase().includes(q) ||
          link.slug.toLowerCase().includes(q)
        ) {
          all.push({
            title: link.title,
            desc: link.desc,
            href: `${section.href}/${link.slug}`,
          });
        }
      }
    }
    return all.slice(0, 8);
  }, [query]);

  return (
    <div className="relative">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          placeholder="Search docs..."
          className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        {query && (
          <button
            onClick={() => {
              setQuery("");
              setOpen(false);
            }}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {open && results.length > 0 && (
        <div className="absolute top-full mt-2 w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-lg z-50 max-h-80 overflow-y-auto">
          {results.map((r, i) => (
            <Link
              key={i}
              href={r.href}
              onClick={() => {
                setQuery("");
                setOpen(false);
              }}
              className="block px-4 py-2.5 hover:bg-gray-50 dark:hover:bg-gray-800 border-b border-gray-100 dark:border-gray-800 last:border-0"
            >
              <div className="text-sm font-medium">{r.title}</div>
              <div className="text-xs text-gray-500 truncate">{r.desc}</div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
