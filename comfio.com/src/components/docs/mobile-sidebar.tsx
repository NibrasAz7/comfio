"use client";

import { useState } from "react";
import { Menu, X } from "lucide-react";
import { DocsSidebar } from "@/components/docs/docs-sidebar";
import { DocsSearch } from "@/components/docs/docs-search";

export function MobileSidebar() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(true)}
        className="lg:hidden inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-800 text-sm font-medium mb-4"
      >
        <Menu className="w-4 h-4" /> Contents
      </button>

      {/* Overlay */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 z-50 bg-black/40"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Drawer */}
      <aside
        className={`lg:hidden fixed top-0 left-0 z-50 w-72 h-full bg-white dark:bg-gray-950 shadow-xl overflow-y-auto p-6 transition-transform duration-300 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between mb-6">
          <span className="font-semibold">Documentation</span>
          <button onClick={() => setOpen(false)} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="mb-6">
          <DocsSearch />
        </div>
        <DocsSidebar />
      </aside>
    </>
  );
}
