"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { docsNav } from "@/lib/docs";
import { ChevronRight } from "lucide-react";

export function DocsSidebar() {
  const pathname = usePathname();

  return (
    <nav className="space-y-6">
      {docsNav.map((section) => {
        const sectionActive = pathname?.startsWith(section.href);
        return (
          <div key={section.title}>
            <Link
              href={section.href}
              className={`block text-sm font-semibold mb-2 transition-colors ${
                sectionActive
                  ? "text-brand-600 dark:text-brand-400"
                  : "text-gray-700 dark:text-gray-300 hover:text-brand-500"
              }`}
            >
              {section.title}
            </Link>
            <ul className="space-y-1 ml-3">
              {section.links.map((link) => {
                const href = `${section.href}/${link.slug}`;
                const active = pathname === href;
                return (
                  <li key={link.slug}>
                    <Link
                      href={href}
                      className={`block text-sm py-1 px-2 rounded transition-colors ${
                        active
                          ? "bg-brand-100 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 font-medium"
                          : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                      }`}
                    >
                      {link.title}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </nav>
  );
}
