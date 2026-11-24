import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { docsContent, type SectionContent } from "@/content/docs";
import type { DocSection } from "@/lib/docs";
import { MarkdownRenderer } from "@/components/docs/markdown-renderer";
import { CopyButton } from "@/components/copy-button";

interface DocPageViewProps {
  slugParts: string[];
  sectionKey: string;
  sectionNav: DocSection;
  sectionTitle: string;
  sectionHref: string;
}

export function generateDocStaticParams(sectionNav: DocSection) {
  return sectionNav.links.map((link) => ({
    slug: link.slug.split("/"),
  }));
}

export function DocPageView({
  slugParts,
  sectionKey,
  sectionNav,
  sectionTitle,
  sectionHref,
}: DocPageViewProps) {
  const slugStr = slugParts.join("/");
  const content = (docsContent as SectionContent)[sectionKey]?.[slugStr];

  if (!content) {
    notFound();
  }

  const currentIndex = sectionNav.links.findIndex((l) => l.slug === slugStr);
  const prev = currentIndex > 0 ? sectionNav.links[currentIndex - 1] : null;
  const next = currentIndex < sectionNav.links.length - 1 ? sectionNav.links[currentIndex + 1] : null;
  const current = currentIndex >= 0 ? sectionNav.links[currentIndex] : null;

  // Handle Jupyter notebook content (JSON)
  let renderContent = content;
  if (content.trim().startsWith("{")) {
    try {
      const nb = JSON.parse(content);
      const cells: string[] = [];
      for (const cell of nb.cells || []) {
        if (cell.cell_type === "markdown") {
          cells.push((cell.source || []).join(""));
        } else if (cell.cell_type === "code") {
          const code = (cell.source || []).join("");
          cells.push("```python\n" + code + "\n```");
        }
      }
      renderContent = cells.join("\n\n");
    } catch {
      // If JSON parse fails, show raw
    }
  }

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-6">
        <Link href="/docs" className="hover:text-brand-500">Docs</Link>
        <span>/</span>
        <Link href={sectionHref} className="hover:text-brand-500">{sectionTitle}</Link>
        {current && (
          <>
            <span>/</span>
            <span className="text-gray-700 dark:text-gray-300">{current.title}</span>
          </>
        )}
      </div>

      {/* Content */}
      <MarkdownRenderer content={renderContent} />

      {/* Copy full page */}
      <div className="mt-8 flex items-center justify-between border-t border-gray-200 dark:border-gray-800 pt-4">
        <span className="text-xs text-gray-400">
          Source: docs/{sectionKey}/{slugStr}.md
        </span>
        <CopyButton text={renderContent} />
      </div>

      {/* Pagination */}
      <div className="mt-8 flex items-center justify-between">
        {prev ? (
          <Link
            href={`${sectionHref}/${prev.slug}`}
            className="inline-flex items-center gap-2 text-sm hover:text-brand-500 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link
            href={`${sectionHref}/${next.slug}`}
            className="inline-flex items-center gap-2 text-sm hover:text-brand-500 transition-colors"
          >
            {next.title} <ArrowRight className="w-4 h-4" />
          </Link>
        ) : (
          <span />
        )}
      </div>
    </div>
  );
}
