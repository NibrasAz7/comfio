"use client";

import { CopyButton } from "@/components/copy-button";
import { useMemo } from "react";
import { InlineMath, BlockMath } from "react-katex";

interface MarkdownRendererProps {
  content: string;
}

interface Block {
  type: "heading" | "paragraph" | "code" | "math" | "list" | "table" | "hr" | "blockquote" | "directive";
  level?: number;
  text?: string;
  lang?: string;
  items?: string[];
  ordered?: boolean;
  rows?: string[][];
  header?: string[];
}

function parseMarkdown(md: string): Block[] {
  const lines = md.split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Skip empty lines
    if (!line.trim()) {
      i++;
      continue;
    }

    // mkdocstrings directive (::: module.path)
    if (line.trim().startsWith(":::")) {
      blocks.push({ type: "directive", text: line.trim().replace(":::", "").trim() });
      i++;
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        text: headingMatch[2],
      });
      i++;
      continue;
    }

    // Horizontal rule
    if (/^---+$/.test(line.trim()) || /^\*\*\*+$/.test(line.trim())) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }

    // Block math ($$...$$)
    if (line.trim().startsWith("$$")) {
      const mathLines: string[] = [];
      if (line.trim().endsWith("$$") && line.trim().length > 4) {
        // Single-line math
        blocks.push({
          type: "math",
          text: line.trim().slice(2, -2).trim(),
        });
        i++;
        continue;
      }
      i++;
      while (i < lines.length && !lines[i].trim().endsWith("$$")) {
        mathLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) {
        i++; // skip closing $$
      }
      blocks.push({ type: "math", text: mathLines.join("\n").trim() });
      continue;
    }

    // Code blocks (```)
    if (line.trim().startsWith("```")) {
      const lang = line.trim().replace("```", "").trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) {
        i++; // skip closing ```
      }
      blocks.push({ type: "code", lang: lang || "text", text: codeLines.join("\n") });
      continue;
    }

    // Blockquote
    if (line.trim().startsWith(">")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      blocks.push({ type: "blockquote", text: quoteLines.join("\n") });
      continue;
    }

    // Tables
    if (line.includes("|") && i + 1 < lines.length && lines[i + 1].includes("---")) {
      const header = line.split("|").map((c) => c.trim()).filter(Boolean);
      i += 2; // skip header + separator
      const rows: string[][] = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(lines[i].split("|").map((c) => c.trim()).filter(Boolean));
        i++;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    // Lists
    if (/^[-*]\s+/.test(line.trim()) || /^\d+\.\s+/.test(line.trim())) {
      const items: string[] = [];
      const ordered = /^\d+\.\s+/.test(line.trim());
      while (i < lines.length && (/^[-*]\s+/.test(lines[i].trim()) || /^\d+\.\s+/.test(lines[i].trim()))) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, "").replace(/^\d+\.\s+/, ""));
        i++;
      }
      blocks.push({ type: "list", items, ordered });
      continue;
    }

    // Paragraph (collect consecutive non-empty, non-special lines)
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("$$") &&
      !lines[i].trim().startsWith(">") &&
      !lines[i].trim().startsWith("---") &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "paragraph", text: paraLines.join(" ") });
    } else {
      i++;
    }
  }

  return blocks;
}

function renderInline(text: string): React.ReactNode {
  // Handle inline code, bold, italic, links, and inline math
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let keyCounter = 0;

  while (remaining) {
    // Inline code
    const codeMatch = remaining.match(/`([^`]+)`/);
    // Bold
    const boldMatch = remaining.match(/\*\*([^*]+)\*\*/);
    // Italic
    const italicMatch = remaining.match(/\*([^*]+)\*/);
    // Link
    const linkMatch = remaining.match(/\[([^\]]+)\]\(([^)]+)\)/);
    // Inline math
    const mathMatch = remaining.match(/\$([^$]+)\$/);

    const matches = [
      codeMatch ? { type: "code", match: codeMatch, index: codeMatch.index! } : null,
      boldMatch ? { type: "bold", match: boldMatch, index: boldMatch.index! } : null,
      italicMatch ? { type: "italic", match: italicMatch, index: italicMatch.index! } : null,
      linkMatch ? { type: "link", match: linkMatch, index: linkMatch.index! } : null,
      mathMatch ? { type: "math", match: mathMatch, index: mathMatch.index! } : null,
    ].filter(Boolean) as { type: string; match: RegExpMatchArray; index: number }[];

    if (matches.length === 0) {
      parts.push(remaining);
      break;
    }

    // Find earliest match
    matches.sort((a, b) => a.index - b.index);
    const earliest = matches[0];

    if (earliest.index > 0) {
      parts.push(remaining.slice(0, earliest.index));
    }

    const key = `inline-${keyCounter++}`;

    switch (earliest.type) {
      case "code":
        parts.push(
          <code key={key} className="px-1.5 py-0.5 rounded text-sm bg-gray-100 dark:bg-gray-800 text-brand-600 dark:text-brand-400 font-mono">
            {earliest.match[1]}
          </code>
        );
        break;
      case "bold":
        parts.push(<strong key={key} className="font-semibold">{earliest.match[1]}</strong>);
        break;
      case "italic":
        parts.push(<em key={key}>{earliest.match[1]}</em>);
        break;
      case "link":
        parts.push(
          <a key={key} href={earliest.match[2]} className="text-brand-500 hover:underline" target="_blank" rel="noopener noreferrer">
            {earliest.match[1]}
          </a>
        );
        break;
      case "math":
        parts.push(
          <span key={key}>
            <InlineMath math={earliest.match[1]} />
          </span>
        );
        break;
    }

    remaining = remaining.slice(earliest.index + earliest.match[0].length);
  }

  return parts;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const blocks = useMemo(() => parseMarkdown(content), [content]);

  return (
    <div className="space-y-4">
      {blocks.map((block, i) => {
        const key = `block-${i}`;

        switch (block.type) {
          case "heading":
            const Tag = `h${block.level}` as keyof React.JSX.IntrinsicElements;
            const sizes: Record<number, string> = {
              1: "text-3xl font-bold mb-4 mt-8",
              2: "text-2xl font-semibold mb-3 mt-6",
              3: "text-xl font-semibold mb-2 mt-4",
              4: "text-lg font-medium mb-2 mt-3",
              5: "text-base font-medium mb-1 mt-2",
              6: "text-sm font-medium mb-1 mt-2",
            };
            return (
              <Tag key={key} className={sizes[block.level || 6]}>
                {renderInline(block.text || "")}
              </Tag>
            );

          case "paragraph":
            return (
              <p key={key} className="text-gray-700 dark:text-gray-300 leading-relaxed">
                {renderInline(block.text || "")}
              </p>
            );

          case "code":
            return (
              <div key={key} className="relative rounded-lg overflow-hidden border border-gray-200 dark:border-gray-800 my-4">
                <div className="flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
                  <span className="text-xs text-gray-500 font-mono">{block.lang}</span>
                  <CopyButton text={block.text || ""} />
                </div>
                <pre className="p-4 overflow-x-auto text-sm bg-gray-50 dark:bg-gray-900">
                  <code className="font-mono text-gray-800 dark:text-gray-200">{block.text}</code>
                </pre>
              </div>
            );

          case "math":
            return (
              <div key={key} className="my-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-x-auto text-center">
                <BlockMath math={block.text || ""} />
              </div>
            );

          case "directive":
            return (
              <div key={key} className="my-4 p-4 rounded-lg border border-brand-200 dark:border-brand-800 bg-brand-50 dark:bg-brand-900/20">
                <div className="flex items-center gap-2 mb-2">
                  <code className="text-sm font-mono text-brand-600 dark:text-brand-400 font-semibold">
                    {block.text}
                  </code>
                </div>
                <p className="text-xs text-gray-500">
                  This is an auto-generated API reference. See the{" "}
                  <a
                    href={`https://github.com/NibrasAz7/comfio/blob/main/src/${(block.text || "").replace(/\./g, "/")}.py`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-brand-500 hover:underline"
                  >
                    source code on GitHub
                  </a>{" "}
                  for full details.
                </p>
              </div>
            );

          case "list":
            const ListTag = block.ordered ? "ol" : "ul";
            return (
              <ListTag key={key} className={`space-y-1 ${block.ordered ? "list-decimal" : "list-disc"} ml-6 text-gray-700 dark:text-gray-300`}>
                {block.items?.map((item, j) => (
                  <li key={`${key}-${j}`}>{renderInline(item)}</li>
                ))}
              </ListTag>
            );

          case "table":
            return (
              <div key={key} className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800 my-4">
                <table className="w-full text-sm">
                  {block.header && (
                    <thead>
                      <tr className="bg-gray-50 dark:bg-gray-900">
                        {block.header.map((h, j) => (
                          <th key={j} className="text-left px-4 py-2 font-semibold border-b border-gray-200 dark:border-gray-800">
                            {renderInline(h)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                  )}
                  <tbody>
                    {block.rows?.map((row, j) => (
                      <tr key={j} className="border-b border-gray-100 dark:border-gray-800 last:border-0">
                        {row.map((cell, k) => (
                          <td key={k} className="px-4 py-2 text-gray-700 dark:text-gray-300">
                            {renderInline(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );

          case "hr":
            return <hr key={key} className="border-gray-200 dark:border-gray-800 my-6" />;

          case "blockquote":
            return (
              <blockquote key={key} className="border-l-4 border-brand-400 pl-4 py-2 my-4 text-gray-600 dark:text-gray-400 italic">
                {renderInline(block.text || "")}
              </blockquote>
            );

          default:
            return null;
        }
      })}
    </div>
  );
}
