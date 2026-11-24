import Link from "next/link";
import { docsNav } from "@/lib/docs";

export default function ReferencePage() {
  const section = docsNav.find((s) => s.title === "API Reference")!;
  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">API Reference</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        Auto-generated documentation for every public function, class, and module in comfio.
      </p>
      <div className="space-y-3">
        {section.links.map((link) => (
          <Link
            key={link.slug}
            href={`/docs/reference/${link.slug}`}
            className="group block p-4 rounded-lg border border-gray-200 dark:border-gray-800 hover:border-brand-400 dark:hover:border-brand-600 transition-all"
          >
            <h3 className="font-semibold group-hover:text-brand-500 transition-colors">
              <code className="font-mono text-brand-600 dark:text-brand-400">{link.title}</code>
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-0.5">{link.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
