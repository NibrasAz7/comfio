import Link from "next/link";
import { docsNav } from "@/lib/docs";

export default function TheoryPage() {
  const section = docsNav.find((s) => s.title === "Theory")!;
  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">Theory</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        Understanding-oriented documentation covering the mathematical foundations, standards, and equations behind comfio&apos;s calculations.
      </p>
      <div className="space-y-3">
        {section.links.map((link) => (
          <Link
            key={link.slug}
            href={`/docs/theory/${link.slug}`}
            className="group block p-4 rounded-lg border border-gray-200 dark:border-gray-800 hover:border-brand-400 dark:hover:border-brand-600 transition-all"
          >
            <h3 className="font-semibold group-hover:text-brand-500 transition-colors">{link.title}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-0.5">{link.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
