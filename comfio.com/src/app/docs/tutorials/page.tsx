import Link from "next/link";
import { docsNav } from "@/lib/docs";
import { BookOpen } from "lucide-react";

export default function TutorialsPage() {
  const section = docsNav.find((s) => s.title === "Tutorials")!;
  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">Tutorials</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        Step-by-step Jupyter notebooks covering the full comfio workflow, from basic compliance contracts to advanced ML/DL integration.
      </p>
      <div className="space-y-3">
        {section.links.map((link, i) => (
          <Link
            key={link.slug}
            href={`/docs/tutorials/${link.slug}`}
            className="group block p-4 rounded-lg border border-gray-200 dark:border-gray-800 hover:border-brand-400 dark:hover:border-brand-600 transition-all"
          >
            <div className="flex items-start gap-3">
              <span className="text-brand-500 font-mono text-sm mt-0.5">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <h3 className="font-semibold group-hover:text-brand-500 transition-colors">{link.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-0.5">{link.desc}</p>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
