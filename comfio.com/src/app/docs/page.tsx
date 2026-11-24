import Link from "next/link";
import { docsNav } from "@/lib/docs";
import { BookOpen, Wrench, FlaskConical, Code2, ArrowRight } from "lucide-react";

const sectionIcons = [BookOpen, Wrench, FlaskConical, Code2];

export default function DocsPage() {
  return (
    <div>
      <h1 className="text-3xl font-bold mb-4">Documentation</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        Explore comfio through four lenses, following the{" "}
        <a
          href="https://diataxis.fr/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-brand-500 hover:underline"
        >
          Diátaxis
        </a>{" "}
        framework.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {docsNav.map((section, i) => {
          const Icon = sectionIcons[i] || BookOpen;
          return (
            <Link
              key={section.title}
              href={section.href}
              className="group p-6 rounded-xl border border-gray-200 dark:border-gray-800 hover:border-brand-400 dark:hover:border-brand-600 transition-all"
            >
              <div className="flex items-center gap-3 mb-3">
                <Icon className="w-6 h-6 text-brand-500" />
                <h2 className="text-xl font-semibold">{section.title}</h2>
              </div>
              <ul className="space-y-1.5 mb-4">
                {section.links.slice(0, 3).map((link) => (
                  <li key={link.slug} className="text-sm text-gray-600 dark:text-gray-400">
                    • {link.title}
                  </li>
                ))}
                {section.links.length > 3 && (
                  <li className="text-sm text-gray-400">
                    • +{section.links.length - 3} more...
                  </li>
                )}
              </ul>
              <span className="inline-flex items-center gap-1 text-sm text-brand-500 group-hover:gap-2 transition-all">
                Explore <ArrowRight className="w-4 h-4" />
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
