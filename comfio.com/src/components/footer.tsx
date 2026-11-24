import Link from "next/link";
import { Github, Thermometer } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="md:col-span-2">
          <div className="flex items-center gap-2 font-semibold text-lg mb-2">
            <Thermometer className="w-5 h-5 text-brand-500" />
            comfio
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 max-w-md">
            Multi-domain IEQ &amp; performance contract framework for smart buildings.
            Unifies Thermal, Visual, Acoustic, and IAQ metrics into a single Global IEQ Index.
          </p>
        </div>

        <div>
          <h3 className="font-semibold text-sm mb-3">Resources</h3>
          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <li><Link href="/docs" className="hover:text-brand-500">Documentation</Link></li>
            <li><Link href="/docs/tutorials" className="hover:text-brand-500">Tutorials</Link></li>
            <li><Link href="/playground" className="hover:text-brand-500">Playground</Link></li>
            <li><a href="https://pypi.org/project/comfio/" className="hover:text-brand-500">PyPI</a></li>
          </ul>
        </div>

        <div>
          <h3 className="font-semibold text-sm mb-3">Community</h3>
          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <li>
              <a href="https://github.com/NibrasAz7/comfio" className="flex items-center gap-1.5 hover:text-brand-500">
                <Github className="w-4 h-4" /> GitHub
              </a>
            </li>
            <li><a href="https://github.com/NibrasAz7/comfio/issues" className="hover:text-brand-500">Issues</a></li>
            <li><a href="https://github.com/NibrasAz7/comfio/blob/main/LICENSE" className="hover:text-brand-500">MIT License</a></li>
          </ul>
        </div>
      </div>

      <div className="max-w-7xl mx-auto mt-8 pt-8 border-t border-gray-200 dark:border-gray-800 text-center text-sm text-gray-500">
        <p>&copy; 2025 comfio Contributors. Released under MIT.</p>
      </div>
    </footer>
  );
}
