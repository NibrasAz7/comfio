import Link from "next/link";
import {
  Thermometer, Eye, Volume2, Wind, Activity, Brain,
  FileJson, Cpu, Sparkles, Gauge, BookOpen, ArrowRight,
  Github, Package,
} from "lucide-react";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";
import { CopyButton } from "@/components/copy-button";

const features = [
  { icon: Thermometer, title: "Multi-Domain IEQ", desc: "Thermal, Visual, Acoustic, IAQ unified into a single 0–100 index." },
  { icon: Activity, title: "Time-Series Native", desc: "Built for Pandas/NumPy arrays, not single-point calculations." },
  { icon: FileJson, title: "Performance Contracts", desc: "Compliance rates + structured JSON for blockchain Oracle integration." },
  { icon: Brain, title: "ML/DL Compatible", desc: "Optional adapters for scikit-learn, PyTorch, and TensorFlow/Keras." },
  { icon: Sparkles, title: "Advanced Physics", desc: "Radiance daylighting, CRI/CCT, RT60 reverberation, STI, CO₂ decay." },
  { icon: Wind, title: "Pollutant IAQ", desc: "PM2.5, PM10, TVOC, formaldehyde, CO against WHO/EPA/WELL thresholds." },
  { icon: Gauge, title: "Adaptive Comfort", desc: "ASHRAE 55-2023 and EN 16798-1:2019 adaptive models." },
  { icon: Cpu, title: "sPMV", desc: "Buratti et al. (2009) seasonal simplified PMV model." },
  { icon: Activity, title: "TSV Augmentation", desc: "CDF-based quantile mapping for sparse occupant vote expansion." },
  { icon: Brain, title: "Personalised Comfort", desc: "OLS regression personalisation with per-season support." },
  { icon: Sparkles, title: "LLM-Native", desc: "Diagnostic prompts, OpenAI/LangChain tool schemas, markdown summaries." },
];

const standards = [
  { name: "ISO 7730", domain: "Thermal comfort — PMV/PPD" },
  { name: "ASHRAE 55", domain: "Thermal environmental conditions" },
  { name: "ASHRAE 55-2023 App. L", domain: "TSV compliance (|TSV| ≤ 1.5)" },
  { name: "EN 16798-1:2019", domain: "Adaptive thermal comfort" },
  { name: "EN 12464-1", domain: "Lighting of work places" },
  { name: "ASHRAE 62.1", domain: "Ventilation for acceptable IAQ" },
  { name: "WHO AQG (2021)", domain: "PM2.5, PM10 thresholds" },
  { name: "EPA NAAQS", domain: "Criteria pollutant thresholds" },
  { name: "WELL v2", domain: "Feature A01 pollutant thresholds" },
];

const quickstartCode = `import numpy as np
from comfio import (
    evaluate_thermal, evaluate_visual,
    evaluate_acoustic, evaluate_iaq,
    calculate_global_ieq, calculate_compliance,
)

# Evaluate each domain
thermal = evaluate_thermal(
    tdb=np.array([24.0, 25.0, 26.0]),
    tr=np.array([24.0, 25.0, 26.0]),
    vr=np.array([0.1, 0.1, 0.1]),
    rh=np.array([50.0, 50.0, 50.0]),
    met=1.2, clo=0.5, category="B",
)
visual = evaluate_visual(illuminance=np.array([450.0, 500.0, 600.0]))
acoustic = evaluate_acoustic(laeq=np.array([35.0, 40.0, 45.0]))
iaq = evaluate_iaq(co2=np.array([700.0, 900.0, 1100.0]))

# Merge into a single 0-100 score
ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)
print(f"Global IEQ Index: {ieq.index}")

# Compliance tracking & contract-ready JSON
report = calculate_compliance(ieq, threshold=80.0)
print(f"Compliance rate: {report.compliance_rate_pct:.1f}%")
print(report.to_contract_json())`;

const installCmd = "pip install comfio";

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <Header />

      {/* Hero */}
      <section id="hero" className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-brand-50/50 to-transparent dark:from-brand-950/20" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-brand-100/40 via-transparent to-transparent dark:from-brand-900/10" />

        <div className="relative max-w-7xl mx-auto text-center">
          <div data-hero-badge className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-100 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 text-sm font-medium mb-6">
            <Package className="w-4 h-4" />
            v0.1.1 on PyPI
          </div>

          <h1 data-hero-title className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
            <span>The story of{" "}</span>
            <span className="bg-gradient-to-r from-brand-500 to-brand-700 bg-clip-text text-transparent">
              IEQ data
            </span>
            <span>, unified.</span>
          </h1>

          <p data-hero-desc className="text-lg sm:text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-8">
            A multi-domain IEQ &amp; performance contract framework for smart buildings.
            Unifies Thermal, Visual, Acoustic, and IAQ metrics into a single Global IEQ Index.
          </p>

          <div data-hero-actions className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-gray-100 dark:bg-gray-800 font-mono text-sm">
              <span className="text-gray-500">$</span>
              <span>{installCmd}</span>
              <CopyButton text={installCmd} />
            </div>
            <Link
              href="/docs"
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-brand-600 text-white hover:bg-brand-700 transition-colors font-medium"
            >
              Read the docs <ArrowRight className="w-4 h-4" />
            </Link>
          </div>

          {/* Domain icons row */}
          <div data-hero-domains className="flex items-center justify-center gap-8 sm:gap-12 text-gray-400">
            <div className="flex flex-col items-center gap-1.5">
              <Thermometer className="w-6 h-6" />
              <span className="text-xs">Thermal</span>
            </div>
            <div className="flex flex-col items-center gap-1.5">
              <Eye className="w-6 h-6" />
              <span className="text-xs">Visual</span>
            </div>
            <div className="flex flex-col items-center gap-1.5">
              <Volume2 className="w-6 h-6" />
              <span className="text-xs">Acoustic</span>
            </div>
            <div className="flex flex-col items-center gap-1.5">
              <Wind className="w-6 h-6" />
              <span className="text-xs">IAQ</span>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Start */}
      <section id="quickstart" className="py-20 px-4 sm:px-6 lg:px-8">
        <div data-reveal className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">30-Second Quickstart</h2>
          <p className="text-center text-gray-600 dark:text-gray-400 mb-8">
            From raw sensor data to a compliance contract in 15 lines of Python.
          </p>

          <div className="relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800">
            <div className="flex items-center justify-between px-4 py-2.5 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
              <span className="text-xs text-gray-500 font-mono">quickstart.py</span>
              <CopyButton text={quickstartCode} />
            </div>
            <pre className="p-4 overflow-x-auto text-sm bg-gray-50 dark:bg-gray-900 leading-relaxed">
              <code className="font-mono text-gray-800 dark:text-gray-200">{quickstartCode}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-gray-900/50">
        <div data-reveal className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">Features</h2>
          <p className="text-center text-gray-600 dark:text-gray-400 mb-12">
            Everything you need to turn building sensor data into actionable intelligence.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <div
                key={i}
                data-feature-card
                className="p-6 rounded-xl bg-white dark:bg-gray-800/50 border border-gray-200 dark:border-gray-800 hover:border-brand-300 dark:hover:border-brand-700 transition-colors"
              >
                <f.icon className="w-8 h-8 text-brand-500 mb-3" />
                <h3 className="font-semibold text-lg mb-1">{f.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Standards */}
      <section id="standards" className="py-20 px-4 sm:px-6 lg:px-8">
        <div data-reveal className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">Standards Referenced</h2>
          <p className="text-center text-gray-600 dark:text-gray-400 mb-8">
            comfio implements calculations from internationally recognized standards.
          </p>

          <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 dark:bg-gray-900">
                  <th className="text-left px-4 py-3 font-semibold">Standard</th>
                  <th className="text-left px-4 py-3 font-semibold">Domain</th>
                </tr>
              </thead>
              <tbody>
                {standards.map((s, i) => (
                  <tr
                    key={i}
                    className="border-t border-gray-200 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900/50"
                  >
                    <td className="px-4 py-3 font-mono text-brand-600 dark:text-brand-400">{s.name}</td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400">{s.domain}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Docs CTA */}
      <section id="docs-cta" className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-gray-900/50">
        <div data-reveal className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">Explore the Documentation</h2>
          <p className="text-center text-gray-600 dark:text-gray-400 mb-12">
            Four lenses to understand and use comfio effectively.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { title: "Tutorials", desc: "Step-by-step Jupyter notebooks", href: "/docs/tutorials", icon: BookOpen },
              { title: "How-To Guides", desc: "Practical, copy-pasteable solutions", href: "/docs/how-to", icon: ArrowRight },
              { title: "Theory", desc: "Mathematical background & equations", href: "/docs/theory", icon: Sparkles },
              { title: "API Reference", desc: "Auto-generated function docs", href: "/docs/reference", icon: Cpu },
            ].map((d, i) => (
              <Link
                key={i}
                href={d.href}
                className="group p-6 rounded-xl bg-white dark:bg-gray-800/50 border border-gray-200 dark:border-gray-800 hover:border-brand-400 dark:hover:border-brand-600 transition-all"
              >
                <d.icon className="w-8 h-8 text-brand-500 mb-3 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold text-lg mb-1">{d.title}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{d.desc}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section id="final-cta" className="py-20 px-4 sm:px-6 lg:px-8">
        <div data-reveal className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Get started in seconds</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            Install comfio, evaluate your first IEQ index, and generate a compliance contract.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-gray-100 dark:bg-gray-800 font-mono text-sm">
              <span className="text-gray-500">$</span>
              <span>pip install comfio[all]</span>
              <CopyButton text="pip install comfio[all]" />
            </div>
            <a
              href="https://github.com/NibrasAz7/comfio"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg border border-gray-300 dark:border-gray-700 hover:border-brand-400 transition-colors font-medium"
            >
              <Github className="w-4 h-4" /> Star on GitHub
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
