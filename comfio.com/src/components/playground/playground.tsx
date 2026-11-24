"use client";

import { useState, useMemo, useCallback } from "react";
import { Thermometer, Eye, Volume2, Wind, Gauge, RotateCcw } from "lucide-react";
import { CopyButton } from "@/components/copy-button";

interface DomainScore {
  score: number;
  label: string;
  color: string;
}

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function thermalScore(tdb: number, rh: number, vel: number, clo: number, met: number): DomainScore {
  // Simplified PMV approximation for demo purposes
  const pmvApprox = 0.303 * Math.exp(-0.036 * met) * (met - 0.5) * (tdb - 22) + 0.1 * (rh - 50) * 0.01 - 0.1 * (vel - 0.1) * 10;
  const pmv = clamp(pmvApprox, -3, 3);
  const score = clamp(100 - Math.abs(pmv) * 35, 0, 100);
  const label = score >= 80 ? "Comfortable" : score >= 60 ? "Slightly uncomfortable" : "Uncomfortable";
  const color = score >= 80 ? "text-green-500" : score >= 60 ? "text-yellow-500" : "text-red-500";
  return { score: Math.round(score), label, color };
}

function visualScore(lux: number): DomainScore {
  const score = clamp(100 - Math.abs(lux - 500) / 5, 0, 100);
  const label = lux >= 400 && lux <= 600 ? "Well lit" : lux < 300 ? "Too dim" : "Too bright";
  const color = score >= 80 ? "text-green-500" : score >= 60 ? "text-yellow-500" : "text-red-500";
  return { score: Math.round(score), label, color };
}

function acousticScore(db: number): DomainScore {
  const score = clamp(100 - (db - 35) * 2.5, 0, 100);
  const label = db <= 40 ? "Quiet" : db <= 55 ? "Moderate" : "Noisy";
  const color = score >= 80 ? "text-green-500" : score >= 60 ? "text-yellow-500" : "text-red-500";
  return { score: Math.round(score), label, color };
}

function iaqScore(co2: number): DomainScore {
  const score = clamp(100 - (co2 - 400) * 0.08, 0, 100);
  const label = co2 <= 800 ? "Good" : co2 <= 1000 ? "Moderate" : "Poor";
  const color = score >= 80 ? "text-green-500" : score >= 60 ? "text-yellow-500" : "text-red-500";
  return { score: Math.round(score), label, color };
}

const weights = { thermal: 0.4, visual: 0.2, acoustic: 0.15, iaq: 0.25 };

function ScoreGauge({ value }: { value: number }) {
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const strokeColor = value >= 80 ? "#22c55e" : value >= 60 ? "#eab308" : "#ef4444";

  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r={radius} fill="none" stroke="currentColor" strokeWidth="12" className="text-gray-200 dark:text-gray-800" />
        <circle
          cx="100" cy="100" r={radius} fill="none" stroke={strokeColor} strokeWidth="12"
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.5s ease, stroke 0.3s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold">{Math.round(value)}</span>
        <span className="text-xs text-gray-500 mt-1">Global IEQ</span>
      </div>
    </div>
  );
}

interface SliderProps {
  label: string;
  icon: React.ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}

function Slider({ label, icon, value, min, max, step, unit, onChange }: SliderProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="flex items-center gap-2 text-sm font-medium">
          {icon}
          {label}
        </label>
        <span className="text-sm font-mono text-brand-600 dark:text-brand-400">
          {value.toFixed(step < 1 ? 1 : 0)}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-brand-500 cursor-pointer"
      />
    </div>
  );
}

function DomainCard({ title, icon, score }: { title: string; icon: React.ReactNode; score: DomainScore }) {
  return (
    <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800/50">
      <div className="flex items-center justify-between mb-2">
        <span className="flex items-center gap-2 text-sm font-medium">
          {icon}
          {title}
        </span>
        <span className={`text-lg font-bold ${score.color}`}>{score.score}</span>
      </div>
      <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${score.score}%`,
            backgroundColor: score.score >= 80 ? "#22c55e" : score.score >= 60 ? "#eab308" : "#ef4444",
          }}
        />
      </div>
      <p className={`text-xs mt-1.5 ${score.color}`}>{score.label}</p>
    </div>
  );
}

const defaultParams = { tdb: 24, rh: 50, vel: 0.1, clo: 0.5, met: 1.2, lux: 500, db: 35, co2: 400 };

export function Playground() {
  const [p, setP] = useState(defaultParams);

  const update = useCallback((key: keyof typeof defaultParams, v: number) => {
    setP((prev) => ({ ...prev, [key]: v }));
  }, []);

  const reset = useCallback(() => setP(defaultParams), []);

  const thermal = useMemo(() => thermalScore(p.tdb, p.rh, p.vel, p.clo, p.met), [p.tdb, p.rh, p.vel, p.clo, p.met]);
  const visual = useMemo(() => visualScore(p.lux), [p.lux]);
  const acoustic = useMemo(() => acousticScore(p.db), [p.db]);
  const iaq = useMemo(() => iaqScore(p.co2), [p.co2]);

  const globalIeq = useMemo(() => {
    return Math.round(
      thermal.score * weights.thermal +
      visual.score * weights.visual +
      acoustic.score * weights.acoustic +
      iaq.score * weights.iaq
    );
  }, [thermal, visual, acoustic, iaq]);

  const codeSnippet = useMemo(() => {
    return `from comfio import (
    evaluate_thermal, evaluate_visual,
    evaluate_acoustic, evaluate_iaq,
    calculate_global_ieq,
)
import numpy as np

thermal = evaluate_thermal(
    tdb=np.array([${p.tdb}]), tr=np.array([${p.tdb}]),
    vr=np.array([${p.vel}]), rh=np.array([${p.rh}]),
    met=${p.met}, clo=${p.clo}, category="B",
)
visual = evaluate_visual(illuminance=np.array([${p.lux}]))
acoustic = evaluate_acoustic(laeq=np.array([${p.db}]))
iaq = evaluate_iaq(co2=np.array([${p.co2}]))

ieq = calculate_global_ieq(
    thermal=thermal, visual=visual,
    acoustic=acoustic, iaq=iaq,
)
print(f"Global IEQ Index: {ieq.index}")`;
  }, [p]);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Controls */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Gauge className="w-5 h-5 text-brand-500" />
              Input Parameters
            </h3>
            <button
              onClick={reset}
              className="inline-flex items-center gap-1.5 text-xs text-gray-500 hover:text-brand-500 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          </div>

          {/* Thermal */}
          <div className="space-y-3 p-4 rounded-lg border border-gray-200 dark:border-gray-800">
            <h4 className="text-sm font-semibold text-brand-600 dark:text-brand-400 flex items-center gap-2">
              <Thermometer className="w-4 h-4" /> Thermal
            </h4>
            <Slider label="Air temp" icon={<Thermometer className="w-3.5 h-3.5 text-gray-400" />} value={p.tdb} min={10} max={40} step={0.5} unit="°C" onChange={(v) => update("tdb", v)} />
            <Slider label="Humidity" icon={<Wind className="w-3.5 h-3.5 text-gray-400" />} value={p.rh} min={0} max={100} step={1} unit="%" onChange={(v) => update("rh", v)} />
            <Slider label="Air velocity" icon={<Wind className="w-3.5 h-3.5 text-gray-400" />} value={p.vel} min={0} max={1} step={0.05} unit="m/s" onChange={(v) => update("vel", v)} />
            <Slider label="Clothing" icon={<Thermometer className="w-3.5 h-3.5 text-gray-400" />} value={p.clo} min={0} max={2} step={0.1} unit="clo" onChange={(v) => update("clo", v)} />
            <Slider label="Metabolic" icon={<Thermometer className="w-3.5 h-3.5 text-gray-400" />} value={p.met} min={0.8} max={3} step={0.1} unit="met" onChange={(v) => update("met", v)} />
          </div>

          {/* Visual */}
          <div className="space-y-3 p-4 rounded-lg border border-gray-200 dark:border-gray-800">
            <h4 className="text-sm font-semibold text-brand-600 dark:text-brand-400 flex items-center gap-2">
              <Eye className="w-4 h-4" /> Visual
            </h4>
            <Slider label="Illuminance" icon={<Eye className="w-3.5 h-3.5 text-gray-400" />} value={p.lux} min={0} max={2000} step={10} unit="lx" onChange={(v) => update("lux", v)} />
          </div>

          {/* Acoustic */}
          <div className="space-y-3 p-4 rounded-lg border border-gray-200 dark:border-gray-800">
            <h4 className="text-sm font-semibold text-brand-600 dark:text-brand-400 flex items-center gap-2">
              <Volume2 className="w-4 h-4" /> Acoustic
            </h4>
            <Slider label="Noise level" icon={<Volume2 className="w-3.5 h-3.5 text-gray-400" />} value={p.db} min={25} max={80} step={1} unit="dB" onChange={(v) => update("db", v)} />
          </div>

          {/* IAQ */}
          <div className="space-y-3 p-4 rounded-lg border border-gray-200 dark:border-gray-800">
            <h4 className="text-sm font-semibold text-brand-600 dark:text-brand-400 flex items-center gap-2">
              <Wind className="w-4 h-4" /> IAQ
            </h4>
            <Slider label="CO₂" icon={<Wind className="w-3.5 h-3.5 text-gray-400" />} value={p.co2} min={400} max={2000} step={10} unit="ppm" onChange={(v) => update("co2", v)} />
          </div>
        </div>

        {/* Results */}
        <div className="space-y-6">
          <h3 className="text-lg font-semibold">Results</h3>

          {/* Global IEQ Gauge */}
          <div className="p-6 rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800/50">
            <ScoreGauge value={globalIeq} />
            <p className="text-center text-sm text-gray-500 mt-3">
              {globalIeq >= 80 ? "Excellent — compliant (Category B)" : globalIeq >= 60 ? "Acceptable — borderline" : "Poor — non-compliant"}
            </p>
          </div>

          {/* Domain scores */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <DomainCard title="Thermal" icon={<Thermometer className="w-4 h-4 text-brand-500" />} score={thermal} />
            <DomainCard title="Visual" icon={<Eye className="w-4 h-4 text-brand-500" />} score={visual} />
            <DomainCard title="Acoustic" icon={<Volume2 className="w-4 h-4 text-brand-500" />} score={acoustic} />
            <DomainCard title="IAQ" icon={<Wind className="w-4 h-4 text-brand-500" />} score={iaq} />
          </div>

          {/* Weights info */}
          <div className="text-xs text-gray-500 p-3 rounded-lg bg-gray-50 dark:bg-gray-900/50">
            <p className="font-semibold mb-1">Weighting schema (default):</p>
            <p>Thermal 40% · IAQ 25% · Visual 20% · Acoustic 15%</p>
          </div>

          {/* Code snippet */}
          <div className="relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800">
            <div className="flex items-center justify-between px-4 py-2.5 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
              <span className="text-xs text-gray-500 font-mono">playground.py</span>
              <CopyButton text={codeSnippet} />
            </div>
            <pre className="p-4 overflow-x-auto text-xs bg-gray-50 dark:bg-gray-900 leading-relaxed max-h-64">
              <code className="font-mono text-gray-800 dark:text-gray-200">{codeSnippet}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
