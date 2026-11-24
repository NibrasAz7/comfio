export interface DocLink {
  slug: string;
  title: string;
  desc: string;
}

export interface DocSection {
  title: string;
  href: string;
  links: DocLink[];
}

export const docsNav: DocSection[] = [
  {
    title: "Tutorials",
    href: "/docs/tutorials",
    links: [
      { slug: "01_first_contract", title: "First Compliance Contract", desc: "Load sensor data → evaluate 4 domains → Global IEQ → compliance report" },
      { slug: "02_new_domains", title: "New Domain Modules", desc: "Pollutant IAQ, sPMV, adaptive comfort, TSV augmentation, personalisation" },
      { slug: "03_training_pcm", title: "Training a Personal Comfort Model", desc: "IEQTimeSeriesDataset → PyTorch DataLoader → train PCM → evaluate" },
      { slug: "04_llm_diagnostics", title: "LLM Diagnostics", desc: "Tool schemas, function calling, markdown summaries, diagnostic prompts" },
      { slug: "05_advanced_domains", title: "Advanced Domain Evaluation", desc: "Daylighting, color quality, reverberation, speech intelligibility" },
    ],
  },
  {
    title: "How-To Guides",
    href: "/docs/how-to",
    links: [
      { slug: "handling_missing_sensor_data", title: "Handling Missing Sensor Data", desc: "NaN strategies, physical bounds, SensorData validation" },
      { slug: "export_to_blockchain_oracle", title: "Export to Blockchain Oracle", desc: "ComplianceReport → JSON → Solidity ABI" },
      { slug: "custom_weighting_schemas", title: "Custom Weighting Schemas", desc: "Preset weights, custom weights, domain-specific presets" },
      { slug: "use_tsv_augmentation", title: "TSV Augmentation", desc: "CDF-based vote expansion workflow" },
      { slug: "train_personalisation", title: "Train Personalisation", desc: "OLS regression personalisation with seasonal support" },
      { slug: "integrate_llm_tools", title: "Integrate LLM Tools", desc: "OpenAI/LangChain tool schemas and diagnostics" },
    ],
  },
  {
    title: "Theory",
    href: "/docs/theory",
    links: [
      { slug: "thermal_pmv_ppd", title: "PMV/PPD Thermal Comfort", desc: "ISO 7730 heat balance equation, PPD relation, category limits" },
      { slug: "adaptive_comfort", title: "Adaptive Comfort Models", desc: "ASHRAE 55-2023 & EN 16798-1:2019 adaptive models" },
      { slug: "spmv_buratti", title: "Simplified PMV (Buratti)", desc: "Seasonal sPMV model requiring only temperature and humidity" },
      { slug: "tsv_cdf_remapping", title: "TSV CDF Remapping", desc: "Quantile mapping for augmenting sparse occupant votes" },
      { slug: "pollutant_iaq_thresholds", title: "Pollutant IAQ Thresholds", desc: "WHO/EPA/WELL threshold tables and piecewise scoring" },
      { slug: "weakest_link_aggregation", title: "Global IEQ Aggregation", desc: "Weighted sum, 50/50 IAQ blend, TSV thermal override" },
      { slug: "co2_decay_ventilation", title: "CO₂ Decay Ventilation", desc: "Exponential decay model for ventilation rate estimation" },
      { slug: "limitations", title: "Limitations & Caveats", desc: "Scientific transparency, applicability ranges, known limits" },
    ],
  },
  {
    title: "API Reference",
    href: "/docs/reference",
    links: [
      { slug: "data_handler", title: "SensorData", desc: "Data ingestion, column auto-detection, validation" },
      { slug: "validation", title: "Validation", desc: "Input array validation utilities" },
      { slug: "domains/thermal", title: "Thermal", desc: "PMV/PPD evaluation (ISO 7730 / ASHRAE 55)" },
      { slug: "domains/thermal_spmv", title: "sPMV", desc: "Simplified PMV (Buratti et al. 2009)" },
      { slug: "domains/thermal_adaptive", title: "Adaptive Thermal", desc: "ASHRAE 55-2023 & EN 16798-1:2019" },
      { slug: "domains/thermal_tsv", title: "TSV", desc: "TSV augmentation and compliance" },
      { slug: "domains/thermal_personal", title: "Personalised Thermal", desc: "OLS regression personalisation" },
      { slug: "domains/visual", title: "Visual", desc: "Illuminance evaluation (EN 12464-1)" },
      { slug: "domains/acoustic", title: "Acoustic", desc: "Noise level evaluation (NC curves)" },
      { slug: "domains/iaq", title: "IAQ", desc: "CO₂-based IAQ evaluation (ASHRAE 62.1)" },
      { slug: "domains/iaq_pollutants", title: "Pollutant IAQ", desc: "PM2.5, PM10, TVOC, formaldehyde, CO" },
      { slug: "integration", title: "Global IEQ", desc: "Multi-domain weighted aggregation" },
      { slug: "performance", title: "Compliance", desc: "Compliance rates, contract JSON generation" },
      { slug: "ml_adapters", title: "ML Adapters", desc: "scikit-learn, PyTorch, TensorFlow/Keras" },
      { slug: "llm_tools", title: "LLM Tools", desc: "OpenAI/LangChain tool schemas, diagnostic prompts" },
    ],
  },
];

export function findDocSection(slug: string): DocSection | undefined {
  for (const section of docsNav) {
    if (section.links.some((l) => l.slug === slug)) {
      return section;
    }
  }
  return undefined;
}

export function findDocLink(slug: string): { link: DocLink; section: DocSection } | undefined {
  for (const section of docsNav) {
    const link = section.links.find((l) => l.slug === slug);
    if (link) return { link, section };
  }
  return undefined;
}
