"""Pydantic-wrapped tool schemas for LLM function calling.

Provides typed input models and executable tool functions that wrap
comfio's core domain evaluations for use with OpenAI function calling,
LangChain, Ollama, or any framework that accepts Pydantic schemas.

Requires the ``[agent]`` extra: ``pip install comfio[agent]``
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.iaq_pollutants import evaluate_iaq_pollutants
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.thermal_adaptive import evaluate_adaptive_ashrae, evaluate_adaptive_en
from comfio.domains.thermal_spmv import evaluate_spmv
from comfio.domains.thermal_tsv import evaluate_tsv
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import calculate_global_ieq
from comfio.llm.interpreters import ieq_to_markdown, ieq_to_summary_dict


class ThermalComfortInput:
    """Input schema for the thermal comfort evaluation tool.

    Attributes
    ----------
    air_temperature : list[float]
        Dry-bulb air temperatures in Celsius.
    relative_humidity : list[float]
        Relative humidity percentages (0-100).
    mean_radiant_temperature : list[float] or None
        Optional radiant temperatures. Defaults to match air temperature.
    air_velocity : float
        Air velocity in m/s. Default 0.1 (stagnant indoor air).
    metabolic_rate : float
        Metabolic rate in met. Default 1.2 (sedentary office work).
    clothing_insulation : float
        Clothing insulation in clo. Default 0.5 (light summer clothing).
    """

    def __init__(
        self,
        air_temperature: list[float],
        relative_humidity: list[float],
        mean_radiant_temperature: list[float] | None = None,
        air_velocity: float = 0.1,
        metabolic_rate: float = 1.2,
        clothing_insulation: float = 0.5,
    ) -> None:
        if not air_temperature:
            raise ValueError("air_temperature cannot be empty")
        if not relative_humidity:
            raise ValueError("relative_humidity cannot be empty")
        self.air_temperature = air_temperature
        self.relative_humidity = relative_humidity
        self.mean_radiant_temperature = mean_radiant_temperature
        self.air_velocity = air_velocity
        self.metabolic_rate = metabolic_rate
        self.clothing_insulation = clothing_insulation

    def to_pydantic(self) -> Any:
        """Convert to a Pydantic BaseModel instance if pydantic is installed."""
        return _ThermalComfortInputModel(
            air_temperature=self.air_temperature,
            relative_humidity=self.relative_humidity,
            mean_radiant_temperature=self.mean_radiant_temperature,
            air_velocity=self.air_velocity,
            metabolic_rate=self.metabolic_rate,
            clothing_insulation=self.clothing_insulation,
        )


def _try_import_pydantic() -> Any:
    """Import pydantic lazily, raising a helpful error if not installed."""
    try:
        import pydantic

        return pydantic
    except ImportError as exc:
        raise ImportError(
            "Pydantic is required for LLM tool schemas. Install with: pip install comfio[agent]"
        ) from exc


def _build_pydantic_models() -> None:
    """Build Pydantic model classes dynamically (called once on first import)."""
    pydantic = _try_import_pydantic()
    BaseModel = pydantic.BaseModel  # noqa: N806
    Field = pydantic.Field  # noqa: N806

    global _ThermalComfortInputModel, _IEQEvaluationInputModel
    global _PollutantIAQInputModel, _SPMVInputModel, _AdaptiveInputModel, _TSVInputModel

    class _ThermalComfortInputModel(BaseModel):
        air_temperature: list[float] = Field(
            ..., description="Dry-bulb air temperatures in Celsius from space sensors."
        )
        relative_humidity: list[float] = Field(
            ..., description="Relative humidity percentages (0-100)."
        )
        mean_radiant_temperature: list[float] | None = Field(
            None, description="Optional radiant temperatures. Defaults to match air temperature."
        )
        air_velocity: float = Field(0.1, description="Air velocity in m/s. Default 0.1.")
        metabolic_rate: float = Field(1.2, description="Metabolic rate in met. Default 1.2.")
        clothing_insulation: float = Field(
            0.5, description="Clothing insulation in clo. Default 0.5."
        )

    class _IEQEvaluationInputModel(BaseModel):
        air_temperature: list[float] = Field(..., description="Dry-bulb air temperatures in °C.")
        relative_humidity: list[float] = Field(..., description="Relative humidity (0-100%).")
        mean_radiant_temperature: list[float] | None = Field(
            None, description="Optional radiant temperatures. Defaults to air temperature."
        )
        air_velocity: float = Field(0.1, description="Air velocity in m/s.")
        illuminance: list[float] | None = Field(None, description="Illuminance in lux.")
        noise_laeq: list[float] | None = Field(
            None, description="A-weighted equivalent sound levels in dB."
        )
        co2: list[float] | None = Field(None, description="CO₂ concentration in ppm.")
        metabolic_rate: float = Field(1.2, description="Metabolic rate in met.")
        clothing_insulation: float = Field(0.5, description="Clothing insulation in clo.")
        visual_task_type: str = Field("general", description="EN 12464-1 task type.")
        acoustic_nc_level: str = Field("NC-35", description="Noise Criteria level.")
        iaq_threshold_level: str = Field("good", description="CO₂ threshold level.")

    class _PollutantIAQInputModel(BaseModel):
        pm25: list[float] | None = Field(None, description="PM2.5 concentrations in µg/m³.")
        pm10: list[float] | None = Field(None, description="PM10 concentrations in µg/m³.")
        tvoc: list[float] | None = Field(None, description="TVOC concentrations in µg/m³.")
        formaldehyde: list[float] | None = Field(None, description="Formaldehyde in ppb.")
        co: list[float] | None = Field(None, description="CO concentrations in ppm.")
        threshold_level: str = Field(
            "good", description="Threshold level: excellent/good/moderate/poor."
        )

    class _SPMVInputModel(BaseModel):
        indoor_temp: list[float] = Field(..., description="Indoor air temperatures in °C.")
        indoor_rh: list[float] = Field(..., description="Indoor relative humidity (0-100%).")
        season: str | None = Field(None, description="Season override: winter/mid/summer.")

    class _AdaptiveInputModel(BaseModel):
        tdb: list[float] = Field(..., description="Indoor dry-bulb temperatures in °C.")
        tr: list[float] = Field(..., description="Mean radiant temperatures in °C.")
        t_outdoor: float = Field(..., description="Prevailing/running mean outdoor temp in °C.")
        standard: str = Field("ashrae", description="Standard: ashrae or en.")
        acceptability: int = Field(80, description="ASHRAE acceptability: 80 or 90.")
        category: str = Field("ii", description="EN category: i, ii, or iii.")

    class _TSVInputModel(BaseModel):
        tsv: list[float] = Field(..., description="TSV values (-3 to +3).")
        compliance_threshold: float = Field(1.5, description="ASHRAE 55 compliance threshold.")

    _ThermalComfortInputModel = _ThermalComfortInputModel
    _IEQEvaluationInputModel = _IEQEvaluationInputModel


_ThermalComfortInputModel: Any = None
_IEQEvaluationInputModel: Any = None
_PollutantIAQInputModel: Any = None
_SPMVInputModel: Any = None
_AdaptiveInputModel: Any = None
_TSVInputModel: Any = None


def get_pydantic_models() -> tuple[Any, Any]:
    """Return the Pydantic model classes, building them on first call.

    Returns
    -------
    tuple
        (ThermalComfortInputModel, IEQEvaluationInputModel)
    """
    if _ThermalComfortInputModel is None:
        _build_pydantic_models()
    return _ThermalComfortInputModel, _IEQEvaluationInputModel


def get_all_pydantic_models() -> dict[str, Any]:
    """Return all Pydantic model classes, building them on first call.

    Returns
    -------
    dict[str, Any]
        Mapping of model name to Pydantic model class.
    """
    if _ThermalComfortInputModel is None:
        _build_pydantic_models()
    return {
        "thermal": _ThermalComfortInputModel,
        "ieq": _IEQEvaluationInputModel,
        "pollutant_iaq": _PollutantIAQInputModel,
        "spmv": _SPMVInputModel,
        "adaptive": _AdaptiveInputModel,
        "tsv": _TSVInputModel,
    }


def evaluate_thermal_tool(
    air_temperature: list[float],
    relative_humidity: list[float],
    mean_radiant_temperature: list[float] | None = None,
    air_velocity: float = 0.1,
    metabolic_rate: float = 1.2,
    clothing_insulation: float = 0.5,
) -> dict[str, float | str]:
    """Evaluate thermal comfort from sensor arrays.

    Wraps ``comfio.evaluate_thermal`` for LLM function calling.
    Returns a summary dict with mean PMV, mean PPD, and discomfort risk level.

    Parameters
    ----------
    air_temperature : list[float]
        Dry-bulb air temperatures in °C.
    relative_humidity : list[float]
        Relative humidity (0-100%).
    mean_radiant_temperature : list[float] or None
        Optional radiant temperatures. Defaults to air temperature.
    air_velocity : float
        Air velocity in m/s.
    metabolic_rate : float
        Metabolic rate in met.
    clothing_insulation : float
        Clothing insulation in clo.

    Returns
    -------
    dict
        Summary with keys: mean_pmv, mean_ppd, max_ppd, discomfort_risk.

    Examples
    --------
    >>> result = evaluate_thermal_tool(
    ...     air_temperature=[24.0, 25.0, 26.0],
    ...     relative_humidity=[50.0, 50.0, 50.0],
    ... )
    >>> isinstance(result['mean_pmv'], float)
    True
    >>> result['discomfort_risk']
    'acceptable'
    """
    tdb = np.array(air_temperature, dtype=float)
    rh = np.array(relative_humidity, dtype=float)
    tr = np.array(mean_radiant_temperature, dtype=float) if mean_radiant_temperature else tdb
    vr = np.full_like(tdb, air_velocity)

    result = evaluate_thermal(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=metabolic_rate,
        clo=clothing_insulation,
    )

    mean_pmv = float(np.mean(result.pmv))
    mean_ppd = float(np.mean(result.ppd))
    max_ppd = float(np.max(result.ppd))

    return {
        "mean_pmv": round(mean_pmv, 2),
        "mean_ppd": round(mean_ppd, 1),
        "max_ppd": round(max_ppd, 1),
        "discomfort_risk": "high" if mean_ppd > 10.0 else "acceptable",
    }


def evaluate_ieq_tool(
    air_temperature: list[float],
    relative_humidity: list[float],
    mean_radiant_temperature: list[float] | None = None,
    air_velocity: float = 0.1,
    illuminance: list[float] | None = None,
    noise_laeq: list[float] | None = None,
    co2: list[float] | None = None,
    metabolic_rate: float = 1.2,
    clothing_insulation: float = 0.5,
    visual_task_type: str = "general",
    acoustic_nc_level: str = "NC-35",
    iaq_threshold_level: str = "good",
) -> dict[str, Any]:
    """Evaluate multi-domain IEQ and return a summary + markdown for LLM context.

    Wraps comfio's full evaluation pipeline for LLM function calling.
    Returns a structured dict with IEQ index statistics, domain scores,
    worst domain, and a markdown summary string.

    Parameters
    ----------
    air_temperature : list[float]
        Dry-bulb air temperatures in °C.
    relative_humidity : list[float]
        Relative humidity (0-100%).
    mean_radiant_temperature : list[float] or None
        Optional radiant temperatures. Defaults to air temperature.
    air_velocity : float
        Air velocity in m/s.
    illuminance : list[float] or None
        Illuminance in lux (for visual domain).
    noise_laeq : list[float] or None
        A-weighted sound levels in dB (for acoustic domain).
    co2 : list[float] or None
        CO₂ concentration in ppm (for IAQ domain).
    metabolic_rate : float
        Metabolic rate in met.
    clothing_insulation : float
        Clothing insulation in clo.
    visual_task_type : str
        EN 12464-1 task type for visual evaluation.
    acoustic_nc_level : str
        Noise Criteria level for acoustic evaluation.
    iaq_threshold_level : str
        CO₂ threshold level for IAQ evaluation.

    Returns
    -------
    dict
        Summary with IEQ stats, domain scores, worst domain, and markdown.

    Examples
    --------
    >>> result = evaluate_ieq_tool(
    ...     air_temperature=[24.0, 25.0],
    ...     relative_humidity=[50.0, 50.0],
    ...     illuminance=[500.0, 480.0],
    ...     co2=[800.0, 900.0],
    ... )
    >>> 'ieq_index_avg' in result
    True
    >>> 'markdown' in result
    True
    """
    tdb = np.array(air_temperature, dtype=float)
    rh = np.array(relative_humidity, dtype=float)
    tr = np.array(mean_radiant_temperature, dtype=float) if mean_radiant_temperature else tdb
    vr = np.full_like(tdb, air_velocity)

    thermal_res = evaluate_thermal(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=metabolic_rate,
        clo=clothing_insulation,
    )

    visual_res = None
    if illuminance is not None:
        visual_res = evaluate_visual(
            illuminance=np.array(illuminance, dtype=float),
            task_type=visual_task_type,
        )

    acoustic_res = None
    if noise_laeq is not None:
        acoustic_res = evaluate_acoustic(
            laeq=np.array(noise_laeq, dtype=float),
            nc_level=acoustic_nc_level,
        )

    iaq_res = None
    if co2 is not None:
        iaq_res = evaluate_iaq(
            co2=np.array(co2, dtype=float),
            threshold_level=iaq_threshold_level,
        )

    ieq_result = calculate_global_ieq(
        thermal=thermal_res,
        visual=visual_res,
        acoustic=acoustic_res,
        iaq=iaq_res,
    )

    summary = ieq_to_summary_dict(ieq_result)
    summary["markdown"] = ieq_to_markdown(ieq_result)
    return summary


def evaluate_pollutant_iaq_tool(
    pm25: list[float] | None = None,
    pm10: list[float] | None = None,
    tvoc: list[float] | None = None,
    formaldehyde: list[float] | None = None,
    co: list[float] | None = None,
    threshold_level: str = "good",
) -> dict[str, Any]:
    """Evaluate IAQ pollutant concentrations against health thresholds.

    Returns summary statistics for PM2.5, PM10, TVOC, formaldehyde, and CO
    based on WHO, EPA, and WELL Building Standard thresholds.

    Parameters
    ----------
    pm25 : list[float] or None
        PM2.5 concentrations in µg/m³.
    pm10 : list[float] or None
        PM10 concentrations in µg/m³.
    tvoc : list[float] or None
        TVOC concentrations in µg/m³.
    formaldehyde : list[float] or None
        Formaldehyde concentrations in ppb.
    co : list[float] or None
        CO concentrations in ppm.
    threshold_level : str
        Threshold level: excellent/good/moderate/poor.

    Returns
    -------
    dict
        Summary with mean pollutant scores, compliance rates, and overall score.
    """
    kwargs: dict[str, Any] = {"threshold_level": threshold_level}
    if pm25 is not None:
        kwargs["pm25"] = np.array(pm25, dtype=float)
    if pm10 is not None:
        kwargs["pm10"] = np.array(pm10, dtype=float)
    if tvoc is not None:
        kwargs["tvoc"] = np.array(tvoc, dtype=float)
    if formaldehyde is not None:
        kwargs["formaldehyde"] = np.array(formaldehyde, dtype=float)
    if co is not None:
        kwargs["co"] = np.array(co, dtype=float)

    result = evaluate_iaq_pollutants(**kwargs)

    return {
        "mean_pollutant_score": float(np.mean(result.score)),
        "min_pollutant_score": float(np.min(result.score)),
        "threshold_level": result.threshold_level,
        "pollutants_evaluated": [
            k
            for k in ["pm25", "pm10", "tvoc", "formaldehyde", "co"]
            if getattr(result, k) is not None
        ],
    }


def evaluate_spmv_tool(
    indoor_temp: list[float],
    indoor_rh: list[float],
    season: str | None = None,
) -> dict[str, float | str]:
    """Evaluate simplified PMV (sPMV) from indoor temperature and humidity.

    Uses the Buratti et al. (2009) seasonal model. Only requires indoor
    air temperature and relative humidity — no metabolic rate or clothing needed.

    Parameters
    ----------
    indoor_temp : list[float]
        Indoor air temperatures in °C.
    indoor_rh : list[float]
        Indoor relative humidity (0-100%).
    season : str or None
        Season override: winter/mid/summer. Auto-detected if None.

    Returns
    -------
    dict
        Summary with mean sPMV, comfort score, and season.
    """
    result = evaluate_spmv(
        indoor_temp=np.array(indoor_temp, dtype=float),
        indoor_rh=np.array(indoor_rh, dtype=float),
        season=season,
    )
    return {
        "mean_spmv": float(np.mean(result.spmv)),
        "mean_score": float(np.mean(result.score)),
        "season": result.season,
        "n_samples": len(result.spmv),
    }


def evaluate_adaptive_tool(
    tdb: list[float],
    tr: list[float],
    t_outdoor: float,
    standard: str = "ashrae",
    acceptability: int = 80,
    category: str = "ii",
) -> dict[str, Any]:
    """Evaluate adaptive thermal comfort per ASHRAE 55 or EN 16798-1.

    Parameters
    ----------
    tdb : list[float]
        Indoor dry-bulb temperatures in °C.
    tr : list[float]
        Mean radiant temperatures in °C.
    t_outdoor : float
        Prevailing mean (ASHRAE) or running mean (EN) outdoor temp in °C.
    standard : str
        "ashrae" or "en".
    acceptability : int
        ASHRAE acceptability level (80 or 90).
    category : str
        EN category (i, ii, or iii).

    Returns
    -------
    dict
        Summary with comfort temperature, band, compliance rate, and score.
    """
    if standard == "en":
        result = evaluate_adaptive_en(
            tdb=np.array(tdb, dtype=float),
            tr=np.array(tr, dtype=float),
            t_running_mean=t_outdoor,
            category=category,
        )
    else:
        result = evaluate_adaptive_ashrae(
            tdb=np.array(tdb, dtype=float),
            tr=np.array(tr, dtype=float),
            t_prevail=t_outdoor,
            acceptability=acceptability,
        )
    return {
        "t_comf": result.t_comf,
        "t_comf_lower": result.t_comf_lower,
        "t_comf_upper": result.t_comf_upper,
        "compliance_rate": float(np.mean(result.compliant)),
        "mean_score": float(np.mean(result.score)),
        "standard": result.standard,
    }


def evaluate_tsv_tool(
    tsv: list[float],
    compliance_threshold: float = 1.5,
) -> dict[str, float]:
    """Evaluate Thermal Sensation Vote data for comfort and compliance.

    Parameters
    ----------
    tsv : list[float]
        TSV values (-3 to +3).
    compliance_threshold : float
        ASHRAE 55-2023 Appendix L threshold (default 1.5).

    Returns
    -------
    dict
        Summary with mean TSV, compliance rate, PPD approximation, and score.
    """
    result = evaluate_tsv(
        tsv=np.array(tsv, dtype=float),
        compliance_threshold=compliance_threshold,
    )
    return {
        "mean_tsv": result.mean_tsv,
        "compliance_rate": result.compliance_rate,
        "mean_ppd_approx": float(np.mean(result.ppd_approx)),
        "mean_score": float(np.mean(result.score)),
        "n_samples": result.n_samples,
    }


def to_openai_tools() -> list[dict[str, Any]]:
    """Generate OpenAI function-calling JSON schemas for comfio tools.

    Requires pydantic (``pip install comfio[agent]``).

    Returns
    -------
    list[dict]
        List of OpenAI tool definition dicts.

    Examples
    --------
    >>> tools = to_openai_tools()  # doctest: +SKIP
    >>> len(tools)  # doctest: +SKIP
    6
    """
    thermal_model, ieq_model = get_pydantic_models()
    all_models = get_all_pydantic_models()

    def _schema_from_pydantic(model: Any, name: str, description: str) -> dict[str, Any]:
        schema = model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }

    return [
        _schema_from_pydantic(
            thermal_model,
            "evaluate_thermal",
            "Calculate thermal comfort metrics (PMV, PPD) from air temperature, humidity, "
            "radiant temperature, and air velocity arrays.",
        ),
        _schema_from_pydantic(
            ieq_model,
            "evaluate_ieq",
            "Evaluate multi-domain Indoor Environmental Quality (thermal, visual, acoustic, IAQ) "
            "and return a Global IEQ Index with domain breakdown and markdown summary.",
        ),
        _schema_from_pydantic(
            all_models["pollutant_iaq"],
            "evaluate_pollutant_iaq",
            "Evaluate IAQ pollutant concentrations (PM2.5, PM10, TVOC, formaldehyde, CO) "
            "against WHO, EPA, and WELL Building Standard thresholds.",
        ),
        _schema_from_pydantic(
            all_models["spmv"],
            "evaluate_spmv",
            "Calculate simplified PMV (sPMV) from indoor temperature and humidity only, "
            "using the Buratti seasonal model.",
        ),
        _schema_from_pydantic(
            all_models["adaptive"],
            "evaluate_adaptive",
            "Evaluate adaptive thermal comfort per ASHRAE 55-2023 or EN 16798-1:2019 "
            "for naturally ventilated buildings.",
        ),
        _schema_from_pydantic(
            all_models["tsv"],
            "evaluate_tsv",
            "Evaluate Thermal Sensation Vote data for comfort and compliance per "
            "ASHRAE 55-2023 Appendix L.",
        ),
    ]


def to_langchain_tools() -> list[Any]:
    """Generate LangChain StructuredTool objects for comfio.

    Requires pydantic and langchain (``pip install comfio[agent] langchain``).

    Returns
    -------
    list
        List of ``langchain.tools.StructuredTool`` objects.
    """
    thermal_model, ieq_model = get_pydantic_models()
    all_models = get_all_pydantic_models()

    try:
        from langchain.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "LangChain is required for to_langchain_tools(). Install with: pip install langchain"
        ) from exc

    thermal_tool = StructuredTool.from_function(
        func=evaluate_thermal_tool,
        name="evaluate_thermal",
        description=(
            "Calculate thermal comfort metrics (PMV, PPD) from air temperature, "
            "humidity, radiant temperature, and air velocity arrays."
        ),
        args_schema=thermal_model,
    )

    ieq_tool = StructuredTool.from_function(
        func=evaluate_ieq_tool,
        name="evaluate_ieq",
        description=(
            "Evaluate multi-domain Indoor Environmental Quality (thermal, visual, "
            "acoustic, IAQ) and return a Global IEQ Index with domain breakdown "
            "and markdown summary."
        ),
        args_schema=ieq_model,
    )

    pollutant_tool = StructuredTool.from_function(
        func=evaluate_pollutant_iaq_tool,
        name="evaluate_pollutant_iaq",
        description=(
            "Evaluate IAQ pollutant concentrations (PM2.5, PM10, TVOC, formaldehyde, CO) "
            "against WHO, EPA, and WELL thresholds."
        ),
        args_schema=all_models["pollutant_iaq"],
    )

    spmv_tool = StructuredTool.from_function(
        func=evaluate_spmv_tool,
        name="evaluate_spmv",
        description=("Calculate simplified PMV from indoor temperature and humidity only."),
        args_schema=all_models["spmv"],
    )

    adaptive_tool = StructuredTool.from_function(
        func=evaluate_adaptive_tool,
        name="evaluate_adaptive",
        description=("Evaluate adaptive thermal comfort per ASHRAE 55 or EN 16798-1."),
        args_schema=all_models["adaptive"],
    )

    tsv_tool = StructuredTool.from_function(
        func=evaluate_tsv_tool,
        name="evaluate_tsv",
        description=("Evaluate Thermal Sensation Vote data for comfort and compliance."),
        args_schema=all_models["tsv"],
    )

    return [thermal_tool, ieq_tool, pollutant_tool, spmv_tool, adaptive_tool, tsv_tool]
