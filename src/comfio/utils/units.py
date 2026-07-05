"""Unit conversion utilities.

All conversions operate on scalars or numpy arrays interchangeably.
Functions are kept minimal — only the conversions needed by the domain
modules are provided.
"""

from __future__ import annotations

from typing import Union

import numpy as np

Number = Union[float, int, np.ndarray]


def fahrenheit_to_celsius(f: Number) -> Number:
    """Convert Fahrenheit to Celsius.

    Parameters
    ----------
    f : float or np.ndarray
        Temperature in degrees Fahrenheit.

    Returns
    -------
    float or np.ndarray
        Temperature in degrees Celsius.
    """
    return (f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c: Number) -> Number:
    """Convert Celsius to Fahrenheit.

    Parameters
    ----------
    c : float or np.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    float or np.ndarray
        Temperature in degrees Fahrenheit.
    """
    return c * 9.0 / 5.0 + 32.0


def ppm_to_mgm3(ppm: Number, molar_mass: float, temperature_c: Number, pressure_kpa: Number = 101.325) -> Number:
    """Convert gas concentration from ppm to mg/m³.

    Uses the ideal gas law: mg/m³ = ppm * M * P / (R * T)

    Parameters
    ----------
    ppm : float or np.ndarray
        Concentration in parts per million.
    molar_mass : float
        Molar mass of the gas (g/mol), e.g. 44.01 for CO₂.
    temperature_c : float or np.ndarray
        Air temperature in degrees Celsius.
    pressure_kpa : float, default 101.325
        Atmospheric pressure in kPa.

    Returns
    -------
    float or np.ndarray
        Concentration in mg/m³.
    """
    r = 8.314  # J/(mol·K)
    temp_k = temperature_c + 273.15
    return ppm * molar_mass * pressure_kpa / (r * temp_k)


def mgm3_to_ppm(mgm3: Number, molar_mass: float, temperature_c: Number, pressure_kpa: Number = 101.325) -> Number:
    """Convert gas concentration from mg/m³ to ppm.

    Parameters
    ----------
    mgm3 : float or np.ndarray
        Concentration in mg/m³.
    molar_mass : float
        Molar mass of the gas (g/mol).
    temperature_c : float or np.ndarray
        Air temperature in degrees Celsius.
    pressure_kpa : float, default 101.325
        Atmospheric pressure in kPa.

    Returns
    -------
    float or np.ndarray
        Concentration in parts per million.
    """
    r = 8.314
    temp_k = temperature_c + 273.15
    return mgm3 * r * temp_k / (molar_mass * pressure_kpa)


def db_to_pa(db: Number, ref_pa: float = 2e-5) -> Number:
    """Convert sound pressure level (dB SPL) to Pascals.

    Parameters
    ----------
    db : float or np.ndarray
        Sound pressure level in dB.
    ref_pa : float, default 2e-5
        Reference pressure in Pa (20 µPa for air).

    Returns
    -------
    float or np.ndarray
        Sound pressure in Pascals.
    """
    return ref_pa * 10.0 ** (db / 20.0)


def pa_to_db(pa: Number, ref_pa: float = 2e-5) -> Number:
    """Convert Pascals to sound pressure level (dB SPL).

    Parameters
    ----------
    pa : float or np.ndarray
        Sound pressure in Pascals.
    ref_pa : float, default 2e-5
        Reference pressure in Pa.

    Returns
    -------
    float or np.ndarray
        Sound pressure level in dB.
    """
    return 20.0 * np.log10(pa / ref_pa)


def clo_to_m2kw(clo: Number) -> Number:
    """Convert clothing insulation from clo to m²·K/W.

    1 clo = 0.155 m²·K/W

    Parameters
    ----------
    clo : float or np.ndarray
        Clothing insulation in clo units.

    Returns
    -------
    float or np.ndarray
        Clothing insulation in m²·K/W.
    """
    return clo * 0.155


def met_to_wm2(met: Number) -> Number:
    """Convert metabolic rate from met to W/m².

    1 met = 58.2 W/m²

    Parameters
    ----------
    met : float or np.ndarray
        Metabolic rate in met units.

    Returns
    -------
    float or np.ndarray
        Metabolic rate in W/m².
    """
    return met * 58.2
