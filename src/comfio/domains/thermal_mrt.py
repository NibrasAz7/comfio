"""Mean radiant temperature (MRT) of a rectangular room enclosure.

Computes view factors from an occupant point to the six enclosure
surfaces (plus optional window patches on the four vertical walls) with
the Hottel point-to-rectangle formula, then the black-body MRT

.. math::

    T_r = \\Big( \\sum_i F_i \\, (T_i + 273.15)^4 \\Big)^{1/4} - 273.15

The view-factor helpers (``_aux``, ``_hottel_rect``,
``_hottel_quadrant``) and the enclosure geometry are ported from
``mrt_replication/src/view_factor.py``, released under the MIT License:

    MIT License

    Copyright (c) 2026 Nibras Abo Alzahab, Edoardo Sulpizii,
    Vittoria Cipollone, Serena Serroni, Gian Marco Revel

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Geometry convention: the room is a box ``length`` (x) × ``width`` (y) ×
``height`` (z); front is the ``x=0`` wall, rear ``x=L``, left ``y=0``,
right ``y=W``, floor ``z=0``, ceiling ``z=H``. Points strictly outside
``[0,L]×[0,W]×[0,H]`` raise ``ValueError``; a point lying exactly on a
surface is treated as ``_MIN_PERP_DIST`` metres inside the room so the
formulas never divide by zero (the surface it sits on then subtends
nearly a hemisphere, which matches the physical limit).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

SURFACES = ("front", "rear", "left", "right", "floor", "ceiling")
WALLS = ("front", "rear", "left", "right")

KELVIN = 273.15

# Perpendicular distance used when the point lies exactly on a surface:
# treated as 1 mm inside the room (see module docstring).
_MIN_PERP_DIST = 1e-3  # m

# Centre-of-gravity height per posture (ISO 7726 convention).
_POSTURE_Z = {"seated": 0.6, "standing": 1.1}

# pythermalcomfort's f_svv divides by the occupant-window distance.
_MIN_SOLAR_DIST = 1e-3  # m


@dataclass(frozen=True)
class Window:
    """Rectangular window patch on a vertical wall.

    ``u0`` is the offset along the wall measured from its near corner —
    the ``y=0`` edge for front/rear walls, the ``x=0`` edge for
    left/right walls. ``z0`` is the sill height above the floor.
    """

    wall: Literal["front", "rear", "left", "right"]
    u0: float
    z0: float
    width: float
    height: float
    temp_c: float

    def __post_init__(self) -> None:
        if self.wall not in WALLS:
            raise ValueError(f"wall: expected one of {WALLS}, got {self.wall!r}")
        if self.width <= 0:
            raise ValueError(f"width: window width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height: window height must be > 0, got {self.height}")


@dataclass(frozen=True)
class Room:
    """Rectangular room enclosure with uniform surface temperatures."""

    length: float  # x extent, front(x=0) -> rear(x=L)  [m]
    width: float  # y extent, left(y=0) -> right(y=W)   [m]
    height: float  # z extent, floor(z=0) -> ceiling(z=H) [m]
    surface_temps: Mapping[str, float]  # all six SURFACES keys, °C
    windows: tuple[Window, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
        ):
            if value <= 0:
                raise ValueError(f"{name}: room dimension must be > 0, got {value}")
        missing = [s for s in SURFACES if s not in self.surface_temps]
        if missing:
            raise ValueError(f"surface_temps: missing temperatures for {missing}")

        # wall horizontal extent: front/rear span the width, left/right
        # span the length
        extent = {
            "front": self.width,
            "rear": self.width,
            "left": self.length,
            "right": self.length,
        }
        for i, win in enumerate(self.windows):
            ext = extent[win.wall]
            if win.u0 < 0 or win.u0 + win.width > ext:
                raise ValueError(
                    f"u0/width: windows[{i}] extends outside wall "
                    f"{win.wall!r} (u0={win.u0}, width={win.width}, "
                    f"wall extent={ext})"
                )
            if win.z0 < 0 or win.z0 + win.height > self.height:
                raise ValueError(
                    f"z0/height: windows[{i}] extends outside wall "
                    f"{win.wall!r} (z0={win.z0}, height={win.height}, "
                    f"room height={self.height})"
                )
        for (i, wa), (j, wb) in combinations(enumerate(self.windows), 2):
            if wa.wall != wb.wall:
                continue
            u_overlap = min(wa.u0 + wa.width, wb.u0 + wb.width) > max(wa.u0, wb.u0)
            z_overlap = min(wa.z0 + wa.height, wb.z0 + wb.height) > max(
                wa.z0, wb.z0
            )
            if u_overlap and z_overlap:
                raise ValueError(
                    f"windows: windows[{i}] and windows[{j}] overlap on "
                    f"wall {wa.wall!r}"
                )


# ---------------------------------------------------------------------------
# Hottel analytical view factors (ported from mrt_replication, MIT — see
# module docstring). A point-to-rectangle formula; a general rectangle is
# obtained by 4-quadrant superposition around the point's projection.
# ---------------------------------------------------------------------------


def _aux(x: float, y: float) -> float:
    """Auxiliary function for the Hottel point-to-rectangle formula."""
    if abs(x) < 1e-10 and abs(y) < 1e-10:
        return 0.0
    d1 = math.sqrt(1.0 + x * x)
    d2 = math.sqrt(1.0 + y * y)
    return x / d1 * math.atan(y / d1) + y / d2 * math.atan(x / d2)


def _hottel_rect(a: float, b: float, c: float) -> float:
    """Hottel view factor from a point to an ``a`` × ``b`` corner rectangle
    in a parallel plane at distance ``c``. F = aux(a/c, b/c) / (2π)."""
    if c <= 1e-6 or a <= 0 or b <= 0:
        return 0.0
    return _aux(a / c, b / c) / (2.0 * math.pi)


def _hottel_quadrant(
    a1: float, a2: float, b1: float, b2: float, c: float
) -> float:
    """View factor to the rectangle ``a∈[a1,a2]``, ``b∈[b1,b2]`` in a
    parallel plane at distance ``c``; bounds may straddle the point's
    projection (negative ``a1``/``b1``)."""
    if c <= 1e-6:
        return 0.0

    def _f(a: float, b: float) -> float:
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            return 0.0
        sign = math.copysign(1.0, a) * math.copysign(1.0, b) if a != 0 and b != 0 else 0.0
        return sign * _aux(abs(a) / c, abs(b) / c) / (2.0 * math.pi)

    return _f(a2, b2) - _f(a1, b2) - _f(a2, b1) + _f(a1, b1)


def _surface_geometry(
    room: Room, x: float, y: float, z: float
) -> dict[str, tuple[float, float, float, float, float]]:
    """Per surface: (perpendicular distance, u of point, v of point,
    u extent, v extent) in the surface's own plane coordinates."""
    length, width, height = room.length, room.width, room.height
    return {
        "front": (x, y, z, width, height),
        "rear": (length - x, y, z, width, height),
        "left": (y, x, z, length, height),
        "right": (width - y, x, z, length, height),
        "floor": (z, x, y, length, width),
        "ceiling": (height - z, x, y, length, width),
    }


def _check_point(room: Room, x: float, y: float, z: float) -> None:
    for name, value, hi in (
        ("x", x, room.length),
        ("y", y, room.width),
        ("z", z, room.height),
    ):
        if not 0.0 <= value <= hi:
            raise ValueError(
                f"{name}: point {value} outside room extent [0, {hi}]"
            )


def view_factors(room: Room, x: float, y: float, z: float) -> dict[str, float]:
    """Normalised view factors from the point to every enclosure surface.

    Keys are the six entries of ``SURFACES`` plus ``"window:<i>"`` for
    each window; a wall's factor is net of its windows, so the whole
    dict sums to 1 (±1e-9). A point exactly on a surface is clamped to
    ``_MIN_PERP_DIST`` inside the room rather than treated as an error.
    """
    _check_point(room, x, y, z)
    geom = _surface_geometry(room, x, y, z)

    factors: dict[str, float] = {}
    for name, (c, up, vp, u_ext, v_ext) in geom.items():
        c = max(c, _MIN_PERP_DIST)
        factors[name] = _hottel_quadrant(-up, u_ext - up, -vp, v_ext - vp, c)

    for i, win in enumerate(room.windows):
        c, up, vp, _, _ = geom[win.wall]
        c = max(c, _MIN_PERP_DIST)
        f_win = _hottel_quadrant(
            win.u0 - up,
            win.u0 + win.width - up,
            win.z0 - vp,
            win.z0 + win.height - vp,
            c,
        )
        factors[win.wall] -= f_win
        factors[f"window:{i}"] = f_win

    total = sum(factors.values())
    return {k: v / total for k, v in factors.items()}


def mrt_point(room: Room, x: float, y: float, z: float) -> float:
    """Black-body MRT at a point: (Σ Fᵢ·(Tᵢ+273.15)⁴)^¼ − 273.15 [°C]."""
    factors = view_factors(room, x, y, z)
    acc = 0.0
    for surf in SURFACES:
        acc += factors[surf] * (room.surface_temps[surf] + KELVIN) ** 4
    for i, win in enumerate(room.windows):
        acc += factors[f"window:{i}"] * (win.temp_c + KELVIN) ** 4
    return float(acc**0.25 - KELVIN)


def mrt_grid(
    room: Room,
    posture: Literal["seated", "standing"] = "seated",
    spacing: float = 0.25,
    margin: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan-view MRT grid at the posture's centre-of-gravity height.

    Returns ``(xs, ys, grid)`` where ``grid[iy, ix]`` is the MRT at
    ``(xs[ix], ys[ix], z_posture)``; ``z`` is 0.6 m seated, 1.1 m
    standing. Points span ``margin`` … ``dim − margin`` in both axes.
    """
    if posture not in _POSTURE_Z:
        raise ValueError(f"posture: expected one of {tuple(_POSTURE_Z)}, got {posture!r}")
    if spacing <= 0:
        raise ValueError(f"spacing: grid spacing must be > 0, got {spacing}")
    z = _POSTURE_Z[posture]
    if z > room.height:
        raise ValueError(
            f"height: posture {posture!r} needs z={z} m above the floor, "
            f"room height is {room.height}"
        )

    nx = math.floor((room.length - 2.0 * margin) / spacing) + 1
    ny = math.floor((room.width - 2.0 * margin) / spacing) + 1
    if nx < 1 or ny < 1:
        raise ValueError(
            f"margin: margin={margin} leaves no grid points inside "
            f"{room.length}×{room.width} m"
        )
    xs = margin + np.arange(nx) * spacing
    ys = margin + np.arange(ny) * spacing

    grid = np.empty((ny, nx), dtype=float)
    for iy, yp in enumerate(ys):
        for ix, xp in enumerate(xs):
            grid[iy, ix] = mrt_point(room, float(xp), float(yp), z)
    return xs, ys, grid


def solar_delta_mrt(
    window: Window,
    distance: float,
    *,
    sol_altitude: float,
    sharp: float,
    sol_radiation_dir: float,
    sol_transmittance: float,
    f_bes: float,
    asw: float = 0.7,
    posture: str = "sitting",
    floor_reflectance: float = 0.6,
) -> float:
    """Short-wave solar ΔMRT for an occupant facing ``window``.

    Wraps ``pythermalcomfort`` (ASHRAE 55-2023 §C2): the sky-vault view
    fraction is ``f_svv(window.width, window.height, distance)`` and the
    ERF model is ``solar_gain``; returns its ``delta_mrt`` field [°C].
    ``distance`` is the occupant-to-window distance — clamped to 1 mm so
    an occupant flush with the glazing does not divide by zero.
    """
    from pythermalcomfort.models.solar_gain import solar_gain
    from pythermalcomfort.utilities import f_svv

    fsvv = f_svv(window.width, window.height, max(float(distance), _MIN_SOLAR_DIST))
    result = solar_gain(
        sol_altitude=sol_altitude,
        sharp=sharp,
        sol_radiation_dir=sol_radiation_dir,
        sol_transmittance=sol_transmittance,
        f_svv=fsvv,
        f_bes=f_bes,
        asw=asw,
        posture=posture,
        floor_reflectance=floor_reflectance,
    )
    return float(result.delta_mrt)
