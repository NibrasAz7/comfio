#!/usr/bin/env python3
"""Verify comfio installation and report available extras."""

import importlib
import sys

EXTRAS = {
    "ml": "sklearn",
    "torch": "torch",
    "keras": "tensorflow",
    "agent": "pydantic",
    "daylighting": "pyradiance",
    "color": "colour",
    "acoustics": "acoustics",
    "psychrometrics": "psychrolib",
}


def main() -> int:
    try:
        import comfio
        print(f"comfio version: {comfio.__version__}")
    except ImportError:
        print("ERROR: comfio is not installed.")
        print("Install with: pip install comfio")
        return 1

    print(f"Python: {sys.version.split()[0]}")

    deps = {"numpy": None, "pandas": None, "pythermalcomfort": None}
    for pkg in deps:
        try:
            mod = importlib.import_module(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = "MISSING"

    print("\nCore dependencies:")
    for pkg, ver in deps.items():
        status = "OK" if ver != "MISSING" else "MISSING"
        print(f"  {pkg}: {ver} [{status}]")

    print("\nOptional extras:")
    for extra, module_name in EXTRAS.items():
        try:
            importlib.import_module(module_name)
            print(f"  [{extra}] {module_name}: installed")
        except ImportError:
            print(f"  [{extra}] {module_name}: not installed")

    print("\nPublic API exports:")
    exports = getattr(comfio, "__all__", [])
    print(f"  {len(exports)} symbols exported")
    for name in sorted(exports):
        print(f"    - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
