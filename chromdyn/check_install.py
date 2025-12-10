# chromdyn/check_install.py

from __future__ import annotations

from typing import Optional
from importlib import metadata

try:
    import openmm
    import openmm.unit as unit
except ImportError:
    openmm = None  # type: ignore[assignment]
    unit = None    # type: ignore[assignment]

try:
    import chromdyn
except ImportError:
    chromdyn = None  # type: ignore[assignment]


def _check_chromdyn_import() -> None:
    """
    Check that chromdyn can be imported and return its version string.
    Raises if import fails.
    """
    if chromdyn is None:
        raise ImportError("chromdyn is not importable.")
    
    ver: Optional[str] = metadata.version("chromdyn")
    if ver is None:
        ver = "UNKNOWN"
    print(f"chromdyn version: {ver}\n")
    


def _check_openmm() -> None:
    """
    Check that OpenMM is available and print available platforms.
    Raises if OpenMM is not importable.
    """
    if openmm is None:
        raise ImportError("OpenMM is not installed (import openmm failed).")

    print(f"OpenMM version: {openmm.__version__}")
    print("Available platforms:")

    for i in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(i)
        print(f"  - {platform.getName()}")

    print()  # blank line after platform list


def _run_minimal_simulation() -> None:
    """
    Run a tiny chromdyn simulation on CPU to confirm basic functionality.
    Raises if it fails.
    """
    if openmm is None or chromdyn is None:
        raise RuntimeError("OpenMM or chromdyn not available for minimal simulation.")

    from chromdyn import ChromatinDynamics, TopologyGenerator

    # Very small toy system: 5 particles
    nparticles = 5

    generator = TopologyGenerator()
    generator.gen_top([nparticles])

    sim = ChromatinDynamics(
        generator.topology,
        name="test",
        platform_name="CPU",
        console_stream=False,
        write_logs=False,
    )
    sim.force_field_manager.add_harmonic_bonds()

    # Set up the simulation
    sim.simulation_setup(
        init_struct="randomwalk",
        integrator="langevin",
        temperature=120.0,
        timestep=0.01,
        save_pos=False,
        save_energy=False,
        stability_check=False,
    )

    # Run a short simulation and sanity-check positions
    sim.run(10, report=False)
    state = sim.simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # type: ignore[union-attr]
    if pos.shape != (nparticles, 3):
        raise RuntimeError(
            f"Unexpected position array shape {pos.shape}, expected ({nparticles}, 3)."
        )


def main() -> None:
    """
    Entry point for `python -m chromdyn.testInstallation`.
    Runs concise checks on chromdyn, OpenMM, and a minimal simulation.
    """

    print("\n=== chromdyn installation test ===\n")

    results: list[str] = []

    # Check chromdyn import
    try:
        _check_chromdyn_import()
        results.append(f"✔ chromdyn import: OK")
    except Exception as exc:  # noqa: BLE001
        results.append(f"✘ chromdyn import: FAILED ({type(exc).__name__}: {exc})")
        print("\n".join(results))
        raise SystemExit(1)

    # Check OpenMM
    try:
        _check_openmm()
        results.append("✔ OpenMM import & platforms: OK")
    except Exception as exc:  # noqa: BLE001
        results.append(f"✘ OpenMM import/platforms: FAILED ({type(exc).__name__}: {exc})")
        print("\n".join(results))
        raise SystemExit(1)

    # Minimal chromdyn simulation
    try:
        _run_minimal_simulation()
        results.append("✔ Minimal chromdyn simulation: OK")
    except Exception as exc:  # noqa: BLE001
        results.append(
            f"✘ Minimal chromdyn simulation: FAILED ({type(exc).__name__}: {exc})"
        )
        print("\n".join(results))
        raise SystemExit(1)

    print("\n".join(results))
    print("\nAll checks completed successfully.")


if __name__ == "__main__":
    main()