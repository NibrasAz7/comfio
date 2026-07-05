"""Advanced domains tutorial: optional physics-based modules.

Demonstrates Radiance daylighting, CRI/CCT color quality,
RT60 reverberation, STI speech intelligibility, CO2 decay ventilation,
and psychrometric properties.

Requires optional extras:
    pip install comfio[daylighting,color,acoustics,psychrometrics]
"""

import numpy as np

from comfio import (
    calculate_global_ieq,
    evaluate_acoustic,
    evaluate_color_quality,
    evaluate_daylighting,
    evaluate_iaq,
    evaluate_reverberation,
    evaluate_speech_intelligibility,
    evaluate_thermal,
    evaluate_ventilation,
    get_psychrometrics,
    evaluate_visual,
)


def main() -> None:
    # Basic domains
    thermal = evaluate_thermal(
        tdb=np.array([24.0, 25.0, 26.0]),
        tr=np.array([24.0, 25.0, 26.0]),
        vr=np.array([0.1, 0.1, 0.1]),
        rh=np.array([50.0, 50.0, 50.0]),
        met=1.2, clo=0.5,
    )
    visual = evaluate_visual(illuminance=np.array([450.0, 500.0, 600.0]))
    acoustic = evaluate_acoustic(laeq=np.array([35.0, 40.0, 45.0]))
    iaq = evaluate_iaq(co2=np.array([700.0, 900.0, 1100.0]))

    # Advanced: reverberation
    surfaces = {"floor": 50.0, "ceiling": 50.0, "walls": 120.0}
    absorption = {"floor": 0.05, "ceiling": 0.80, "walls": 0.10}
    reverb = evaluate_reverberation(surfaces, absorption, volume=300.0, room_type="office")
    print(f"RT60: {reverb.rt60:.2f}s, score: {reverb.score:.1f}")

    # Advanced: speech intelligibility
    ir_signal = np.random.randn(16000)
    sti = evaluate_speech_intelligibility(ir_signal, sample_rate=16000)
    print(f"STI: {sti.sti:.3f}, score: {sti.score:.1f}")

    # Advanced: ventilation from CO2 decay
    co2_arr = 800.0 * np.exp(-0.5 * np.linspace(0, 4, 100)) + 420.0
    timestamps = np.linspace(0, 4, 100)
    vent = evaluate_ventilation(co2_arr, timestamps, occupancy_type="office")
    print(f"ACH: {vent.ach:.2f}, score: {vent.score:.1f}")

    # Advanced: psychrometrics
    psych = get_psychrometrics(tdb=25.0, rh=0.50)
    print(f"Dew point: {psych.dew_point:.1f}°C, enthalpy: {psych.enthalpy:.1f} kJ/kg")

    # Blend advanced results into Global IEQ
    ieq = calculate_global_ieq(
        thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
        reverberation=reverb, speech_intelligibility=sti, ventilation=vent,
    )
    print(f"\nGlobal IEQ (with advanced): {np.mean(ieq.index):.1f}/100")


if __name__ == "__main__":
    main()
