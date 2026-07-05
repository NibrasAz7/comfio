"""New domains tutorial: pollutant IAQ, sPMV, adaptive, TSV, personalisation.

Demonstrates the newer comfio domain modules that extend
the basic thermal/visual/acoustic/IAQ evaluation.
"""

import numpy as np

from comfio import (
    augment_tsv_cdf,
    calculate_global_ieq,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
    evaluate_iaq,
    evaluate_iaq_pollutants,
    evaluate_personalised_pmv,
    evaluate_spmv,
    evaluate_thermal,
    evaluate_tsv,
    evaluate_visual,
    evaluate_acoustic,
    train_personalisation,
)


def main() -> None:
    # --- Pollutant IAQ ---
    pollutant = evaluate_iaq_pollutants(
        pm25=np.array([8.0, 12.0, 35.0]),
        tvoc=np.array([150.0, 300.0, 500.0]),
        formaldehyde=np.array([20.0, 27.0, 50.0]),
        co=np.array([1.5, 5.0, 10.0]),
        threshold_level="good",
    )
    print(f"Pollutant IAQ score: {pollutant.score}")

    # --- sPMV (simplified PMV) ---
    spmv = evaluate_spmv(
        indoor_temp=np.array([23.0, 24.0, 25.0]),
        indoor_rh=np.array([50.0, 50.0, 50.0]),
        season="mid",
    )
    print(f"sPMV: {spmv.spmv}, score: {spmv.score}")

    # --- Adaptive comfort (ASHRAE 55) ---
    ashrae = evaluate_adaptive_ashrae(
        tdb=np.array([24.0, 25.0, 26.0]),
        tr=np.array([24.0, 25.0, 26.0]),
        t_prevail=20.0,
        acceptability=80,
    )
    print(f"ASHRAE adaptive score: {ashrae.score}")

    # --- Adaptive comfort (EN 16798) ---
    en = evaluate_adaptive_en(
        tdb=np.array([24.0, 25.0, 26.0]),
        tr=np.array([24.0, 25.0, 26.0]),
        t_running_mean=20.0,
        category="ii",
    )
    print(f"EN adaptive score: {en.score}")

    # --- TSV augmentation ---
    augmented = augment_tsv_cdf(
        sparse_votes=np.array([-2, -1, 0, 0, 1, 1, 2, -1, 0, 1]),
        vote_timestamps=np.arange(10),
        target_timestamps=np.arange(100),
    )
    tsv_result = evaluate_tsv(augmented)
    print(f"TSV mean: {tsv_result.mean_tsv:.2f}, compliance: {tsv_result.compliance_rate:.1%}")

    # --- Personalisation ---
    historical_pmv = np.array([0.2, -0.3, 0.5, -0.1, 0.3, 0.0, -0.2, 0.4])
    historical_tsv = np.array([0, -1, 1, 0, 0, -1, -1, 1])
    index = train_personalisation(pmv=historical_pmv, tsv=historical_tsv)
    print(f"Personalisation: alpha={index.alpha:.3f}, beta={index.beta:.3f}")

    pers_result = evaluate_personalised_pmv(
        tdb=np.array([24.0, 25.0, 26.0]),
        tr=np.array([24.0, 25.0, 26.0]),
        vr=np.array([0.1, 0.1, 0.1]),
        rh=np.array([50.0, 50.0, 50.0]),
        met=1.2, clo=0.5,
        personalisation_index=index,
    )
    print(f"Personalised PMV: {pers_result.personalised_pmv}")

    # --- Integration with Global IEQ ---
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

    ieq = calculate_global_ieq(
        thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
        pollutant_iaq=pollutant,
        tsv=tsv_result,
    )
    print(f"\nGlobal IEQ (with new domains): {np.mean(ieq.index):.1f}/100")


if __name__ == "__main__":
    main()
