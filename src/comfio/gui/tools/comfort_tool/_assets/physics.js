/* Comfio Comfort Tool — physics core (pure math, no DOM).
 *
 * PMV/PPD: Fanger model, ISO 7730:2005 (same equations as
 *   pythermalcomfort.models._pmv_ppd_optimized).
 * Relative air speed and dynamic clothing insulation: ASHRAE 55-2023
 *   corrections, matching pythermalcomfort.utilities.v_relative and
 *   clo_dynamic_ashrae.
 * SET two-node model ported from pythermalcomfort (MIT),
 *   doi:10.1016/j.softx.2020.100578.
 * Adaptive models: pythermalcomfort.models.adaptive_ashrae / adaptive_en.
 * Cooling effect: ASHRAE 55 Appendix H definition, matching
 *   pythermalcomfort.models.cooling_effect.
 *
 * All pressures in kPa unless noted (atmospheric pressure default 101.325).
 * All temperatures in C, speeds in m/s. Exposed via CommonJS exports so the
 * same file can be unit-tested under Node.
 */
'use strict';

/* ---------- Fanger PMV / PPD (ISO 7730) ---------- */
function pmvCalc(ta, tr, vel, rh, met, clo) {
  var pa = rh * 10 * Math.exp(16.6536 - 4030.183 / (ta + 235));
  var icl = 0.155 * clo;
  var m = met * 58.15;
  var mw = m; // wme = 0
  var fcl = icl <= 0.078 ? 1 + 1.29 * icl : 1.05 + 0.645 * icl;
  var hcf = 12.1 * Math.sqrt(vel);
  var hc = hcf;
  var taa = ta + 273;
  var tra = tr + 273;
  var tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1);
  var p1 = icl * fcl;
  var p2 = p1 * 3.96;
  var p3 = p1 * 100;
  var p4 = p1 * taa;
  var p5 = 308.7 - 0.028 * mw + p2 * Math.pow(tra / 100, 4);
  var xn = tcla / 100;
  var xf = tcla / 50;
  var eps = 0.00015;
  var n = 0;
  var hcn;
  while (Math.abs(xn - xf) > eps) {
    xf = (xf + xn) / 2;
    hcn = 2.38 * Math.pow(Math.abs(100 * xf - taa), 0.25);
    hc = Math.max(hcn, hcf);
    xn = (p5 + p4 * hc - p2 * Math.pow(xf, 4)) / (100 + p3 * hc);
    n += 1;
    if (n > 150) throw new Error('Max iterations exceeded');
  }
  var tcl = 100 * xn - 273;
  var hl1 = 3.05 * 0.001 * (5733 - 6.99 * mw - pa);
  var hl2 = mw > 58.15 ? 0.42 * (mw - 58.15) : 0;
  var hl3 = 1.7e-5 * m * (5867 - pa);
  var hl4 = 0.0014 * m * (34 - ta);
  var hl5 = 3.96 * fcl * (Math.pow(xn, 4) - Math.pow(tra / 100, 4));
  var hl6 = fcl * hc * (tcl - ta);
  var ts = 0.303 * Math.exp(-0.036 * m) + 0.028;
  return ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6);
}

function ppdCalc(pmv) {
  return 100 - 95 * Math.exp(-0.03353 * Math.pow(pmv, 4) - 0.2179 * Math.pow(pmv, 2));
}

function sensation(pmv) {
  if (pmv < -2.5) return 'Cold';
  if (pmv < -1.5) return 'Cool';
  if (pmv < -0.5) return 'Slightly Cool';
  if (pmv < 0.5) return 'Neutral';
  if (pmv < 1.5) return 'Slightly Warm';
  if (pmv < 2.5) return 'Warm';
  return 'Hot';
}

/* ---------- psychrometrics ---------- */
function pws(t) { return Math.exp(16.6536 - 4030.183 / (t + 235)); } // kPa
function pSatTorr(t) { return Math.exp(18.6686 - 4030.183 / (t + 235)); } // torr
function humRatio(t, rhPct, pAtm) {
  var P = pAtm || 101.325; // kPa
  var p = (rhPct / 100) * pws(t);
  return 622 * p / (P - p); // g/kg dry air
}
function enthalpy(t, w) { return 1.006 * t + (w / 1000) * (2501 + 1.86 * t); }
function dewPoint(t, rhPct) {
  var p = (rhPct / 100) * pws(t);
  if (p <= 0) return -50;
  return 4030.183 / (16.6536 - Math.log(p)) - 235;
}
function wetBulb(t, rhPct) {
  var r = Math.atan;
  var rh = Math.max(0.01, Math.min(100, rhPct));
  return t * r(0.151977 * Math.pow(rh + 8.313659, 0.5)) +
    r(t + rh) - r(rh - 1.676331) +
    0.00391838 * Math.pow(rh, 1.5) * r(0.023101 * rh) - 4.686035;
}
/* inverses: convert a humidity input in another kind to relative humidity % */
function rhFromHumRatio(w, t, pAtm) { // w in g/kg, pAtm in kPa
  var P = pAtm || 101.325;
  var pv = w * P / (622 + w); // kPa, inverse of humRatio
  return 100 * pv / pws(t);
}
function rhFromDewPoint(dp, t) { return 100 * pws(dp) / pws(t); }
function rhFromVaporPressure(vp, t) { return 100 * vp / pws(t); } // vp in kPa
function rhFromWetBulb(wb, t) { // bisection on wetBulb(t, rh)
  var lo = 0.01, hi = 100;
  for (var i = 0; i < 80; i++) {
    var mid = (lo + hi) / 2;
    if (wetBulb(t, mid) < wb) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/* ---------- ASHRAE 55-2023 corrections ---------- */
function vRelative(v, met) { // relative air speed, m/s
  return met > 1 ? Math.round((v + 0.3 * (met - 1)) * 1000) / 1000 : v;
}
function cloDynamicAshrae(clo, met) { // dynamic clothing insulation, clo
  return met > 1.2 ? Math.round(clo * (0.6 + 0.4 / met) * 1000) / 1000 : clo;
}

/* ---------- operative temperature ---------- */
function operativeTmpAshrae(ta, tr, v) {
  var a = v < 0.2 ? 0.5 : v < 0.6 ? 0.6 : 0.7;
  return a * ta + (1 - a) * tr;
}
function operativeTmpIso(ta, tr, v) {
  return (ta * Math.sqrt(10 * v) + tr) / (1 + Math.sqrt(10 * v));
}

/* ---------- Gagge two-node model / SET ----------
 * Faithful port of pythermalcomfort's _gagge_two_nodes_optimized (MIT),
 * doi:10.1016/j.softx.2020.100578 — same constants, iteration structure and
 * convergence criteria. Options: wme [met], bodySurfaceArea [m2],
 * pAtm [kPa], position ('standing'|'sitting'), calculateCe (Appendix H mode:
 * disables the metabolic convection correction). Returns model internals
 * including .set (Standard Effective Temperature).
 */
function twoNodesGagge(tdb, tr, v, rh, met, clo, o) {
  o = o || {};
  var wme = o.wme || 0;
  var body_surface_area = o.bodySurfaceArea || 1.8258;
  var p_atm = o.pAtm || 101.325; // kPa
  var position = o.position || 'standing';
  var calculate_ce = !!o.calculateCe;
  var max_skin_blood_flow = o.maxSkinBloodFlow || 90;
  var max_sweating = o.maxSweating || 500;
  var w_max = o.wMax || 0;
  var MET_TO_W_M2 = 58.15;

  var vapor_pressure = rh * pSatTorr(tdb) / 100; // torr

  // Initial variables as defined in the ASHRAE 55-2020
  var air_speed = Math.max(v, 0.1);
  var k_clo = 0.25;
  var body_weight = 70; // body weight in kg
  var met_factor = 58.2; // met conversion factor
  var sbc = 0.000000056697; // Stefan-Boltzmann constant (W/m2K4)
  var c_sw = 170; // driving coefficient for regulatory sweating
  var c_dil = 120; // driving coefficient for vasodilation
  var c_str = 0.5; // driving coefficient for vasoconstriction

  var temp_skin_neutral = 33.7;
  var temp_core_neutral = 36.8;
  var alfa = 0.1;
  var temp_body_neutral = alfa * temp_skin_neutral + (1 - alfa) * temp_core_neutral;
  var skin_blood_flow_neutral = 6.3;

  var t_skin = temp_skin_neutral;
  var t_core = temp_core_neutral;
  var m_bl = skin_blood_flow_neutral;

  var e_skin = 0.1 * met; // total evaporative heat loss, W
  var q_sensible = 0; // total sensible heat loss, W
  var w = 0; // skin wettedness
  var _set = 0; // standard effective temperature
  var e_rsw = 0; // heat lost by vaporization sweat
  var e_diff = 0; // vapor diffusion through skin
  var e_max = 0; // maximum evaporative capacity
  var m_rsw = 0; // regulatory sweating
  var q_res = 0; // heat loss due to respiration
  var et = 0; // effective temperature
  var e_req = 0; // evaporative heat loss required for tmp regulation
  var r_ea = 0;
  var r_ecl = 0;
  var c_res = 0; // convective heat loss respiration

  var pressure_in_atmospheres = p_atm / 101.325;
  var length_time_simulation = 60; // length time simulation
  var n_simulation = 1;

  var r_clo = 0.155 * clo; // thermal resistance of clothing, C M^2 /W
  var f_a_cl = 1.0 + 0.15 * clo; // increase in body surface area due to clothing
  var lr = 2.2 / pressure_in_atmospheres; // Lewis ratio
  var rm = (met - wme) * met_factor; // metabolic rate
  var m = met * met_factor; // metabolic rate

  var e_comfort = 0.42 * (rm - met_factor); // evaporative heat loss during comfort
  e_comfort = Math.max(e_comfort, 0);

  var i_cl = 1.0; // permeation efficiency of water vapour naked skin
  if (clo > 0) i_cl = 0.45; // through the clothing layer

  if (!w_max) { // if the user did not pass a value of w_max
    w_max = 0.38 * Math.pow(air_speed, -0.29); // critical skin wettedness naked
    if (clo > 0) w_max = 0.59 * Math.pow(air_speed, -0.08); // clothed
  }

  // h_cc corrected convective heat transfer coefficient
  var h_cc = 3.0 * Math.pow(pressure_in_atmospheres, 0.53);
  // h_fc forced convective heat transfer coefficient, W/(m2 C)
  var h_fc = 8.600001 * Math.pow(air_speed * pressure_in_atmospheres, 0.53);
  h_cc = Math.max(h_cc, h_fc);
  if (!calculate_ce && met > 0.85) {
    var h_c_met = 5.66 * Math.pow(met - 0.85, 0.39);
    h_cc = Math.max(h_cc, h_c_met);
  }

  var h_r = 4.7; // linearized radiative heat transfer coefficient
  var h_t = h_r + h_cc; // sum of convective and radiant heat transfer coefficient
  var r_a = 1.0 / (f_a_cl * h_t); // resistance of air layer to dry heat
  var t_op = (h_r * tr + h_cc * tdb) / h_t; // operative temperature

  var t_body = alfa * t_skin + (1 - alfa) * t_core; // mean body temperature, C

  // respiration
  q_res = 0.0023 * m * (44.0 - vapor_pressure); // latent heat loss due to respiration
  c_res = 0.0014 * m * (34.0 - tdb); // sensible convective heat loss respiration

  var t_cl, t_cl_new, hf_cs, s_core, s_skin, tc_sk, tc_cr, d_t_sk, d_t_cr;
  var sk_sig, warm_sk, colds, c_reg_sig, c_warm, c_cold, bd_sig, warm_b;
  var p_rsw, met_shivering;

  while (n_simulation < length_time_simulation) {
    n_simulation += 1;

    var iteration_limit = 150; // for following while loop
    // t_cl temperature of the outer surface of clothing
    t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo); // initial guess
    var n_iterations = 0;
    var tc_converged = false;

    while (!tc_converged) {
      // 0.95 is the clothing emissivity from ASHRAE fundamentals Ch. 9.7 Eq. 35
      if (position === 'sitting') {
        // 0.7 ratio between radiation area of the body and the body area
        h_r = 4.0 * 0.95 * sbc * Math.pow((t_cl + tr) / 2.0 + 273.15, 3.0) * 0.7;
      } else { // if standing
        // 0.73 ratio between radiation area of the body and the body area
        h_r = 4.0 * 0.95 * sbc * Math.pow((t_cl + tr) / 2.0 + 273.15, 3.0) * 0.73;
      }
      h_t = h_r + h_cc;
      r_a = 1.0 / (f_a_cl * h_t);
      t_op = (h_r * tr + h_cc * tdb) / h_t;
      t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo);
      if (Math.abs(t_cl_new - t_cl) <= 0.01) tc_converged = true;
      t_cl = t_cl_new;
      n_iterations += 1;
      if (n_iterations > iteration_limit) throw new Error('Max iterations exceeded');
    }

    q_sensible = (t_skin - t_op) / (r_a + r_clo); // total sensible heat loss, W
    // hf_cs rate of energy transport between core and skin, W
    // 5.28 is the average body tissue conductance in W/(m2 C)
    // 1.163 is the thermal capacity of blood in Wh/(L C)
    hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl);
    s_core = m - hf_cs - q_res - c_res - wme; // rate of energy storage in the core
    s_skin = hf_cs - q_sensible - e_skin; // rate of energy storage in the skin
    tc_sk = 0.97 * alfa * body_weight; // thermal capacity skin
    tc_cr = 0.97 * (1 - alfa) * body_weight; // thermal capacity core
    d_t_sk = (s_skin * body_surface_area) / (tc_sk * 60.0); // rate of change skin temperature C per minute
    d_t_cr = s_core * body_surface_area / (tc_cr * 60.0); // rate of change core temperature C per minute
    t_skin = t_skin + d_t_sk;
    t_core = t_core + d_t_cr;
    t_body = alfa * t_skin + (1 - alfa) * t_core;
    // sk_sig thermoregulatory control signal from the skin
    sk_sig = t_skin - temp_skin_neutral;
    warm_sk = sk_sig > 0 ? sk_sig : 0; // vasodilation signal
    colds = -sk_sig > 0 ? -sk_sig : 0; // vasoconstriction signal
    // c_reg_sig thermoregulatory control signal from the skin, C
    c_reg_sig = t_core - temp_core_neutral;
    c_warm = c_reg_sig > 0 ? c_reg_sig : 0; // vasodilation signal
    c_cold = -c_reg_sig > 0 ? -c_reg_sig : 0; // vasoconstriction signal
    // bd_sig thermoregulatory control signal from the body
    bd_sig = t_body - temp_body_neutral;
    warm_b = bd_sig > 0 ? bd_sig : 0;
    m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1 + c_str * colds);
    m_bl = Math.min(m_bl, max_skin_blood_flow);
    m_bl = Math.max(m_bl, 0.5);
    m_rsw = c_sw * warm_b * Math.exp(warm_sk / 10.7); // regulatory sweating
    m_rsw = Math.min(m_rsw, max_sweating);
    e_rsw = 0.68 * m_rsw; // heat lost by vaporization sweat
    r_ea = 1.0 / (lr * f_a_cl * h_cc); // evaporative resistance air layer
    r_ecl = r_clo / (lr * i_cl);
    e_req = rm - q_res - c_res - q_sensible; // evaporative heat loss required for tmp regulation
    e_max = (Math.exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / (r_ea + r_ecl);
    if (e_max === 0) e_max = 0.001;
    p_rsw = e_rsw / e_max; // ratio heat loss sweating to max heat loss sweating
    w = 0.06 + 0.94 * p_rsw; // skin wetness
    e_diff = w * e_max - e_rsw; // vapor diffusion through skin
    if (w > w_max) {
      w = w_max;
      p_rsw = w_max / 0.94;
      e_rsw = p_rsw * e_max;
      e_diff = 0.06 * (1.0 - p_rsw) * e_max;
    }
    if (e_max < 0) {
      e_diff = 0;
      e_rsw = 0;
      w = w_max;
    }
    e_skin = e_rsw + e_diff; // total evaporative heat loss sweating and vapor diffusion
    m_rsw = e_rsw / 0.68; // back calculating the mass of regulatory sweating
    met_shivering = 19.4 * colds * c_cold; // met shivering W/m2
    m = rm + met_shivering;
    alfa = 0.0417737 + 0.7451833 / (m_bl + 0.585417);
  }

  var q_skin = q_sensible + e_skin; // total heat loss from skin, W
  // p_s_sk saturation vapour pressure of water of the skin
  var p_s_sk = Math.exp(18.6686 - 4030.183 / (t_skin + 235.0));

  // standard environment - where _s at end of the variable names stands for standard
  var h_r_s = h_r; // standard environment radiative heat transfer coefficient

  var h_c_s = 3.0 * Math.pow(pressure_in_atmospheres, 0.53);
  if (!calculate_ce && met > 0.85) {
    h_c_s = Math.max(h_c_s, 5.66 * Math.pow(met - 0.85, 0.39));
  }
  h_c_s = Math.max(h_c_s, 3.0);

  var h_t_s = h_c_s + h_r_s; // sum of convective and radiant heat transfer coefficient
  var r_clo_s = 1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835; // thermal resistance of clothing, C M^2 /W
  var r_cl_s = 0.155 * r_clo_s; // thermal insulation of the clothing in M2K/W
  var f_a_cl_s = 1.0 + k_clo * r_clo_s; // increase in body surface area due to clothing
  var f_cl_s = 1.0 / (1.0 + 0.155 * f_a_cl_s * h_t_s * r_clo_s); // ratio of surface clothed body over nude body
  var i_m_s = 0.45; // permeation efficiency of water vapour through the clothing layer
  var i_cl_s = i_m_s * h_c_s / h_t_s * (1 - f_cl_s) / (h_c_s / h_t_s - f_cl_s * i_m_s); // clothing vapor permeation efficiency
  var r_a_s = 1.0 / (f_a_cl_s * h_t_s); // resistance of air layer to dry heat
  var r_ea_s = 1.0 / (lr * f_a_cl_s * h_c_s);
  var r_ecl_s = r_cl_s / (lr * i_cl_s);
  var h_d_s = 1.0 / (r_a_s + r_cl_s);
  var h_e_s = 1.0 / (r_ea_s + r_ecl_s);

  // calculate Standard Effective Temperature (SET)
  var delta = 0.0001;
  var dx = 100.0;
  var err_1, err_2;
  var set_old = Math.round((t_skin - q_skin / h_d_s) * 100) / 100;
  var guard = 0;
  while (Math.abs(dx) > 0.01) {
    err_1 = q_skin - h_d_s * (t_skin - set_old) -
      w * h_e_s * (p_s_sk - 0.5 * Math.exp(18.6686 - 4030.183 / (set_old + 235.0)));
    err_2 = q_skin - h_d_s * (t_skin - (set_old + delta)) -
      w * h_e_s * (p_s_sk - 0.5 * Math.exp(18.6686 - 4030.183 / (set_old + delta + 235.0)));
    _set = set_old - delta * err_1 / (err_2 - err_1);
    dx = _set - set_old;
    set_old = _set;
    if (++guard > 1000) break;
  }

  // calculate Effective Temperature (ET)
  var h_d = 1 / (r_a + r_clo);
  var h_e = 1 / (r_ea + r_ecl);
  var et_old = t_skin - q_skin / h_d;
  dx = 100.0;
  guard = 0;
  while (Math.abs(dx) > 0.01) {
    err_1 = q_skin - h_d * (t_skin - et_old) -
      w * h_e * (p_s_sk - 0.5 * Math.exp(18.6686 - 4030.183 / (et_old + 235.0)));
    err_2 = q_skin - h_d * (t_skin - (et_old + delta)) -
      w * h_e * (p_s_sk - 0.5 * Math.exp(18.6686 - 4030.183 / (et_old + delta + 235.0)));
    et = et_old - delta * err_1 / (err_2 - err_1);
    dx = et - et_old;
    et_old = et;
    if (++guard > 1000) break;
  }

  var tbm_l = (0.194 / MET_TO_W_M2) * rm + 36.301; // lower limit for evaporative regulation
  var tbm_h = (0.347 / MET_TO_W_M2) * rm + 36.669; // upper limit for evaporative regulation

  var t_sens = 0.4685 * (t_body - tbm_l); // predicted thermal sensation
  if (t_body >= tbm_l && t_body < tbm_h) {
    t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l);
  } else if (t_body >= tbm_h) {
    t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h);
  }

  var disc;
  if (t_sens > 0 && 0 > e_max * w_max - e_comfort - e_diff) {
    disc = 6.0;
  } else {
    disc = 4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff); // predicted thermal discomfort
  }
  if (disc <= 0) disc = t_sens;
  if (disc > 6) disc = 6.0;

  // PMV Gagge
  var pmv_gagge = (0.303 * Math.exp(-0.036 * m) + 0.028) * (e_req - e_comfort - e_diff);

  // PMV SET
  var dry_set = h_d_s * (t_skin - _set);
  var e_req_set = rm - c_res - q_res - dry_set;
  var pmv_set = (0.303 * Math.exp(-0.036 * m) + 0.028) * (e_req_set - e_comfort - e_diff);

  return {
    set: _set,
    e_skin: e_skin,
    e_rsw: e_rsw,
    e_max: e_max,
    q_sensible: q_sensible,
    q_skin: q_skin,
    q_res: q_res,
    t_core: t_core,
    t_skin: t_skin,
    m_bl: m_bl,
    m_rsw: m_rsw,
    w: w,
    w_max: w_max,
    et: et,
    pmv_gagge: pmv_gagge,
    pmv_set: pmv_set,
    disc: disc,
    t_sens: t_sens
  };
}

/* Standard Effective Temperature [C], rounded like pythermalcomfort set_tmp. */
function setTmp(tdb, tr, v, rh, met, clo, pAtm) {
  var r = twoNodesGagge(tdb, tr, v, rh, met, clo, { pAtm: pAtm || 101.325 });
  return Math.round(r.set * 10) / 10;
}

/* ---------- Cooling Effect (ASHRAE 55 Appendix H) ----------
 * The CE of the elevated air speed is the value that, when subtracted equally
 * from both the air temperature and the mean radiant temperature, yields the
 * same SET under still air (0.1 m/s) as the first SET calculation under the
 * elevated air speed. Computed only for air speed higher than 0.1 m/s.
 * Matches pythermalcomfort.models.cooling_effect. */
function _setCe(tdb, tr, v, rh, met, clo, wme, pAtm) {
  return twoNodesGagge(tdb, tr, v, rh, met, clo,
    { wme: wme, pAtm: pAtm, calculateCe: true }).set;
}
function coolingEffect(tdb, tr, vr, rh, met, clo, wme, pAtm) {
  wme = wme || 0;
  pAtm = pAtm || 101.325;
  if (vr <= 0.1) return 0;
  var still_air = 0.1;
  var initial_set = _setCe(tdb, tr, vr, rh, met, clo, wme, pAtm);
  var f = function (x) {
    return _setCe(tdb - x, tr - x, still_air, rh, met, clo, wme, pAtm) - initial_set;
  };
  // root of f over [0, 40]: bisection (f(0) >= 0, f decreasing in x)
  var lo = 0.0, hi = 40.0;
  var flo = f(lo), fhi = f(hi);
  var ce = 0;
  if (flo * fhi <= 0) {
    for (var i = 0; i < 60; i++) {
      var mid = (lo + hi) / 2;
      var fm = f(mid);
      if (hi - lo < 0.0005) { ce = mid; break; }
      if (flo * fm <= 0) { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
      ce = mid;
    }
  }
  return Math.round(ce * 100) / 100;
}

/* ---------- adaptive models ---------- */
function adaptiveCoolingEffect(v, to) { // [C], ASHRAE 55 elevated-air-speed table
  var mag = v >= 1.2 ? 2.2 : v >= 0.9 ? 1.8 : v >= 0.6 ? 1.2 : 0;
  return to >= 25 ? mag : 0;
}
function round1(x) { return Math.round(x * 10) / 10; }

/* pythermalcomfort.models.adaptive_ashrae (SI). ta/tr/to [C], tRm [C], v [m/s]. */
function adaptiveAshrae(ta, tr, tRm, v) {
  var to = operativeTmpAshrae(ta, tr, v);
  var ce = adaptiveCoolingEffect(v, to);
  var valid = !(ta < 10 || ta > 40 || tr < 10 || tr > 40 ||
    v < 0 || v > 2 || tRm < 10 || tRm > 33.5);
  var tCmf = valid ? round1(0.31 * tRm + 17.8) : NaN;
  var lo80 = tCmf - 3.5, up80 = tCmf + 3.5 + ce;
  var lo90 = tCmf - 2.5, up90 = tCmf + 2.5 + ce;
  return {
    to: to, ce: ce, valid: valid, tCmf: tCmf,
    tCmf80Low: lo80, tCmf80Up: up80,
    tCmf90Low: lo90, tCmf90Up: up90,
    acceptability80: valid && lo80 <= to && to <= up80,
    acceptability90: valid && lo90 <= to && to <= up90
  };
}

/* pythermalcomfort.models.adaptive_en (SI). Bands per EN 16798-1:2019. */
function adaptiveEn(ta, tr, tRm, v) {
  var to = operativeTmpIso(ta, tr, v);
  var ce = adaptiveCoolingEffect(v, to);
  var valid = !(tRm < 10 || tRm > 33.5);
  var tCmf = valid ? 0.33 * tRm + 18.8 : NaN;
  var iLo = tCmf - 3.0, iiLo = tCmf - 4.0, iiiLo = tCmf - 5.0;
  var iUp = tCmf + 2.0 + ce, iiUp = tCmf + 3.0 + ce, iiiUp = tCmf + 4.0 + ce;
  return {
    to: to, ce: ce, valid: valid, tCmf: round1(tCmf),
    catI: { low: round1(iLo), up: round1(iUp) },
    catII: { low: round1(iiLo), up: round1(iiUp) },
    catIII: { low: round1(iiiLo), up: round1(iiiUp) },
    acceptabilityI: valid && iLo <= to && to <= iUp,
    acceptabilityII: valid && iiLo <= to && to <= iiUp,
    acceptabilityIII: valid && iiiLo <= to && to <= iiiUp
  };
}

/* ---------- ASHRAE 55 PMV entry point ----------
 * Applies the relative-air-speed and dynamic-clothing corrections, then the
 * Appendix H cooling effect when vr > 0.1 — identical to
 * pythermalcomfort.models.pmv_ppd_ashrae. `airspeedControl` selects the
 * occupant-local-control validity limits (Table: v caps without control). */
function ashrae55Valid(ta, tr, v, met, clo, airspeedControl) {
  if (ta < 10 || ta > 40 || tr < 10 || tr > 40 ||
    v < 0 || v > 2 || met < 1 || met > 4 || clo < 0 || clo > 1.5) return false;
  if (!airspeedControl) {
    if (v > 0.8 && clo < 0.7 && met < 1.3) return false;
    var to = operativeTmpIso(ta, tr, v);
    var vLimit = 50.49 - 4.4047 * to + 0.096425 * to * to;
    if (to > 23 && to < 25.5 && v > vLimit && clo < 0.7 && met < 1.3) return false;
    if (to <= 23 && v > 0.2 && clo < 0.7 && met < 1.3) return false;
  }
  return true;
}
function pmvAshrae(ta, tr, v, rh, met, clo, o) {
  o = o || {};
  var pAtm = o.pAtm || 101.325;
  var airspeedControl = o.airspeedControl !== false; // default true
  var vr = vRelative(v, met);
  var cloD = cloDynamicAshrae(clo, met);
  var ce = 0;
  if (vr > 0.1) ce = coolingEffect(ta, tr, vr, rh, met, cloD, 0, pAtm);
  var vrCe = ce > 0 ? 0.1 : vr;
  var pmv = pmvCalc(ta - ce, tr - ce, vrCe, rh, met, cloD);
  var valid = ashrae55Valid(ta, tr, vr, met, cloD, airspeedControl);
  return { pmv: pmv, ppd: ppdCalc(pmv), ce: ce, vr: vr, cloDyn: cloD, valid: valid };
}

/* ---------- exports for Node testing ---------- */
var ComfioPhysics = {
  pmvCalc: pmvCalc,
  ppdCalc: ppdCalc,
  sensation: sensation,
  pws: pws,
  pSatTorr: pSatTorr,
  humRatio: humRatio,
  enthalpy: enthalpy,
  dewPoint: dewPoint,
  wetBulb: wetBulb,
  rhFromHumRatio: rhFromHumRatio,
  rhFromDewPoint: rhFromDewPoint,
  rhFromVaporPressure: rhFromVaporPressure,
  rhFromWetBulb: rhFromWetBulb,
  vRelative: vRelative,
  cloDynamicAshrae: cloDynamicAshrae,
  operativeTmpAshrae: operativeTmpAshrae,
  operativeTmpIso: operativeTmpIso,
  twoNodesGagge: twoNodesGagge,
  setTmp: setTmp,
  coolingEffect: coolingEffect,
  adaptiveCoolingEffect: adaptiveCoolingEffect,
  adaptiveAshrae: adaptiveAshrae,
  adaptiveEn: adaptiveEn,
  ashrae55Valid: ashrae55Valid,
  pmvAshrae: pmvAshrae
};
if (typeof module !== 'undefined' && module.exports) module.exports = ComfioPhysics;
