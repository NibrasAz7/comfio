/* Comfio Comfort Tool — UI layer (DOM only; all physics lives in physics.js).
 * Single-page app: the five tool pages are merged into sections switched
 * by the nav tab bar. All controllers are initialised once at load. */
(function () {
  'use strict';

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function pageRoot(name) { return $('section[data-page="' + name + '"]'); }

  /* ---------- units (SI/IP) and atmospheric pressure ---------- */
  var unitsIP = false;
  var atmP = 101.325; // kPa

  function recalc() { document.dispatchEvent(new Event('recalc')); }
  function toSI(v, kind) {
    if (!unitsIP) return v;
    if (kind === 't') return (v - 32) * 5 / 9;
    if (kind === 'v') return v / 196.85;
    return v;
  }
  function convDisp(v, kind, toIP) {
    if (kind === 't') return toIP ? v * 9 / 5 + 32 : (v - 32) * 5 / 9;
    if (kind === 'v') return toIP ? v * 196.85 : v / 196.85;
    return v;
  }

  /* ---------- number spinners ---------- */
  function wireSpins(scope) {
    $$('.num-wrap', scope).forEach(function (wrap) {
      if (wrap.__spinWired) return;
      wrap.__spinWired = true;
      var input = $('input', wrap);
      var up = $('.spin-up', wrap);
      var down = $('.spin-down', wrap);
      if (!input) return;
      var step = parseFloat(input.getAttribute('data-step') || '1');
      function bump(dir) {
        var v = parseFloat(input.value) || 0;
        var min = input.hasAttribute('data-min') ? parseFloat(input.getAttribute('data-min')) : -Infinity;
        var max = input.hasAttribute('data-max') ? parseFloat(input.getAttribute('data-max')) : Infinity;
        v = Math.min(max, Math.max(min, v + dir * step));
        var dec = (String(step).split('.')[1] || '').length;
        input.value = v.toFixed(dec);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
      if (up) up.addEventListener('click', function () { bump(1); });
      if (down) down.addEventListener('click', function () { bump(-1); });
    });
  }

  /* ---------- accessible names for unnamed controls ---------- */
  $$('input').forEach(function (i) {
    if (i.getAttribute('aria-label')) return;
    var l = i.id && document.querySelector('label[for="' + i.id + '"]');
    i.setAttribute('aria-label', l ? l.textContent.trim() : (i.id || i.name || i.type));
  });

  /* ---------- preset selects that set a numeric input (delegated) ---------- */
  document.addEventListener('change', function (e) {
    var sel = e.target;
    if (!sel.matches || !sel.matches('select[data-set]')) return;
    var t = document.getElementById(sel.getAttribute('data-set'));
    if (!t) return;
    t.value = sel.value;
    t.dispatchEvent(new Event('input', { bubbles: true }));
    t.dispatchEvent(new Event('change', { bubbles: true }));
  });

  /* ---------- humidity input kinds ---------- */
  /* select.hum-sel order: 0 RH % | 1 humidity ratio g/kg | 2 dew point | 3 wet bulb | 4 vapor pressure kPa */
  function humUnitLabel(kind) {
    var labels = ['%', 'g/kg', '\u00B0C', '\u00B0C', 'kPa'];
    var s = labels[kind] || '%';
    if (unitsIP && (kind === 2 || kind === 3)) s = '\u00B0F';
    return s;
  }
  function humidityToRh(kind, val, t) { // t in C (SI)
    switch (kind) {
      case 1: return rhFromHumRatio(val, t, atmP);
      case 2: return rhFromDewPoint(toSI(val, 't'), t);
      case 3: return rhFromWetBulb(toSI(val, 't'), t);
      case 4: return rhFromVaporPressure(val, t);
      default: return val;
    }
  }
  function rhToDisplay(kind, rh, t) {
    var v;
    switch (kind) {
      case 1: v = humRatio(t, rh, atmP); break;
      case 2: v = dewPoint(t, rh); break;
      case 3: v = wetBulb(t, rh); break;
      case 4: v = rh / 100 * pws(t); break;
      default: v = rh;
    }
    if ((kind === 2 || kind === 3) && unitsIP) v = convDisp(v, 't', true);
    return v;
  }
  function updateHumUnitLabels() {
    $$('select.hum-sel').forEach(function (s) {
      var line = s.closest('.field-line');
      var u = line && $('.hum-unit', line);
      if (u) u.textContent = humUnitLabel(s.selectedIndex);
    });
  }
  /* wire one humidity-kind select; getT() returns the reference air
   * temperature in C used for the humidity conversions */
  function wireHumSel(sel, getT) {
    if (!sel || sel.__humWired) return;
    sel.__humWired = true;
    sel.__kind = sel.selectedIndex;
    sel.__getT = getT;
    sel.addEventListener('change', function () {
      var line = sel.closest('.field-line');
      var input = line && $('.num-wrap input', line);
      var unit = line && $('.hum-unit', line);
      var kind = sel.selectedIndex;
      if (input) {
        var t = sel.__getT();
        var rh = humidityToRh(sel.__kind, parseFloat(input.value) || 0, t);
        input.value = Math.round(rhToDisplay(kind, rh, t) * 100) / 100;
        /* dew point / wet bulb are temperatures: let the SI/IP toggle convert them */
        if (kind === 2 || kind === 3) input.setAttribute('data-kind', 't');
        else input.removeAttribute('data-kind');
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (unit) unit.textContent = humUnitLabel(kind);
      sel.__kind = kind;
      recalc();
    });
  }

  /* ---------- units toggle (synced across all sections) ---------- */
  $$('.opt-units').forEach(function (c) {
    c.addEventListener('change', function () {
      var toIP = c.checked;
      $$('.opt-units').forEach(function (o) { o.checked = toIP; });
      $$('input[data-kind]').forEach(function (i) {
        var v = parseFloat(i.value);
        if (!isNaN(v)) {
          var nv = convDisp(v, i.getAttribute('data-kind'), toIP);
          i.value = Math.round(nv * 100) / 100;
        }
      });
      unitsIP = toIP;
      $$('.u-t').forEach(function (s) { s.textContent = toIP ? '\u00B0F' : '\u00B0C'; });
      $$('.u-v').forEach(function (s) { s.textContent = toIP ? 'fpm' : 'm/s'; });
      updateHumUnitLabels();
      recalc();
    });
  });

  /* ---------- pressure rows ---------- */
  $$('.opt-press').forEach(function (c) {
    c.addEventListener('change', function () {
      $$('.opt-press').forEach(function (o) {
        o.checked = c.checked;
        var row = document.getElementById(o.getAttribute('data-row'));
        if (row) row.classList.toggle('show', c.checked);
      });
    });
  });
  $$('.pressure-sel').forEach(function (s) {
    s.addEventListener('change', function () {
      atmP = parseFloat(s.value) || 101.325;
      $$('.pressure-sel').forEach(function (o) { o.value = s.value; });
      recalc();
    });
  });

  /* ---------- input helpers ---------- */
  function num(scope, id) {
    var e = $('#' + id, scope || document) || document.getElementById(id);
    return e ? (parseFloat(e.value) || 0) : 0;
  }
  function onInputs(scope, fn) {
    $$('input, select', scope).forEach(function (i) {
      i.addEventListener('input', fn);
      i.addEventListener('change', fn);
    });
  }
  function fmt(v, dec) {
    if (v !== v) return '—';
    var s = v.toFixed(dec);
    return s === '-0' || s === '-0.0' || s === '-0.00' ? s.slice(1) : s;
  }

  /* ---------- psychrometric-style chart ---------- */
  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(x, y, str, attrs, parent) {
    var t = el('text', attrs || {}, parent);
    t.setAttribute('x', x);
    t.setAttribute('y', y);
    t.textContent = str;
    return t;
  }

  var CH = { x0: 72, x1: 520, y0: 20, y1: 440, tmin: 10, tmax: 36, wmax: 30 };
  function px(t) { return CH.x0 + (t - CH.tmin) / (CH.tmax - CH.tmin) * (CH.x1 - CH.x0); }
  function py(w) { return CH.y1 - (w / CH.wmax) * (CH.y1 - CH.y0); }

  // find t where f(t) crosses `target`, scanning t in [10,36]; f must be
  // monotone increasing in t (PMV on the tdb=tr axis is).
  function solveT(target, f) {
    var t0 = null, f0 = null;
    for (var t = CH.tmin; t <= CH.tmax + 0.001; t += 1.0) {
      var ft = f(t) - target;
      if (f0 !== null && f0 * ft <= 0 && ft !== f0) {
        var a = t0, b = t, fa = f0;
        for (var i = 0; i < 40; i++) {
          var m = (a + b) / 2;
          var fm = f(m) - target;
          if (fa * fm <= 0) { b = m; } else { a = m; fa = fm; }
        }
        return (a + b) / 2;
      }
      t0 = t; f0 = ft;
    }
    return null;
  }

  function zonePath(tol, f) { // f(t, rh) -> pmv
    var left = [], right = [];
    for (var rh = 95; rh >= 5; rh -= 5) {
      var lo = solveT(-tol, function (t) { return f(t, rh); });
      var hi = solveT(tol, function (t) { return f(t, rh); });
      if (lo !== null && hi !== null) {
        left.push([lo, humRatio(lo, rh, atmP)]);
        right.push([hi, humRatio(hi, rh, atmP)]);
      }
    }
    var pts = left.concat(right.reverse());
    if (!pts.length) return null;
    return pts.map(function (p, i) {
      return (i ? 'L' : 'M') + px(p[0]).toFixed(1) + ',' + py(p[1]).toFixed(1);
    }).join(' ') + ' Z';
  }

  function drawChart(svg, opts) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('viewBox', '0 0 580 500');
    var clip = el('clipPath', { id: 'clip' + Math.floor(Math.random() * 1e6) }, svg);
    el('rect', { x: CH.x0, y: CH.y0, width: CH.x1 - CH.x0, height: CH.y1 - CH.y0 }, clip);
    var clipId = clip.getAttribute('id');

    // RH curves
    var g = el('g', { 'clip-path': 'url(#' + clipId + ')' }, svg);
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].forEach(function (rh) {
      var d = '';
      for (var t = CH.tmin; t <= CH.tmax + 0.01; t += 0.5) {
        var w = humRatio(t, rh, atmP);
        d += (d ? 'L' : 'M') + px(t).toFixed(1) + ',' + py(w).toFixed(1);
      }
      el('path', { d: d, fill: 'none', stroke: '#c4c4c4', 'stroke-width': 1 }, g);
    });

    // comfort zones (outermost first)
    (opts.zones || []).forEach(function (z) {
      var d = zonePath(z.tol, z.f);
      if (d) el('path', { d: d, fill: z.fill, 'fill-opacity': z.opacity == null ? 0.9 : z.opacity }, g);
    });

    // extra path (ranges page red line)
    if (opts.line && opts.line.length > 1) {
      var dl = opts.line.map(function (p, i) {
        return (i ? 'L' : 'M') + px(p[0]).toFixed(1) + ',' + py(p[1]).toFixed(1);
      }).join(' ');
      el('path', { d: dl, fill: 'none', stroke: '#d62728', 'stroke-width': 2.2 }, g);
    }

    // axes
    for (var t2 = CH.tmin; t2 <= CH.tmax; t2 += 2) {
      txt(px(t2), CH.y1 + 18, String(t2), { 'font-size': '12', 'text-anchor': 'middle', fill: '#212529', 'font-family': 'sans-serif' }, svg);
    }
    for (var w2 = 0; w2 <= CH.wmax; w2 += 5) {
      txt(CH.x1 + 8, py(w2) + 4, String(w2), { 'font-size': '12', fill: '#212529', 'font-family': 'sans-serif' }, svg);
    }
    txt((CH.x0 + CH.x1) / 2, CH.y1 + 42, opts.xLabel, { 'font-size': '14', 'text-anchor': 'middle', fill: '#212529', 'font-family': 'sans-serif' }, svg);
    var ry = el('text', { 'font-size': '14', 'text-anchor': 'middle', fill: '#212529', 'font-family': 'sans-serif', transform: 'translate(' + (CH.x1 + 34) + ' ' + (CH.y0 + CH.y1) / 2 + ') rotate(-90)' }, svg);
    ry.textContent = 'Humidity Ratio [g';
    var sub = el('tspan', { 'baseline-shift': 'sub', 'font-size': '10' }, ry);
    sub.textContent = 'w';
    var tail = el('tspan', {}, ry);
    tail.textContent = ' / kg';
    var tailSub = el('tspan', { 'baseline-shift': 'sub', 'font-size': '10' }, ry);
    tailSub.textContent = 'da';
    var tail2 = el('tspan', {}, ry);
    tail2.textContent = ']';

    // psych values legend (top-left inside plot)
    if (opts.legend) {
      var lg = opts.legend;
      var items = [
        ['t', 'db', lg.tdb.toFixed(1), ' °C'],
        ['rh', '', lg.rh.toFixed(1), ' %'],
        ['W', 'a', lg.w.toFixed(1), ' g w/kg da'],
        ['t', 'wb', lg.twb.toFixed(1), ' °C'],
        ['t', 'dp', lg.tdp.toFixed(1), ' °C'],
        ['h', '', lg.h.toFixed(1), ' kJ/kg']
      ];
      items.forEach(function (it, i) {
        var ty = 40 + i * 20;
        var te = txt(CH.x0 + 10, ty, '', { 'font-size': '12', 'font-family': 'monospace', fill: '#212529' }, svg);
        var s1 = el('tspan', {}, te); s1.textContent = it[0];
        if (it[1]) { var s2 = el('tspan', { 'baseline-shift': 'sub', 'font-size': '9' }, te); s2.textContent = it[1]; }
        var s3 = el('tspan', {}, te); s3.textContent = '  ' + it[2];
        var s4 = el('tspan', { 'font-size': '10' }, te); s4.textContent = it[3];
      });
    }

    // points
    (opts.points || []).forEach(function (p) {
      el('circle', { cx: px(p.t), cy: py(p.w), r: 8, fill: 'none', stroke: p.color, 'stroke-width': 2.4 }, svg);
      el('circle', { cx: px(p.t), cy: py(p.w), r: 2.6, fill: p.color }, svg);
    });
  }

  /* ---------- adaptive-model chart (outdoor prevailing mean vs operative) ---------- */
  var AC = { x0: 72, x1: 520, y0: 20, y1: 440, xmin: 10, xmax: 33.5 };
  function drawAdaptiveChart(svg, o) {
    // o = {bands: [{low(x), up(x), fill, opacity}] (outermost first),
    //      xLabel, yLabel, point: {x, y}, legend: [ [main, sub, val, unit] ]}
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute('viewBox', '0 0 580 500');
    var ymin = 1e9, ymax = -1e9;
    var x;
    for (x = AC.xmin; x <= AC.xmax + 0.01; x += 0.5) {
      o.bands.forEach(function (b) {
        ymin = Math.min(ymin, b.low(x));
        ymax = Math.max(ymax, b.up(x));
      });
    }
    if (o.point) { ymin = Math.min(ymin, o.point.y); ymax = Math.max(ymax, o.point.y); }
    ymin = Math.floor(ymin - 2);
    ymax = Math.ceil(ymax + 2);
    function ax(xv) { return AC.x0 + (xv - AC.xmin) / (AC.xmax - AC.xmin) * (AC.x1 - AC.x0); }
    function ay(yv) { return AC.y1 - (yv - ymin) / (ymax - ymin) * (AC.y1 - AC.y0); }
    var clip = el('clipPath', { id: 'aclip' + Math.floor(Math.random() * 1e6) }, svg);
    el('rect', { x: AC.x0, y: AC.y0, width: AC.x1 - AC.x0, height: AC.y1 - AC.y0 }, clip);
    var g = el('g', { 'clip-path': 'url(#' + clip.getAttribute('id') + ')' }, svg);
    // frame + gridlines
    el('rect', { x: AC.x0, y: AC.y0, width: AC.x1 - AC.x0, height: AC.y1 - AC.y0, fill: '#ffffff', stroke: '#e3e5e8' }, g);
    for (x = 15; x <= 30; x += 5) {
      el('line', { x1: ax(x), y1: AC.y0, x2: ax(x), y2: AC.y1, stroke: '#eeeeee', 'stroke-width': 1 }, g);
    }
    for (var gy = Math.ceil(ymin / 5) * 5; gy <= ymax; gy += 5) {
      el('line', { x1: AC.x0, y1: ay(gy), x2: AC.x1, y2: ay(gy), stroke: '#eeeeee', 'stroke-width': 1 }, g);
    }
    // bands
    o.bands.forEach(function (b) {
      var d = '';
      for (x = AC.xmin; x <= AC.xmax + 0.01; x += 0.5) {
        d += (d ? 'L' : 'M') + ax(x).toFixed(1) + ',' + ay(b.up(x)).toFixed(1);
      }
      for (x = AC.xmax; x >= AC.xmin - 0.01; x -= 0.5) {
        d += 'L' + ax(x).toFixed(1) + ',' + ay(b.low(x)).toFixed(1);
      }
      d += ' Z';
      el('path', { d: d, fill: b.fill, 'fill-opacity': b.opacity == null ? 0.9 : b.opacity }, g);
    });
    // axes
    for (x = 10; x <= 35; x += 5) {
      txt(ax(x), AC.y1 + 18, String(x), { 'font-size': '12', 'text-anchor': 'middle', fill: '#212529', 'font-family': 'sans-serif' }, svg);
    }
    for (var y = Math.ceil(ymin / 5) * 5; y <= ymax; y += 5) {
      txt(AC.x1 + 8, ay(y) + 4, String(y), { 'font-size': '12', fill: '#212529', 'font-family': 'sans-serif' }, svg);
    }
    txt((AC.x0 + AC.x1) / 2, AC.y1 + 42, o.xLabel, { 'font-size': '14', 'text-anchor': 'middle', fill: '#212529', 'font-family': 'sans-serif' }, svg);
    var ry = el('text', { 'font-size': '14', 'text-anchor': 'middle', fill: '#212529', 'font-family': 'sans-serif', transform: 'translate(' + (AC.x1 + 34) + ' ' + (AC.y0 + AC.y1) / 2 + ') rotate(-90)' }, svg);
    ry.textContent = o.yLabel;
    // legend (top-left)
    (o.legend || []).forEach(function (it, i) {
      var ty = 40 + i * 20;
      var te = txt(AC.x0 + 10, ty, '', { 'font-size': '12', 'font-family': 'monospace', fill: '#212529' }, svg);
      var s1 = el('tspan', {}, te); s1.textContent = it[0];
      if (it[1]) { var s2 = el('tspan', { 'baseline-shift': 'sub', 'font-size': '9' }, te); s2.textContent = it[1]; }
      var s3 = el('tspan', {}, te); s3.textContent = '  ' + it[2];
      var s4 = el('tspan', { 'font-size': '10' }, te); s4.textContent = it[3];
    });
    // point
    if (o.point) {
      el('circle', { cx: ax(o.point.x), cy: ay(o.point.y), r: 8, fill: 'none', stroke: '#d62728', 'stroke-width': 2.4 }, svg);
      el('circle', { cx: ax(o.point.x), cy: ay(o.point.y), r: 2.6, fill: '#d62728' }, svg);
    }
  }

  function psychLegend(t, rhPct) {
    var w = humRatio(t, rhPct, atmP);
    return {
      tdb: t, rh: rhPct, w: w,
      twb: wetBulb(t, rhPct),
      tdp: dewPoint(t, rhPct),
      h: enthalpy(t, w)
    };
  }

  /* ---------- adaptive band shapes ---------- */
  function round1(x) { return Math.round(x * 10) / 10; }
  function adaptiveBandsAshrae(v, point) {
    function tc(x) { return round1(0.31 * x + 17.8); }
    function up(d) {
      return function (x) {
        var b = tc(x) + d;
        return b + adaptiveCoolingEffect(v, b);
      };
    }
    function lo(d) { return function (x) { return tc(x) - d; }; }
    return {
      bands: [
        { low: lo(3.5), up: up(3.5), fill: '#a7d7a7', opacity: 0.8 },
        { low: lo(2.5), up: up(2.5), fill: '#3d9c3d', opacity: 0.85 }
      ],
      xLabel: 'Prevailing mean outdoor temperature [\u00B0C]',
      yLabel: 'Operative temperature [\u00B0C]',
      point: point
    };
  }
  function adaptiveBandsEn(v, point) {
    function tc(x) { return 0.33 * x + 18.8; }
    function up(d) {
      return function (x) {
        var b = tc(x) + d;
        return b + adaptiveCoolingEffect(v, b);
      };
    }
    function lo(d) { return function (x) { return tc(x) - d; }; }
    return {
      bands: [
        { low: lo(5), up: up(4), fill: '#a7d7a7', opacity: 0.8 },
        { low: lo(4), up: up(3), fill: '#6fbf6f', opacity: 0.8 },
        { low: lo(3), up: up(2), fill: '#3d9c3d', opacity: 0.9 }
      ],
      xLabel: 'Running mean outdoor temperature [\u00B0C]',
      yLabel: 'Operative temperature [\u00B0C]',
      point: point
    };
  }

  /* ================= HOME (ASHRAE-55) ================= */
  (function initIndex() {
    var root = pageRoot('index');
    var els = {
      method: $('#f-method'), trmField: $('#f-trm-field'),
      metField: $('#f-met-field'), cloField: $('#f-clo-field'),
      comp: $('#compliance'), pmv: $('#r-pmv'), ppd: $('#r-ppd'),
      sens: $('#r-sens'), set: $('#r-set'), svg: $('#chart'),
      ctRow: $('#chart-type-row'), ct: $('#chart-type'), reset: $('#btnReset'),
      humSel: $('.hum-sel', root), velCtrl: $('.vel-ctrl', root)
    };
    wireHumSel(els.humSel, function () { return toSI(num(null, 'f-to'), 't'); });
    function rhOf(t) {
      return humidityToRh(els.humSel.selectedIndex, num(null, 'f-rh'), t);
    }
    function updateHome() {
      var adaptive = els.method.selectedIndex === 1;
      els.trmField.style.display = adaptive ? '' : 'none';
      els.metField.style.display = adaptive ? 'none' : '';
      els.cloField.style.display = adaptive ? 'none' : '';
      els.ctRow.style.display = adaptive ? 'none' : '';
      var to = toSI(num(null, 'f-to'), 't');
      var vel = toSI(num(null, 'f-vel'), 'v');
      var rh = rhOf(to);
      var met = num(null, 'f-met');
      var clo = num(null, 'f-clo');
      var local = els.velCtrl.selectedIndex === 1;
      if (!adaptive) {
        var r = pmvAshrae(to, to, vel, rh, met, clo, { pAtm: atmP, airspeedControl: local });
        var pmv = r.pmv;
        var ok = r.valid && Math.abs(pmv) <= 0.5;
        els.comp.className = 'compliance ' + (ok ? 'ok' : 'no');
        els.comp.innerHTML = '<span class="tick">' + (ok ? '\u2714' : '\u2718') + '</span> ' +
          (ok ? 'Complies with' : 'Does not comply with') + ' ASHRAE Standard 55-2023' +
          (!r.valid ? ' (inputs outside the applicability limits)' : '');
        els.pmv.textContent = 'PMV = ' + fmt(pmv, 2);
        els.ppd.textContent = 'PPD = ' + fmt(r.ppd, 0) + ' %';
        els.sens.textContent = 'Sensation = ' + sensation(pmv);
        els.set.textContent = 'SET = ' + fmt(setTmp(to, to, r.vr, rh, met, r.cloDyn, atmP), 1) + ' \u00B0C' +
          (r.ce > 0 ? ' (CE ' + fmt(r.ce, 1) + ' \u00B0C)' : '');
        var xl = els.ct.selectedIndex === 1 ? 'Dry-bulb Temperature [\u00B0C]' : 'Operative Temperature [\u00B0C]';
        drawChart(els.svg, {
          xLabel: xl,
          zones: [{
            tol: 0.5, fill: '#4da74d', opacity: 0.85,
            f: function (t, rhx) { return pmvAshrae(t, t, vel, rhx, met, clo, { pAtm: atmP }).pmv; }
          }],
          points: [{ t: to, w: humRatio(to, rh, atmP), color: '#d62728' }],
          legend: psychLegend(to, rh)
        });
      } else {
        var tRm = toSI(num(null, 'f-trm'), 't');
        var a = adaptiveAshrae(to, to, tRm, vel);
        if (!a.valid) {
          els.comp.className = 'compliance no';
          els.comp.innerHTML = '<span class="tick">\u2718</span> Outside the adaptive model range (prevailing mean 10\u201333.5 \u00B0C)';
        } else {
          var okA = a.acceptability80;
          els.comp.className = 'compliance ' + (okA ? 'ok' : 'no');
          els.comp.innerHTML = '<span class="tick">' + (okA ? '\u2714' : '\u2718') + '</span> ' +
            (okA ? 'Complies with' : 'Does not comply with') + ' ASHRAE Standard 55-2023 (adaptive model)';
        }
        els.pmv.textContent = 'Comfort temp = ' + fmt(a.tCmf, 1) + ' \u00B0C';
        els.ppd.textContent = '80% band = ' + fmt(a.tCmf80Low, 1) + '\u2013' + fmt(a.tCmf80Up, 1) + ' \u00B0C';
        els.sens.textContent = '90% band = ' + fmt(a.tCmf90Low, 1) + '\u2013' + fmt(a.tCmf90Up, 1) + ' \u00B0C';
        els.set.textContent = 'Cooling effect = ' + fmt(a.ce, 1) + ' \u00B0C';
        drawAdaptiveChart(els.svg, adaptiveBandsAshrae(vel, { x: tRm, y: to }));
      }
    }
    onInputs(root, updateHome);
    if (els.reset) els.reset.addEventListener('click', function () {
      $('#f-to').value = 25; $('#f-vel').value = 0.1; $('#f-rh').value = 50;
      $('#f-met').value = 1; $('#f-clo').value = 0.61; $('#f-trm').value = 20;
      els.method.selectedIndex = 0;
      els.velCtrl.selectedIndex = 1;
      els.humSel.selectedIndex = 0; els.humSel.__kind = 0;
      var line = els.humSel.closest('.field-line');
      if (line) { $('.hum-unit', line).textContent = '%'; $('#f-rh').removeAttribute('data-kind'); }
      updateHome();
    });
    document.addEventListener('recalc', updateHome);
    updateHome();
  })();

  /* ================= EN-16798 ================= */
  (function initEn() {
    var root = pageRoot('en');
    var en = {
      method: $('#en-f-method'), trmField: $('#en-trm-field'),
      rhField: $('#en-rh-field'), metField: $('#en-met-field'), cloField: $('#en-clo-field'),
      comp: $('#en-compliance'), pmv: $('#en-r-pmv'), ppd: $('#en-r-ppd'),
      cat: $('#en-r-cat'), set: $('#en-r-set'), svg: $('#en-chart'),
      ct: $('#en-chart-type'), reset: $('#en-btnReset'),
      useOp: $('#en-f-useop'), trField: $('#en-tr-field'),
      humSel: $('.hum-sel', root)
    };
    wireHumSel(en.humSel, function () { return toSI(num(null, 'en-f-ta'), 't'); });
    function updateEN() {
      var adaptive = en.method.selectedIndex === 1;
      en.trmField.style.display = adaptive ? '' : 'none';
      en.rhField.style.display = adaptive ? 'none' : '';
      en.metField.style.display = adaptive ? 'none' : '';
      en.cloField.style.display = adaptive ? 'none' : '';
      var ta = toSI(num(null, 'en-f-ta'), 't');
      var tr = en.useOp.checked ? ta : toSI(num(null, 'en-f-tr'), 't');
      var vel = toSI(num(null, 'en-f-vel'), 'v');
      var rh = humidityToRh(en.humSel.selectedIndex, num(null, 'en-f-rh'), ta);
      var met = num(null, 'en-f-met');
      var clo = num(null, 'en-f-clo');
      en.trField.style.display = en.useOp.checked ? 'none' : '';
      if (!adaptive) {
        var pmv = pmvCalc(ta, tr, vel, rh, met, clo);
        var ppd = ppdCalc(pmv);
        var ap = Math.abs(pmv);
        var ok = ap <= 0.7;
        var cat = ap <= 0.2 ? 'I' : ap <= 0.5 ? 'II' : ap <= 0.7 ? 'III' : 'IV';
        en.comp.className = 'compliance ' + (ok ? 'ok' : 'no');
        en.comp.innerHTML = '<span class="tick">' + (ok ? '\u2714' : '\u2718') + '</span> ' +
          (ok ? 'Complies with' : 'Does not comply with') + ' EN-16798';
        en.pmv.textContent = 'PMV = ' + fmt(pmv, 2);
        en.ppd.textContent = 'PPD = ' + fmt(ppd, 0) + ' %';
        en.cat.textContent = 'Category = ' + cat;
        en.set.textContent = 'SET = ' + fmt(setTmp(ta, tr, vel, rh, met, clo, atmP), 1) + ' \u00B0C';
        var xl = en.ct.selectedIndex === 1 ? 'Operative Temperature [\u00B0C]' : 'Dry-bulb Temperature [\u00B0C]';
        drawChart(en.svg, {
          xLabel: xl,
          zones: [
            { tol: 0.7, fill: '#a7d7a7', opacity: 0.8, f: function (t, rhx) { return pmvCalc(t, tr, vel, rhx, met, clo); } },
            { tol: 0.5, fill: '#6fbf6f', opacity: 0.8, f: function (t, rhx) { return pmvCalc(t, tr, vel, rhx, met, clo); } },
            { tol: 0.2, fill: '#3d9c3d', opacity: 0.9, f: function (t, rhx) { return pmvCalc(t, tr, vel, rhx, met, clo); } }
          ],
          points: [{ t: ta, w: humRatio(ta, rh, atmP), color: '#d62728' }],
          legend: psychLegend(ta, rh)
        });
      } else {
        var tRm = toSI(num(null, 'en-f-trm'), 't');
        var a = adaptiveEn(ta, tr, tRm, vel);
        if (!a.valid) {
          en.comp.className = 'compliance no';
          en.comp.innerHTML = '<span class="tick">\u2718</span> Outside the adaptive model range (running mean 10\u201333.5 \u00B0C)';
        } else {
          var okE = a.acceptabilityIII;
          en.comp.className = 'compliance ' + (okE ? 'ok' : 'no');
          en.comp.innerHTML = '<span class="tick">' + (okE ? '\u2714' : '\u2718') + '</span> ' +
            (okE ? 'Complies with' : 'Does not comply with') + ' EN-16798 (adaptive model)';
        }
        var catTxt = a.acceptabilityI ? 'I' : a.acceptabilityII ? 'II' : a.acceptabilityIII ? 'III' : 'outside';
        en.pmv.textContent = 'Comfort temp = ' + fmt(a.tCmf, 1) + ' \u00B0C';
        en.ppd.textContent = 'Cat I band = ' + fmt(a.catI.low, 1) + '\u2013' + fmt(a.catI.up, 1) + ' \u00B0C';
        en.cat.textContent = 'Category = ' + catTxt;
        en.set.textContent = 'Cooling effect = ' + fmt(a.ce, 1) + ' \u00B0C';
        var toA = a.to;
        drawAdaptiveChart(en.svg, adaptiveBandsEn(vel, { x: tRm, y: toA }));
      }
    }
    onInputs(root, updateEN);
    if (en.reset) en.reset.addEventListener('click', function () {
      $('#en-f-ta').value = 25; $('#en-f-tr').value = 25; $('#en-f-vel').value = 0.1;
      $('#en-f-rh').value = 50; $('#en-f-met').value = 1; $('#en-f-clo').value = 0.61;
      $('#en-f-trm').value = 20;
      en.useOp.checked = false;
      en.method.selectedIndex = 0;
      en.humSel.selectedIndex = 0; en.humSel.__kind = 0;
      var line = en.humSel.closest('.field-line');
      if (line) { $('.hum-unit', line).textContent = '%'; $('#en-f-rh').removeAttribute('data-kind'); }
      updateEN();
    });
    document.addEventListener('recalc', updateEN);
    updateEN();
  })();

  /* ================= COMPARE ================= */
  (function initCompare() {
    var root = pageRoot('compare');
    var conds = [
      { ta: 26, tr: 25, vel: 0.1, hum: 50, humKind: 0, met: 1, clo: 0.51 },
      { ta: 22, tr: 22, vel: 0.1, hum: 50, humKind: 0, met: 1, clo: 1.0 }
    ];
    var cur = 0;
    var ids = ['c-ta', 'c-tr', 'c-vel', 'c-rh', 'c-met', 'c-clo'];
    var keys = ['ta', 'tr', 'vel', 'hum', 'met', 'clo'];
    var svg = $('#c-chart');
    var cells = $$('.results-table td[data-v]', root);
    var humSel = $('.hum-sel', root);
    var velCtrl = $('.vel-ctrl', root);
    wireHumSel(humSel, function () { return conds[cur].ta; });
    function saveCur() {
      var kinds = { ta: 't', tr: 't', vel: 'v' };
      keys.forEach(function (k, i) {
        var raw = parseFloat($('#' + ids[i]).value) || 0;
        if (k === 'hum') { conds[cur].hum = raw; conds[cur].humKind = humSel.selectedIndex; }
        else conds[cur][k] = toSI(raw, kinds[k]);
      });
    }
    function loadCur() {
      keys.forEach(function (k, i) { $('#' + ids[i]).value = conds[cur][k]; });
      humSel.selectedIndex = conds[cur].humKind;
      humSel.__kind = conds[cur].humKind;
      var line = humSel.closest('.field-line');
      var u = line && $('.hum-unit', line);
      if (u) u.textContent = humUnitLabel(conds[cur].humKind);
      $$('.tabs .tab', root).forEach(function (t, i) {
        t.classList.toggle('active', i === cur);
      });
    }
    function resultsOf(c) {
      var tr = ($('#c-useop') && $('#c-useop').checked) ? c.ta : c.tr;
      var rh = humidityToRh(c.humKind, c.hum, c.ta);
      var r = pmvAshrae(c.ta, tr, c.vel, rh, c.met, c.clo,
        { pAtm: atmP, airspeedControl: velCtrl.selectedIndex === 1 });
      return { pmv: r.pmv, ppd: r.ppd, ok: r.valid && Math.abs(r.pmv) <= 0.5 };
    }
    function updateCompare() {
      saveCur();
      var trf = $('#c-tr-field'), uo = $('#c-useop');
      if (trf && uo) trf.style.display = uo.checked ? 'none' : '';
      var rs = conds.map(resultsOf);
      cells.forEach(function (td) {
        var v = td.getAttribute('data-v');
        var i = td.getAttribute('data-i');
        var r = rs[i - 1];
        if (v === 'comp') { td.textContent = r.ok ? '\u2714' : '\u2718'; td.style.color = r.ok ? '#6f42c1' : '#b30000'; }
        if (v === 'pmv') td.textContent = fmt(r.pmv, 2);
        if (v === 'ppd') td.textContent = fmt(r.ppd, 0) + ' %';
        if (v === 'sens') td.textContent = sensation(r.pmv);
      });
      var c0 = conds[0];
      var rh0 = humidityToRh(c0.humKind, c0.hum, c0.ta);
      drawChart(svg, {
        xLabel: 'Operative Temperature [\u00B0C]',
        zones: [{
          tol: 0.5, fill: '#4da74d', opacity: 0.85,
          f: function (t, rhx) {
            return pmvAshrae(t, t, c0.vel, rhx, c0.met, c0.clo, { pAtm: atmP }).pmv;
          }
        }],
        points: [
          { t: conds[0].ta, w: humRatio(conds[0].ta, rh0, atmP), color: '#cc0000' },
          { t: conds[1].ta, w: humRatio(conds[1].ta, humidityToRh(conds[1].humKind, conds[1].hum, conds[1].ta), atmP), color: '#2a6dbf' }
        ],
        legend: psychLegend(conds[0].ta, rh0)
      });
    }
    $$('.tabs .tab', root).forEach(function (t, i) {
      t.addEventListener('click', function () { saveCur(); cur = i; loadCur(); });
    });
    onInputs($('#compare-inputs'), updateCompare);
    var rst = $('#c-btnReset');
    if (rst) rst.addEventListener('click', function () {
      conds[cur] = { ta: 26, tr: 25, vel: 0.1, hum: 50, humKind: 0, met: 1, clo: 0.51 };
      loadCur(); updateCompare();
    });
    document.addEventListener('recalc', updateCompare);
    loadCur();
    updateCompare();
  })();

  /* ================= RANGES ================= */
  (function initRanges() {
    var root = pageRoot('ranges');
    var rg = {
      svg: $('#rg-chart'), outer: $('#r-outer'), inner: $('#r-inner'),
      varSel: $('#var-param'), fixedWrap: $('#fixed-inputs'), reset: $('#rg-btnReset')
    };
    var VAR_INFO = {
      vel: { label: 'Air speed', unit: 'm/s', min: 0.1, max: 0.4, step: 0.05, fstep: 0.05 },
      rh: { label: 'Relative humidity', unit: '%', min: 30, max: 70, step: 5, fstep: 1 },
      met: { label: 'Metabolic rate', unit: 'met', min: 1.0, max: 1.3, step: 0.1, fstep: 0.1 },
      clo: { label: 'Clothing level', unit: 'clo', min: 0.51, max: 0.71, step: 0.05, fstep: 0.01 }
    };
    function fieldHtml(id, label, unit, val, step, kind, unitClass) {
      return '<div class="field"><label>' + label + '</label>' +
        '<div class="field-line"><span class="num-wrap">' +
        '<input id="' + id + '" type="text" inputmode="decimal" value="' + val + '" data-step="' + step + '"' + (kind ? ' data-kind="' + kind + '"' : '') + '>' +
        '<span class="spin"><button type="button" class="spin-up">\u25B2</button><button type="button" class="spin-down">\u25BC</button></span>' +
        '</span><span class="unit ' + (unitClass || '') + (kind ? ' u-' + kind : '') + '">' + unit + '</span></div></div>';
    }
    function getVar() { return rg.varSel.value; }
    function rhKindSel() { return $('select.hum-sel', rg.fixedWrap); }
    function currentParams() {
      var p = { vel: 0.1, rh: 50, met: 1, clo: 0.61 };
      ['vel', 'rh', 'met', 'clo'].forEach(function (k) {
        var e = $('#fx-' + k);
        if (!e) return;
        if (k === 'vel') p.vel = toSI(parseFloat(e.value) || 0, 'v');
        else if (k === 'rh') {
          var ks = rhKindSel();
          p.rh = humidityToRh(ks ? ks.selectedIndex : 0, parseFloat(e.value) || 0, 25);
        } else p[k] = parseFloat(e.value) || 0;
      });
      return p;
    }
    function buildFixed() {
      var v = getVar();
      var defs = { vel: 0.1, rh: 50, met: 1, clo: 0.61 };
      var keep = {};
      ['vel', 'rh', 'met', 'clo'].forEach(function (k) {
        var e = $('#fx-' + k);
        keep[k] = e ? e.value : defs[k];
      });
      var html = '';
      ['vel', 'rh', 'met', 'clo'].forEach(function (k) {
        if (k === v) return;
        var info = VAR_INFO[k];
        html += fieldHtml('fx-' + k, info.label, info.unit, keep[k], info.fstep, k === 'vel' ? 'v' : '', k === 'rh' ? 'hum-unit' : '');
        if (k === 'rh') {
          html = html.slice(0, -12) +
            '<select class="sel hum-sel" style="flex:1 1 140px;max-width:200px;margin-left:auto">' +
            '<option>Relative humidity</option><option>Humidity ratio</option><option>Dew point</option><option>Wet bulb</option><option>Vapor pressure</option>' +
            '</select></div></div>';
        }
        if (k === 'met') {
          html = html.slice(0, -12) +
            '<select class="sel" data-set="fx-met" style="flex:1 1 140px;max-width:200px;margin-left:auto">' +
            '<option value="0.7">Sleeping: 0.7</option><option value="0.8">Reclining: 0.8</option>' +
            '<option value="1" selected>Seated, quiet: 1.0</option><option value="1.2">Standing, relaxed: 1.2</option>' +
            '<option value="1.7">Walking 2 mph: 1.7</option><option value="3.2">Heavy work: 3.2</option>' +
            '</select></div></div>';
        }
        if (k === 'clo') {
          html = html.slice(0, -12) +
            '<select class="sel" data-set="fx-clo" style="flex:1 1 140px;max-width:200px;margin-left:auto">' +
            '<option value="0.36">Walking shorts, short-sleeve shirt: 0.36 clo</option>' +
            '<option value="0.5">Typical summer indoor clothing: 0.5 clo</option>' +
            '<option value="0.61" selected>Trousers, long-sleeve shirt: 0.61 clo</option>' +
            '<option value="0.74">Sweat pants, long-sleeve sweatshirt: 0.74 clo</option>' +
            '<option value="1.14">Winter indoor clothing: 1.14 clo</option>' +
            '</select></div></div>';
        }
      });
      rg.fixedWrap.innerHTML = html;
      wireSpins(rg.fixedWrap);
      wireHumSel(rhKindSel(), function () { return 25; });
      onInputs(rg.fixedWrap, updateRanges);
    }
    function zonePmv(t, rh, vel, met, clo) {
      return pmvAshrae(t, t, vel, rh, met, clo, { pAtm: atmP }).pmv;
    }
    function comfortBand(p, tol) {
      var lo = solveT(-tol, function (t) { return zonePmv(t, p.rh, p.vel, p.met, p.clo); });
      var hi = solveT(tol, function (t) { return zonePmv(t, p.rh, p.vel, p.met, p.clo); });
      return lo !== null && hi !== null ? [lo, hi] : null;
    }
    function updateRanges() {
      var v = getVar();
      var info = VAR_INFO[v];
      var vlo = num(null, 'r-lo'), vhi = num(null, 'r-hi'), vst = num(null, 'r-step') || info.step;
      var base = currentParams();
      var bands = [], pts = [];
      var outerLo = null, outerHi = null, innerLo = null, innerHi = null;
      for (var x = vlo; x <= vhi + 1e-9; x += vst) {
        var p = { vel: base.vel, rh: base.rh, met: base.met, clo: base.clo };
        p[v] = x;
        var b = comfortBand(p, 0.5);
        var c = comfortBand(p, 0);
        if (c) pts.push([(c[0] + c[1]) / 2, humRatio((c[0] + c[1]) / 2, p.rh, atmP)]);
        if (!b) continue;
        if (outerLo === null || b[0] < outerLo) outerLo = b[0];
        if (outerHi === null || b[1] > outerHi) outerHi = b[1];
        if (innerLo === null || b[0] > innerLo) innerLo = b[0];
        if (innerHi === null || b[1] < innerHi) innerHi = b[1];
      }
      var outerRange = (outerLo !== null) ? (outerHi - outerLo) : 0;
      var innerRange = (innerHi > innerLo) ? (innerHi - innerLo) : 0;
      rg.outer.textContent = 'Outer range of air temperatures: ' + fmt(outerRange, 1) + ' \u00B0C' +
        (outerLo !== null ? ' (' + fmt(outerLo, 1) + ' to ' + fmt(outerHi, 1) + ' \u00B0C)' : '');
      rg.inner.textContent = 'Inner range of air temperatures: ' + fmt(innerRange, 1) + ' \u00B0C';
      var pMin = { vel: base.vel, rh: base.rh, met: base.met, clo: base.clo };
      var pMax = { vel: base.vel, rh: base.rh, met: base.met, clo: base.clo };
      pMin[v] = vlo; pMax[v] = vhi;
      drawChart(rg.svg, {
        xLabel: 'Operative Temperature [\u00B0C]',
        zones: [
          { tol: 0.5, fill: '#8fce8f', opacity: 0.6, f: function (t, rhx) { return zonePmv(t, rhx, pMin.vel, pMin.met, pMin.clo); } },
          { tol: 0.5, fill: '#4da74d', opacity: 0.5, f: function (t, rhx) { return zonePmv(t, rhx, pMax.vel, pMax.met, pMax.clo); } }
        ],
        line: pts,
        legend: psychLegend(25, base.rh)
      });
    }
    function syncVarUI() {
      var info = VAR_INFO[getVar()];
      $('#r-lo').value = info.min; $('#r-hi').value = info.max;
      var st = $('#r-step');
      st.innerHTML = '';
      [info.step, info.step * 2, info.step * 4].forEach(function (s) {
        var o = document.createElement('option');
        o.value = s; o.textContent = s + ' ' + info.unit;
        st.appendChild(o);
      });
      buildFixed();
    }
    rg.varSel.addEventListener('change', function () { syncVarUI(); updateRanges(); });
    onInputs($('#range-inputs'), updateRanges);
    if (rg.reset) rg.reset.addEventListener('click', function () {
      syncVarUI(); updateRanges();
    });
    document.addEventListener('recalc', updateRanges);
    syncVarUI();
    updateRanges();
  })();

  /* ---------- nav tab bar (merged single page) ---------- */
  var TABS = { index: 'ASHRAE-55', en: 'EN-16798', compare: 'Compare', ranges: 'Ranges', upload: 'Upload' };
  function activateTab(name) {
    document.body.setAttribute('data-page', name);
    $$('section[data-page]').forEach(function (s) {
      s.hidden = s.getAttribute('data-page') !== name;
    });
    $$('.main-nav a[data-tab]').forEach(function (a) {
      a.classList.toggle('active', a.getAttribute('data-tab') === name);
    });
    var nc = $('#nav-current');
    if (nc) {
      nc.textContent = TABS[name] || name;
      nc.style.display = name === 'index' ? 'none' : '';
    }
    var nav = $('#mainNav');
    if (nav) nav.classList.remove('open');
    var tgl = $('#navToggle');
    if (tgl) tgl.setAttribute('aria-expanded', 'false');
  }
  $$('.main-nav a[data-tab]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      activateTab(a.getAttribute('data-tab'));
    });
  });
  var toggle = $('#navToggle');
  var nav = $('#mainNav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }
  wireSpins(document);
  activateTab('index');
})();
