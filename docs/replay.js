/* Project MV - lap replay viewer (shared by the control panel and the published replay page).
 *
 * Usage:   const v = MVReplay.create(containerElement, {csvHref: lap => url, onReproduce: lap => {...}});
 *          v.load(lap, ghostLap /* optional *\/);
 *
 * A lap is the JSON written by mv/laps.py: {meta, track: {left, right, centre, start}, samples: {t, s, x, y, yaw,
 * speed, steer, thr, brk, gear}}.  Everything is drawn on <canvas>; no external libraries.
 */
(function () {
  "use strict";
  const CSS = `
.mvr { --mvr-ink: var(--ink, #e6e9ef); --mvr-muted: var(--muted, #98a1b3); --mvr-line: var(--line, #2b313b);
       --mvr-bar: var(--bar, #2a3039); --mvr-accent: var(--accent, #d5001c); --mvr-ok: var(--ok, #3fbf6f); --mvr-bad: var(--bad, #e05a55);
       --mvr-card: var(--card, #1c2027); color: var(--mvr-ink); }
.mvr h2 { font-size: 15px; margin: 0 0 12px; }
.mvr h3 { font-size: 12px; margin: 16px 0 6px; color: var(--mvr-muted); text-transform: uppercase; letter-spacing: .5px; }
.mvr .grid { display: grid; grid-template-columns: 1fr 320px; gap: 14px; }
@media (max-width: 980px) { .mvr .grid { grid-template-columns: 1fr; } }
.mvr .canvasbox { background: #0e1116; border-radius: 10px; overflow: hidden; position: relative; }
.mvr .canvasbox canvas { width: 100%; height: 100%; display: block; }
.mvr .map { aspect-ratio: 4 / 3; } .mvr .chase { aspect-ratio: 1 / 1; }
.mvr .badge { position: absolute; left: 10px; top: 8px; font-size: 11px; color: #cfd5de; background: rgba(0,0,0,.45); padding: 2px 7px; border-radius: 5px; }
.mvr .hud { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 8px; margin-top: 10px; }
.mvr .big { font: 700 28px/1 ui-monospace, "Cascadia Mono", Consolas, monospace; }
.mvr .lbl { font-size: 11px; color: var(--mvr-muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
.mvr .pedals { display: flex; gap: 8px; height: 60px; }
.mvr .pedal { width: 18px; height: 100%; background: var(--mvr-bar); border-radius: 4px; position: relative; overflow: hidden; }
.mvr .pedal > div { position: absolute; bottom: 0; left: 0; right: 0; }
.mvr .wheel { width: 52px; height: 52px; border: 5px solid var(--mvr-muted); border-radius: 50%; position: relative; }
.mvr .wheel::before { content: ""; position: absolute; left: 50%; top: -5px; width: 4px; height: 12px; margin-left: -2px; background: var(--mvr-accent); }
.mvr .wheel::after { content: ""; position: absolute; left: -5px; right: -5px; top: 50%; height: 5px; margin-top: -2px; background: var(--mvr-muted); }
.mvr .ahead { color: var(--mvr-ok); } .mvr .behind { color: var(--mvr-bad); }
.mvr .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
.mvr .controls input[type=range] { flex: 1; min-width: 180px; }
.mvr button, .mvr select, .mvr .btn { font: inherit; border: 1px solid var(--mvr-line); background: var(--mvr-card); color: var(--mvr-ink);
       padding: 6px 12px; border-radius: 7px; cursor: pointer; text-decoration: none; display: inline-block; }
.mvr button.primary { background: var(--mvr-accent); border-color: var(--mvr-accent); color: #fff; }
.mvr button:disabled { opacity: .45; cursor: default; }
.mvr .seg { display: inline-flex; border: 1px solid var(--mvr-line); border-radius: 7px; overflow: hidden; }
.mvr .seg button { border: 0; border-radius: 0; }
.mvr .seg button.on { background: var(--mvr-bar); font-weight: 600; }
.mvr .key { height: 8px; border-radius: 4px; flex: 1; }
.mvr .keyrow { display: flex; gap: 8px; align-items: center; margin-top: 8px; font-size: 12px; color: var(--mvr-muted); }
.mvr table { border-collapse: collapse; width: 100%; font-size: 13px; }
.mvr td, .mvr th { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--mvr-line); }
.mvr th { color: var(--mvr-muted); font-weight: 600; font-size: 12px; }
.mvr td.n { font-family: ui-monospace, Consolas, monospace; }
.mvr .chart { width: 100%; height: 150px; display: block; }
.mvr pre { font: 12px/1.45 ui-monospace, Consolas, monospace; background: rgba(127,127,127,.08); border: 1px solid var(--mvr-line);
       border-radius: 7px; padding: 10px; overflow: auto; white-space: pre-wrap; margin: 0; }
.mvr .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--mvr-muted); margin-top: 4px; }
.mvr .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.mvr .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.mvr .hint { font-size: 12px; color: var(--mvr-muted); }
`;
  const HTML = `
<h2 data-r="title">Choose a lap</h2>
<div class="grid">
  <div>
    <div class="canvasbox map"><canvas data-r="map"></canvas><span class="badge">track map</span></div>
    <div class="controls">
      <button class="primary" data-r="play">&#9654; Play</button>
      <select data-r="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option><option value="8">8x</option></select>
      <input type="range" data-r="scrub" min="0" max="1000" value="0">
      <label class="hint" style="display:flex;align-items:center;gap:4px"><input type="checkbox" data-r="loop" checked>loop</label>
    </div>
    <div class="controls">
      <span class="hint">colour the line by</span>
      <span class="seg"><button data-r="mSpeed" class="on">speed</button><button data-r="mDelta">time vs ghost</button></span>
    </div>
    <div class="keyrow" data-r="key"></div>
  </div>
  <div>
    <div class="canvasbox chase"><canvas data-r="chase"></canvas><span class="badge">chase camera</span></div>
    <div class="hud">
      <div><div class="lbl">lap time</div><div class="big" data-r="time">0:00.000</div></div>
      <div><div class="lbl">speed km/h</div><div class="big" data-r="kmh">0</div></div>
      <div><div class="lbl">gap to ghost</div><div class="big" data-r="gap">-</div></div>
      <div><div class="lbl">gear</div><div class="big" data-r="gear">-</div></div>
      <div><div class="lbl">throttle / brake</div><div class="pedals">
        <div class="pedal"><div data-r="thr" style="background:var(--mvr-ok);height:0"></div></div>
        <div class="pedal"><div data-r="brk" style="background:var(--mvr-bad);height:0"></div></div></div></div>
      <div><div class="lbl">steering</div><div style="padding-top:4px"><div class="wheel" data-r="wheel"></div></div></div>
    </div>
  </div>
</div>
<h3>Sectors</h3><div data-r="sectors"></div>
<h3>Speed along the lap (km/h)</h3><canvas class="chart" data-r="speedChart"></canvas>
<h3>Steering along the lap (fraction of full lock, + = left) - <span data-r="steerNote" style="text-transform:none"></span></h3><canvas class="chart" data-r="steerChart"></canvas>
<h3 data-r="gapTitle">Gap to the ghost along the lap (s) - rising = losing time</h3><canvas class="chart" data-r="gapChart"></canvas>
<div class="legend" data-r="legend"></div>
<div class="actions" data-r="actions"></div>
<h3>Where these laps come from</h3><pre data-r="meta"></pre>`;

  let cssDone = false;
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const fmt = t => t == null || !isFinite(t) ? "-" : `${Math.floor(t / 60)}:${(t % 60).toFixed(3).padStart(6, "0")}`;
  const speedColor = (v, v0, v1) => `hsl(${230 - 230 * Math.max(0, Math.min(1, (v - v0) / Math.max(v1 - v0, 1)))},85%,55%)`;
  const deltaColor = r => r >= 0 ? `hsl(0,${30 + 55 * Math.min(r, 1)}%,${62 - 10 * Math.min(r, 1)}%)`
                                 : `hsl(140,${30 + 55 * Math.min(-r, 1)}%,${55 - 8 * Math.min(-r, 1)}%)`;

  function prep(lp) {
    const s = lp.samples; let m = 0;
    s.sm = s.s.map(v => (m = Math.max(m, v)));               // distance, never decreasing
    lp.noSteer = !s.steer;                                     // real laps (e.g. F1 timing data) have no steering channel
    if (!s.steer) s.steer = s.t.map(() => 0);
    if (!s.gear) s.gear = s.t.map(() => 0);
    return lp;
  }
  function at(lp, t) {                                         // state at time t (linear interpolation)
    const s = lp.samples, ts = s.t; let lo = 0, hi = ts.length - 1;
    if (t <= ts[0]) hi = lo = 0; else if (t >= ts[hi]) lo = hi;
    else while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid; else hi = mid; }
    const f = hi > lo ? (t - ts[lo]) / (ts[hi] - ts[lo]) : 0, L = k => s[k][lo] + (s[k][hi] - s[k][lo]) * f;
    let dy = s.yaw[hi] - s.yaw[lo]; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    return {x: L("x"), y: L("y"), yaw: s.yaw[lo] + dy * f, speed: L("speed"), steer: L("steer"), thr: L("thr"), brk: L("brk"),
            gear: s.gear[lo], s: s.sm[lo] + (s.sm[hi] - s.sm[lo]) * f};
  }
  function timeAt(lp, d) {                                     // time at which the lap reached distance d
    const s = lp.samples, sm = s.sm; let lo = 0, hi = sm.length - 1;
    if (d <= sm[0]) return s.t[0] * d / Math.max(sm[0], 1e-6);
    if (d >= sm[hi]) return s.t[hi];
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (sm[mid] <= d) lo = mid; else hi = mid; }
    return s.t[lo] + (s.t[hi] - s.t[lo]) * (d - sm[lo]) / Math.max(sm[hi] - sm[lo], 1e-6);
  }
  function sizeCanvas(c) {
    const r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return [w, h, d];
  }

  function create(root, opts = {}) {
    if (!cssDone) { const st = document.createElement("style"); st.textContent = CSS; document.head.appendChild(st); cssDone = true; }
    root.classList.add("mvr"); root.innerHTML = HTML;
    const el = {}; root.querySelectorAll("[data-r]").forEach(e => (el[e.dataset.r] = e));
    let A = null, B = null, T = 0, playing = false, last = 0, mode = "speed", layer = null, xf = null, base = {}, deltaA = null;

    function computeDelta() {                                  // gap to the ghost at every sample of lap A
      deltaA = null; if (!B) return;
      const s = A.samples; deltaA = s.sm.map((d, i) => s.t[i] - timeAt(B, d));
      // rate of losing (+) / gaining (-) time per metre, smoothed over ~40 m, normalised
      const n = s.sm.length, rate = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let j = i; while (j < n - 1 && s.sm[j] - s.sm[i] < 40) j++;
        let k = i; while (k > 0 && s.sm[i] - s.sm[k] < 0.01) k--;
        rate[i] = (deltaA[j] - deltaA[i]) / Math.max(s.sm[j] - s.sm[i], 1);
      }
      const mags = rate.map(Math.abs).sort((a, b) => a - b), ref = mags[Math.floor(mags.length * 0.95)] || 1e-6;
      A._rate = rate.map(r => r / ref);
    }
    function drawTrack(ctx, tk, P, d) {
      const ring = pts => { ctx.moveTo(...P(pts[0])); for (let i = 1; i < pts.length; i++) ctx.lineTo(...P(pts[i])); ctx.closePath(); };
      ctx.beginPath(); ring(tk.left); ring(tk.right); ctx.fillStyle = "#3b4049"; ctx.fill("evenodd");
      ctx.strokeStyle = "#cfd5de"; ctx.lineWidth = 1.2 * d;
      ctx.beginPath(); ring(tk.left); ctx.stroke(); ctx.beginPath(); ring(tk.right); ctx.stroke();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 3 * d; ctx.beginPath(); ctx.moveTo(...P(tk.left[0])); ctx.lineTo(...P(tk.right[0])); ctx.stroke();
    }
    function drawLine(ctx, P, width, alpha) {
      const s = A.samples, v0 = Math.min(...s.speed), v1 = Math.max(...s.speed);
      ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.lineCap = "round";
      for (let i = 1; i < s.x.length; i++) {
        ctx.strokeStyle = mode === "delta" && A._rate ? deltaColor(A._rate[i]) : speedColor(s.speed[i], v0, v1);
        ctx.beginPath(); ctx.moveTo(...P([s.x[i - 1], s.y[i - 1]])); ctx.lineTo(...P([s.x[i], s.y[i]])); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    function sectorMarks(ctx, P, d) {                          // S1/S2/S3 labels at the sector boundaries
      const c = A.track.centre, L = A.meta.track_length_m;
      ctx.font = `bold ${11 * d}px Segoe UI, sans-serif`; ctx.fillStyle = "#ffd447";
      [1, 2].forEach(k => { const p = c[Math.min(Math.round(k / 3 * c.length), c.length - 1)]; const [x, y] = P(p);
        ctx.beginPath(); ctx.arc(x, y, 3.5 * d, 0, 7); ctx.fill(); ctx.fillText("S" + (k + 1), x + 6 * d, y - 6 * d); });
      void L;
    }
    function buildLayer(W, H, d) {
      const tk = A.track, pts = tk.left.concat(tk.right), xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const k = Math.min((W - 40 * d) / (x1 - x0), (H - 40 * d) / (y1 - y0)), ox = (W - (x1 - x0) * k) / 2, oy = (H - (y1 - y0) * k) / 2;
      xf = p => [ox + (p[0] - x0) * k, H - oy - (p[1] - y0) * k];
      layer = document.createElement("canvas"); layer.width = W; layer.height = H;
      const c = layer.getContext("2d"); drawTrack(c, tk, xf, d);
      if (B) { const s = B.samples; c.setLineDash([4 * d, 4 * d]); c.strokeStyle = "rgba(255,255,255,.45)"; c.lineWidth = 1.2 * d;
        c.beginPath(); c.moveTo(...xf([s.x[0], s.y[0]])); for (let i = 1; i < s.x.length; i++) c.lineTo(...xf([s.x[i], s.y[i]])); c.stroke(); c.setLineDash([]); }
      drawLine(c, xf, 2.6 * d, 1); sectorMarks(c, xf, d);
    }
    function frame() {
      if (!A) return;
      const a = at(A, T), b = B ? at(B, T) : null;
      const [W, H, d] = sizeCanvas(el.map), ctx = el.map.getContext("2d");
      if (!layer || layer.width !== W || layer.height !== H) buildLayer(W, H, d);
      ctx.clearRect(0, 0, W, H); ctx.drawImage(layer, 0, 0);
      const dot = (p, col, r) => { const [x, y] = xf([p.x, p.y]); ctx.fillStyle = col; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * d;
        ctx.beginPath(); ctx.arc(x, y, r * d, 0, 7); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.cos(p.yaw) * 16 * d, y - Math.sin(p.yaw) * 16 * d); ctx.stroke(); };
      if (b) dot(b, "rgba(160,170,185,.95)", 5);
      dot(a, "#d5001c", 6.5);
      // chase camera: 140 m across, car pointing up
      const [ZW, ZH, zd] = sizeCanvas(el.chase), z = el.chase.getContext("2d"), zk = ZW / 140;
      z.setTransform(1, 0, 0, 1, 0, 0); z.fillStyle = "#1d2b1f"; z.fillRect(0, 0, ZW, ZH);
      z.translate(ZW / 2, ZH * 0.62); z.rotate(a.yaw - Math.PI / 2);
      const ZP = p => [(p[0] - a.x) * zk, -(p[1] - a.y) * zk];
      drawTrack(z, A.track, ZP, zd); drawLine(z, ZP, 3 * zd, 0.85);
      const car = (p, col) => { z.save(); z.translate(...ZP([p.x, p.y])); z.rotate(-p.yaw);
        z.fillStyle = col; z.fillRect(-2.3 * zk, -0.97 * zk, 4.6 * zk, 1.94 * zk);
        z.fillStyle = "rgba(255,255,255,.85)"; z.fillRect(0.9 * zk, -0.7 * zk, 0.9 * zk, 1.4 * zk); z.restore(); };
      if (b) car(b, "rgba(170,180,195,.75)");
      car(a, "#d5001c"); z.setTransform(1, 0, 0, 1, 0, 0);
      // dashboard
      el.time.textContent = fmt(T); el.kmh.textContent = Math.round(a.speed * 3.6); el.gear.textContent = a.gear || "-";
      el.thr.style.height = (a.thr * 100).toFixed(0) + "%"; el.brk.style.height = (a.brk * 100).toFixed(0) + "%";
      el.wheel.style.transform = `rotate(${(-a.steer * 220).toFixed(1)}deg)`;
      if (B) { const g = T - timeAt(B, a.s); el.gap.textContent = (g > 0 ? "+" : "") + g.toFixed(2) + " s"; el.gap.className = "big " + (g > 0.005 ? "behind" : "ahead"); }
      else { el.gap.textContent = "-"; el.gap.className = "big"; }
      el.scrub.value = Math.round(T / A.meta.lap_time * 1000);
      cursor(el.speedChart, base.speed, a.s); cursor(el.steerChart, base.steer, a.s); cursor(el.gapChart, base.gap, a.s);
    }
    function chart(c, series, yfmt, zeroLine) {
      const [W, H, d] = sizeCanvas(c), L = A.meta.track_length_m;
      const ys = series.flatMap(s => s.y).filter(v => isFinite(v)); if (!ys.length) return null;
      let y0 = Math.min(...ys), y1 = Math.max(...ys); if (zeroLine) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
      const pad = (y1 - y0) * 0.08 || 1; y0 -= pad; y1 += pad;
      const X = s => 46 * d + s / L * (W - 56 * d), Y = v => 8 * d + (1 - (v - y0) / (y1 - y0)) * (H - 26 * d);
      const cv = document.createElement("canvas"); cv.width = W; cv.height = H; const g = cv.getContext("2d");
      g.font = `${11 * d}px Segoe UI, sans-serif`; g.fillStyle = "#8a93a5"; g.strokeStyle = "rgba(128,128,128,.22)";
      for (let k = 0; k <= 4; k++) { const v = y0 + (y1 - y0) * k / 4; g.beginPath(); g.moveTo(X(0), Y(v)); g.lineTo(X(L), Y(v)); g.stroke(); g.fillText(yfmt(v), 4 * d, Y(v) + 4 * d); }
      [1, 2].forEach(k => { g.strokeStyle = "rgba(255,212,71,.35)"; g.setLineDash([3 * d, 3 * d]); g.beginPath(); g.moveTo(X(k * L / 3), 0); g.lineTo(X(k * L / 3), H - 18 * d); g.stroke(); g.setLineDash([]); });
      g.fillText("distance along the lap ->", X(L) - 140 * d, H - 3 * d);
      series.forEach(sr => { g.strokeStyle = sr.color; g.lineWidth = (sr.w || 2) * d; g.beginPath();
        sr.x.forEach((x, i) => { const p = [X(x), Y(sr.y[i])]; i ? g.lineTo(...p) : g.moveTo(...p); }); g.stroke(); });
      return {cv, X};
    }
    function cursor(c, b, s) {
      if (!b) { const ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height); return; }
      const ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height); ctx.drawImage(b.cv, 0, 0);
      const x = b.X(s); ctx.strokeStyle = "#9aa3b3"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke();
    }
    function sectors() {
      const L = A.meta.track_length_m, cuts = [0, L / 3, 2 * L / 3, L];
      const st = lp => cuts.slice(1).map((c, i) => (i === 2 ? lp.meta.lap_time : timeAt(lp, c)) - (i === 0 ? 0 : timeAt(lp, cuts[i])));
      const a = st(A), b = B ? st(B) : null;
      el.sectors.innerHTML = `<table><tr><th>Sector</th><th>${esc(A.meta.driver)}</th>${B ? `<th>${esc(B.meta.driver)} (ghost)</th><th>Difference</th>` : ""}</tr>` +
        a.map((t, i) => `<tr><td>S${i + 1}</td><td class="n">${t.toFixed(3)} s</td>${b ? `<td class="n">${b[i].toFixed(3)} s</td><td class="n ${t - b[i] > 0 ? "behind" : "ahead"}">${(t - b[i] > 0 ? "+" : "") + (t - b[i]).toFixed(3)} s</td>` : ""}</tr>`).join("") +
        `<tr><td><b>Lap</b></td><td class="n"><b>${fmt(A.meta.lap_time)}</b></td>${B ? `<td class="n">${fmt(B.meta.lap_time)}</td><td class="n ${A.meta.lap_time - B.meta.lap_time > 0 ? "behind" : "ahead"}"><b>${(A.meta.lap_time - B.meta.lap_time > 0 ? "+" : "") + (A.meta.lap_time - B.meta.lap_time).toFixed(3)} s</b></td>` : ""}</tr></table>`;
    }
    function legendKey() {
      el.key.innerHTML = mode === "speed"
        ? `<span>slow</span><div class="key" style="background:linear-gradient(90deg,hsl(230,85%,55%),hsl(150,85%,45%),hsl(55,95%,50%),hsl(0,85%,55%))"></div><span>fast</span>`
        : `<span>gaining time</span><div class="key" style="background:linear-gradient(90deg,hsl(140,85%,47%),hsl(140,30%,55%),hsl(0,30%,62%),hsl(0,85%,52%))"></div><span>losing time</span>`;
    }
    function setMode(m) {
      if (m === "delta" && !B) return;
      mode = m; el.mSpeed.classList.toggle("on", m === "speed"); el.mDelta.classList.toggle("on", m === "delta");
      layer = null; legendKey(); frame();
    }
    function drawCharts() {
      const sA = A.samples, red = "#d5001c", grey = "rgba(160,170,185,.9)";
      const sp = [{x: sA.sm, y: sA.speed.map(v => v * 3.6), color: red}];
      if (B) sp.unshift({x: B.samples.sm, y: B.samples.speed.map(v => v * 3.6), color: grey, w: 1.4});
      base.speed = chart(el.speedChart, sp, v => v.toFixed(0));
      const st = A.noSteer ? [] : [{x: sA.sm, y: sA.steer, color: red, w: 1.4}];
      if (B && !B.noSteer) st.unshift({x: B.samples.sm, y: B.samples.steer, color: grey, w: 1.4});
      base.steer = st.length ? chart(el.steerChart, st, v => v.toFixed(2), true) : null;
      el.steerChart.style.display = st.length ? "" : "none";
      const act = l => { const s = l.samples.steer; let a = 0; for (let i = 1; i < s.length; i++) a += Math.abs(s[i] - s[i - 1]); return a / l.meta.lap_time; };
      const actTxt = l => l.noSteer ? "not recorded" : `${act(l).toFixed(2)} lock/s`;
      el.steerNote.textContent = `steering activity ${actTxt(A)}` + (B ? ` (ghost ${actTxt(B)})` : "") + " - lower = calmer hands";
      base.gap = B ? chart(el.gapChart, [{x: sA.sm, y: deltaA, color: "#ffd447"}], v => (v > 0 ? "+" : "") + v.toFixed(1), true) : null;
      el.gapChart.style.display = el.gapTitle.style.display = B ? "" : "none";
      el.legend.innerHTML = `<span><i style="background:${red}"></i>${esc(A.meta.driver)}</span>` + (B ? `<span><i style="background:#a0aab9"></i>${esc(B.meta.driver)} (ghost)</span>` : "");
    }
    function loop(now) {
      if (!playing || !A) return;
      T += (now - last) / 1000 * parseFloat(el.speed.value); last = now;
      if (T > A.meta.lap_time) { if (el.loop.checked) T = 0; else { T = A.meta.lap_time; setPlay(false); } }
      frame(); requestAnimationFrame(loop);
    }
    function setPlay(p) { playing = p && !!A; el.play.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play"; last = performance.now(); if (playing) requestAnimationFrame(loop); }
    el.play.onclick = () => setPlay(!playing);
    el.scrub.oninput = e => { if (A) { T = e.target.value / 1000 * A.meta.lap_time; frame(); } };
    el.mSpeed.onclick = () => setMode("speed"); el.mDelta.onclick = () => setMode("delta");
    document.addEventListener("keydown", e => { if (e.code === "Space" && A && root.offsetParent && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)) { e.preventDefault(); setPlay(!playing); } });
    window.addEventListener("resize", () => { layer = null; if (A) { drawCharts(); frame(); } });

    return {
      load(lap, ghost, keepTime) {
        A = prep(lap); B = ghost ? prep(ghost) : null;
        if (!keepTime) T = 0;
        computeDelta(); if (!B && mode === "delta") mode = "speed";
        el.mDelta.disabled = !B; layer = null;
        const m = A.meta;
        el.title.innerHTML = `${esc(m.driver)} &middot; ${esc(m.car)} &middot; ${esc(m.track)} &middot; <span style="font-family:ui-monospace,Consolas,monospace">${fmt(m.lap_time)}</span>` +
          (B ? ` <span class="hint" style="font-weight:400">vs ghost ${esc(B.meta.driver)} ${fmt(B.meta.lap_time)}</span>` : "");
        el.meta.textContent = [A, B].filter(Boolean).map((l, i) => (i ? "--- ghost ---\n" : "") + Object.entries(l.meta)
          .filter(([k]) => !["key", "id"].includes(k)).map(([k, v]) => `${k.padEnd(15)} ${v}`).join("\n")).join("\n\n");
        el.actions.innerHTML = "";
        if (opts.csvHref) { const a = document.createElement("a"); a.className = "btn"; a.href = opts.csvHref(A); a.textContent = "Download this lap's data (CSV)"; el.actions.appendChild(a); }
        if (opts.onReproduce && A.meta.checkpoint) { const b = document.createElement("button"); b.textContent = "Reproduce: drive this lap again from the saved driver";
          b.onclick = () => opts.onReproduce(A); el.actions.appendChild(b); }
        if (opts.onViewInAC && (!opts.canViewInAC || opts.canViewInAC(A))) { const b = document.createElement("button");
          b.textContent = "View in Assetto Corsa" + (B ? " (with this ghost)" : " (with the ideal-lap ghost)");
          b.title = "Plays this lap in the running game (MV Viewer app, offline Practice on this circuit with this car)";
          b.onclick = () => opts.onViewInAC(A, B); el.actions.appendChild(b); }
        sectors(); legendKey(); drawCharts(); frame();
      },
      play: setPlay,
      get lap() { return A; }, get ghost() { return B; },
    };
  }
  window.MVReplay = {create, fmt};
})();
