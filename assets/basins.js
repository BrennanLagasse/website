// Drifting-basins background: optimizers descending a non-stationary loss surface.
// Ported from prototypes/drifting-basins.html. The tuning panel (#panel) and stats HUD (#hud)
// are optional: pages that include their markup get them, others just get the background.
(() => {
  const canvas = document.getElementById('field');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const $ = id => document.getElementById(id);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // opt: 0 Momentum GD, 1 AdamW, 2 Muon, 3 compare all
  const S = { opt: 3, n: 420, drift: 1, lr: 1, beta: 0.9, wd: 0.1, temp: 0.3, playing: !reduceMotion };
  const SHORT = ['Mom', 'AdamW', 'Muon'];
  const BASE_LR = [0.0006, 0.0035, 0.0035];
  const ADAM_B2 = 0.999, ADAM_EPS = 1e-8;
  const TRAIL = 18, REST_DISP = 0.012;
  const L0 = -4, STEP = 0.075;

  let W = 0, H = 0, dpr = 1, aspect = 1;
  let cols = 0, rows = 0, cell = 9, grid = null;
  const tintCanvas = document.createElement('canvas'), tintCtx = tintCanvas.getContext('2d');
  let tintImg = null, T = 0;
  let colors = { tint: [0, 0, 0], opt: ['#000', '#000', '#000'] };

  // ---------- Landscape ----------
  const rand = (a, b) => a + Math.random() * (b - a);
  let terms = [], K = 0;
  const PX = new Float64Array(12), PY = new Float64Array(12), PA = new Float64Array(12), PI2 = new Float64Array(12), PIS = new Float64Array(12);

  function seedLandscape() {
    terms = [];
    for (let i = 0; i < 7; i++) terms.push({
      sign: -1, cx: rand(-0.75, 0.75), cy: rand(-0.7, 0.7), ax: rand(0.2, 0.5), ay: rand(0.18, 0.42),
      fx: rand(0.05, 0.14), fy: rand(0.05, 0.14), px: rand(0, 6.283), py: rand(0, 6.283),
      w0: rand(0.55, 1.0), wa: rand(0.25, 0.5), wf: rand(0.04, 0.12), wp: rand(0, 6.283), s: rand(0.15, 0.3)
    });
    for (let i = 0; i < 2; i++) terms.push({
      sign: 1, cx: rand(-0.6, 0.6), cy: rand(-0.6, 0.6), ax: rand(0.2, 0.45), ay: rand(0.2, 0.45),
      fx: rand(0.04, 0.1), fy: rand(0.04, 0.1), px: rand(0, 6.283), py: rand(0, 6.283),
      w0: rand(0.35, 0.6), wa: rand(0.1, 0.25), wf: rand(0.04, 0.1), wp: rand(0, 6.283), s: rand(0.18, 0.32)
    });
    K = terms.length;
  }
  function updateTerms() {
    for (let i = 0; i < K; i++) {
      const t = terms[i];
      PX[i] = (t.cx + t.ax * Math.sin(t.fx * T + t.px)) * aspect;
      PY[i] = t.cy + t.ay * Math.sin(t.fy * T + t.py);
      PA[i] = t.sign * Math.max(0.05, t.w0 + t.wa * Math.sin(t.wf * T + t.wp));
      PIS[i] = 1 / (t.s * t.s);
      PI2[i] = 0.5 * PIS[i];
    }
  }
  function f(x, y) {
    let v = 0.22 * (x * x / (aspect * aspect) + y * y);
    for (let i = 0; i < K; i++) {
      const dx = x - PX[i], dy = y - PY[i];
      v += PA[i] * Math.exp(-(dx * dx + dy * dy) * PI2[i]);
    }
    return v;
  }
  const G = [0, 0];
  function grad(x, y) {
    let gx = 0.44 * x / (aspect * aspect), gy = 0.44 * y;
    for (let i = 0; i < K; i++) {
      const dx = x - PX[i], dy = y - PY[i];
      const e = PA[i] * Math.exp(-(dx * dx + dy * dy) * PI2[i]) * PIS[i];
      gx -= e * dx; gy -= e * dy;
    }
    G[0] = gx; G[1] = gy;
    return G;
  }

  // ---------- Particles ----------
  // m1: momentum velocity (GD), first moment (AdamW) or momentum buffer (Muon); m2: AdamW second moment.
  let N = 0, px, py, m1x, m1y, m2x, m2y, tt, age, opt, rest, hx, hy, head = 0;
  function resetState(i) { m1x[i] = 0; m1y[i] = 0; m2x[i] = 0; m2y[i] = 0; tt[i] = 0; }
  function spawn(i, x, y) {
    px[i] = x !== undefined ? x : rand(-aspect, aspect);
    py[i] = y !== undefined ? y : rand(-1, 1);
    resetState(i);
    age[i] = 0; rest[i] = 0;
    const o = i * TRAIL;
    for (let k = 0; k < TRAIL; k++) { hx[o + k] = px[i]; hy[o + k] = py[i]; }
  }
  const optFor = i => S.opt === 3 ? i % 3 : S.opt;

  function allocParticles(n) {
    const old = N ? { px, py, m1x, m1y, m2x, m2y, tt, age, rest, hx, hy } : null;
    const oldN = N;
    N = n;
    px = new Float32Array(N); py = new Float32Array(N);
    m1x = new Float32Array(N); m1y = new Float32Array(N); m2x = new Float32Array(N); m2y = new Float32Array(N);
    tt = new Float32Array(N); age = new Uint16Array(N); opt = new Uint8Array(N); rest = new Uint8Array(N);
    hx = new Float32Array(N * TRAIL); hy = new Float32Array(N * TRAIL);
    for (let i = 0; i < N; i++) {
      opt[i] = optFor(i);
      if (old && i < oldN) {
        px[i] = old.px[i]; py[i] = old.py[i];
        m1x[i] = old.m1x[i]; m1y[i] = old.m1y[i]; m2x[i] = old.m2x[i]; m2y[i] = old.m2y[i];
        tt[i] = old.tt[i]; age[i] = old.age[i]; rest[i] = old.rest[i];
        hx.set(old.hx.subarray(i * TRAIL, (i + 1) * TRAIL), i * TRAIL);
        hy.set(old.hy.subarray(i * TRAIL, (i + 1) * TRAIL), i * TRAIL);
      } else spawn(i);
    }
  }
  function assignOptimizers() {
    for (let i = 0; i < N; i++) {
      const o = optFor(i);
      if (opt[i] !== o) { opt[i] = o; resetState(i); }
    }
  }
  const gauss = () => (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 1.732;

  const restCount = [0, 0, 0], optCount = [0, 0, 0];
  function stepParticles() {
    const { lr: mult, beta, wd } = S;
    const sg = S.temp * 4;
    restCount[0] = restCount[1] = restCount[2] = 0;
    optCount[0] = optCount[1] = optCount[2] = 0;
    head = (head + 1) % TRAIL;
    const oldest = (head + 1) % TRAIL;
    for (let i = 0; i < N; i++) {
      const g = grad(px[i], py[i]);
      const gx = g[0] + sg * gauss(), gy = g[1] + sg * gauss();
      const o = opt[i];
      let dx, dy;
      if (o === 0) {
        // Heavy-ball momentum: v <- beta v - lr g ; theta <- theta + v
        const lr = BASE_LR[0] * mult;
        m1x[i] = beta * m1x[i] - lr * gx; m1y[i] = beta * m1y[i] - lr * gy;
        dx = m1x[i]; dy = m1y[i];
      } else if (o === 1) {
        // AdamW with bias correction and decoupled weight decay
        const lr = BASE_LR[1] * mult, t = ++tt[i];
        m1x[i] = beta * m1x[i] + (1 - beta) * gx; m1y[i] = beta * m1y[i] + (1 - beta) * gy;
        m2x[i] = ADAM_B2 * m2x[i] + (1 - ADAM_B2) * gx * gx; m2y[i] = ADAM_B2 * m2y[i] + (1 - ADAM_B2) * gy * gy;
        const c1 = 1 - Math.pow(beta, t), c2 = 1 - Math.pow(ADAM_B2, t);
        dx = -lr * ((m1x[i] / c1) / (Math.sqrt(m2x[i] / c2) + ADAM_EPS) + wd * px[i]);
        dy = -lr * ((m1y[i] / c1) / (Math.sqrt(m2y[i] / c2) + ADAM_EPS) + wd * py[i]);
      } else {
        // Muon with Nesterov momentum. For a 1x2 parameter, orthogonalization reduces to unit-normalization.
        const lr = BASE_LR[2] * mult;
        m1x[i] = beta * m1x[i] + gx; m1y[i] = beta * m1y[i] + gy;
        const nx = gx + beta * m1x[i], ny = gy + beta * m1y[i];
        const inv = 1 / (Math.hypot(nx, ny) + 1e-12);
        dx = -lr * nx * inv; dy = -lr * ny * inv;
      }
      px[i] += dx; py[i] += dy;
      optCount[o]++;
      if (!(Math.abs(px[i]) <= aspect * 1.08 && Math.abs(py[i]) <= 1.08) || Math.random() < 0.0012) { spawn(i); continue; }
      const h = i * TRAIL;
      hx[h + head] = px[i]; hy[h + head] = py[i];
      if (age[i] < 65535) age[i]++;
      const ddx = px[i] - hx[h + oldest], ddy = py[i] - hy[h + oldest];
      rest[i] = (age[i] >= TRAIL && ddx * ddx + ddy * ddy < REST_DISP * REST_DISP) ? 1 : 0;
      if (rest[i]) restCount[o]++;
    }
  }
  let dropCursor = 0;
  function dropAt(x, y) {
    for (let k = 0; k < Math.min(45, N); k++) spawn((dropCursor++) % N, x + gauss() * 0.04, y + gauss() * 0.04);
  }

  // ---------- Geometry ----------
  function resize() {
    const oldAspect = aspect;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    aspect = W / H;
    cell = W < 700 ? 7 : 9;
    cols = Math.ceil(W / cell) + 1; rows = Math.ceil(H / cell) + 1;
    grid = new Float32Array(cols * rows);
    tintCanvas.width = cols; tintCanvas.height = rows;
    tintImg = tintCtx.createImageData(cols, rows);
    if (N && oldAspect !== aspect) {
      const r = aspect / oldAspect;
      for (let i = 0; i < N; i++) { px[i] *= r; for (let k = 0; k < TRAIL; k++) hx[i * TRAIL + k] *= r; }
    }
  }
  const toScreenX = x => (x / aspect + 1) * 0.5 * W;
  const toScreenY = y => (y + 1) * 0.5 * H;
  const toWorldX = sx => (sx / W * 2 - 1) * aspect;
  const toWorldY = sy => sy / H * 2 - 1;

  // ---------- Theme ----------
  function hexToRgb(h) {
    h = h.trim().replace('#', '');
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const get = k => cs.getPropertyValue(k).trim();
    colors = {
      bg: get('--bg'), contour: get('--contour'), settled: get('--settled'),
      opt: [get('--c-mom'), get('--c-adam'), get('--c-muon')], tint: hexToRgb(get('--tint'))
    };
    if (!S.playing && grid) draw();
  }

  // ---------- Rendering ----------
  function computeGrid() {
    let gi = 0;
    const [tr, tg, tb] = colors.tint, data = tintImg.data;
    for (let j = 0; j < rows; j++) {
      const wy = toWorldY(j * cell);
      for (let i = 0; i < cols; i++, gi++) {
        const v = f(toWorldX(i * cell), wy);
        grid[gi] = v;
        const t = Math.min(1, Math.max(0, (0.25 - v) / 1.6));
        const o = gi * 4;
        data[o] = tr; data[o + 1] = tg; data[o + 2] = tb; data[o + 3] = (t * t * 0.28 * 255) | 0;
      }
    }
    tintCtx.putImageData(tintImg, 0, 0);
  }
  // edges: 0 top (a-b), 1 right (b-c), 2 bottom (d-c), 3 left (a-d)
  const CASES = [[], [3, 2], [2, 1], [3, 1], [0, 1], [3, 2, 0, 1], [0, 2], [3, 0], [3, 0], [0, 2], [0, 3, 1, 2], [0, 1], [3, 1], [2, 1], [3, 2], []];
  function contours(minor, major) {
    const ex = [0, 0, 0, 0], ey = [0, 0, 0, 0];
    for (let j = 0; j < rows - 1; j++) {
      const y0 = j * cell, y1 = y0 + cell;
      for (let i = 0; i < cols - 1; i++) {
        const a = grid[j * cols + i], b = grid[j * cols + i + 1];
        const d = grid[(j + 1) * cols + i], c = grid[(j + 1) * cols + i + 1];
        const mn = Math.min(a, b, c, d), mx = Math.max(a, b, c, d);
        const k0 = Math.ceil((mn - L0) / STEP), k1 = Math.floor((mx - L0) / STEP);
        if (k1 < k0) continue;
        const x0 = i * cell, x1 = x0 + cell;
        for (let k = k0; k <= k1; k++) {
          const L = L0 + k * STEP;
          const seg = CASES[(a > L ? 8 : 0) | (b > L ? 4 : 0) | (c > L ? 2 : 0) | (d > L ? 1 : 0)];
          if (!seg.length) continue;
          ex[0] = x0 + (L - a) / (b - a) * cell; ey[0] = y0;
          ex[1] = x1; ey[1] = y0 + (L - b) / (c - b) * cell;
          ex[2] = x0 + (L - d) / (c - d) * cell; ey[2] = y1;
          ex[3] = x0; ey[3] = y0 + (L - a) / (d - a) * cell;
          const path = (k % 5 === 0) ? major : minor;
          for (let s = 0; s < seg.length; s += 2) { path.moveTo(ex[seg[s]], ey[seg[s]]); path.lineTo(ex[seg[s + 1]], ey[seg[s + 1]]); }
        }
      }
    }
  }
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, W, H);
    computeGrid();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tintCanvas, 0, 0, cols, rows, 0, 0, cols * cell, rows * cell);
    const minor = new Path2D(), major = new Path2D();
    contours(minor, major);
    ctx.strokeStyle = colors.contour; ctx.lineCap = 'round';
    ctx.globalAlpha = 0.38; ctx.lineWidth = 0.75; ctx.stroke(minor);
    ctx.globalAlpha = 0.7; ctx.lineWidth = 1.4; ctx.stroke(major);
    const trails = [new Path2D(), new Path2D(), new Path2D()], moving = [new Path2D(), new Path2D(), new Path2D()], resting = new Path2D();
    for (let i = 0; i < N; i++) {
      const o = i * TRAIL, tp = trails[opt[i]];
      let k = (head + 1) % TRAIL;
      tp.moveTo(toScreenX(hx[o + k]), toScreenY(hy[o + k]));
      for (let s = 1; s < TRAIL; s++) { k = (k + 1) % TRAIL; tp.lineTo(toScreenX(hx[o + k]), toScreenY(hy[o + k])); }
      const sx = toScreenX(px[i]), sy = toScreenY(py[i]);
      const p = rest[i] ? resting : moving[opt[i]];
      p.moveTo(sx + 1.8, sy); p.arc(sx, sy, 1.8, 0, 6.2832);
    }
    ctx.lineWidth = 1;
    for (let o = 0; o < 3; o++) { ctx.strokeStyle = colors.opt[o]; ctx.globalAlpha = 0.34; ctx.stroke(trails[o]); }
    ctx.globalAlpha = 0.75; ctx.fillStyle = colors.settled; ctx.fill(resting);
    for (let o = 0; o < 3; o++) { ctx.globalAlpha = 0.95; ctx.fillStyle = colors.opt[o]; ctx.fill(moving[o]); }
    ctx.globalAlpha = 1;
  }

  // ---------- HUD (optional) ----------
  const hT = $('h-t'), hRest = $('h-rest'), hCur = $('h-cursor');
  const hasHud = !!(hT && hRest && hCur);
  let pointer = null, hudTick = 0;
  const pct = o => optCount[o] ? Math.round(restCount[o] / optCount[o] * 100) + '%' : '–';
  function updateHud() {
    if (!hasHud) return;
    hT.textContent = T.toFixed(0);
    hRest.innerHTML = S.opt === 3
      ? [0, 1, 2].map(o => `${SHORT[o]} <b>${pct(o)}</b>`).join(' · ')
      : `<b>${pct(S.opt)}</b>`;
    if (pointer) {
      const x = toWorldX(pointer[0]), y = toWorldY(pointer[1]);
      const v = f(x, y), g = grad(x, y);
      hCur.textContent = `θ = (${x.toFixed(2)}, ${(-y).toFixed(2)})  f(θ) = ${v.toFixed(3)}  ‖∇f‖ = ${Math.hypot(g[0], g[1]).toFixed(2)}`;
    }
  }

  // ---------- Loop ----------
  let last = 0, raf = 0;
  function tick() { T += (1 / 60) * S.drift; updateTerms(); stepParticles(); }
  function frame(now) {
    raf = 0;
    const dt = last ? Math.min(0.1, (now - last) / 1000) : 1 / 60;
    last = now;
    const steps = Math.max(1, Math.min(3, Math.round(dt * 60)));
    for (let s = 0; s < steps; s++) tick();
    draw();
    if (++hudTick % 6 === 0) updateHud();
    schedule();
  }
  function schedule() { if (S.playing && !document.hidden && !raf) raf = requestAnimationFrame(frame); }
  function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; last = 0; }
  function redrawIfPaused() { if (!S.playing) { draw(); updateHud(); } }
  function setPlaying(p) {
    S.playing = p;
    const b = $('b-play');
    if (b) b.textContent = p ? 'Pause' : 'Play';
    if (p) schedule(); else stop();
  }

  // ---------- Tuning panel (optional) ----------
  function initPanel() {
    if (!$('panel')) return;
    const RULES = [
      '<b>Momentum GD</b><br>v ← βv − η∇f<br>θ ← θ + v',
      '<b>AdamW</b><br>m ← β₁m + (1−β₁)∇f, v ← β₂v + (1−β₂)∇f²<br>θ ← θ − η(m̂ / (√v̂ + ε) + λθ)<br>β₂ = 0.999',
      '<b>Muon</b><br>b ← μb + ∇f<br>θ ← θ − η · orth(∇f + μb)<br>With only two parameters, orth(x) = x / ‖x‖, so every step has length η.',
      '<b>Compare all</b><br>A third of the points run each optimizer on the same surface, from the same noise level. The momentum slider sets β, β₁ and μ together.'
    ];
    const BETA_NAMES = ['Momentum β', 'First moment β₁', 'Momentum μ', 'Momentum β · β₁ · μ'];
    const fmt = {
      n: v => String(v), drift: v => v.toFixed(1) + '×',
      lr: v => S.opt === 3 ? '×' + v.toFixed(1) : 'η = ' + (BASE_LR[S.opt] * v).toExponential(1),
      beta: v => v.toFixed(2), wd: v => v.toFixed(2), temp: v => 'σ = ' + (v * 4).toFixed(1)
    };
    const sliders = [['c-n', 'n', 'o-n'], ['c-drift', 'drift', 'o-drift'], ['c-lr', 'lr', 'o-lr'], ['c-beta', 'beta', 'o-beta'], ['c-wd', 'wd', 'o-wd'], ['c-temp', 'temp', 'o-temp']];
    const refreshOutputs = () => sliders.forEach(([, key, out]) => { $(out).textContent = fmt[key](S[key]); });
    sliders.forEach(([inp, key]) => {
      const el = $(inp);
      S[key] = parseFloat(el.value);
      el.addEventListener('input', () => {
        S[key] = parseFloat(el.value);
        if (key === 'n') allocParticles(S.n);
        refreshOutputs();
        redrawIfPaused();
      });
    });

    const optBtns = document.querySelectorAll('[data-opt]');
    function setOptimizer(o) {
      S.opt = o;
      optBtns.forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.opt === o)));
      $('rule').innerHTML = RULES[o];
      $('beta-name').textContent = BETA_NAMES[o];
      $('row-wd').hidden = !(o === 1 || o === 3);
      if (N) { assignOptimizers(); redrawIfPaused(); }
      refreshOutputs();
    }
    optBtns.forEach(b => b.addEventListener('click', () => setOptimizer(+b.dataset.opt)));
    setOptimizer(S.opt);

    const vBg = $('v-bg'), vShow = $('v-show');
    function setView(show) {
      document.body.classList.toggle('showcase', show);
      vBg.setAttribute('aria-pressed', String(!show));
      vShow.setAttribute('aria-pressed', String(show));
    }
    vBg.addEventListener('click', () => setView(false));
    vShow.addEventListener('click', () => setView(true));

    $('b-play').addEventListener('click', () => setPlaying(!S.playing));
    $('b-seed').addEventListener('click', () => {
      seedLandscape(); updateTerms();
      for (let i = 0; i < N; i++) spawn(i);
      if (!S.playing) { for (let s = 0; s < 240; s++) tick(); redrawIfPaused(); }
    });
  }

  // ---------- Events ----------
  window.addEventListener('pointermove', e => { pointer = [e.clientX, e.clientY]; if (!S.playing) updateHud(); }, { passive: true });
  // Click on empty background drops a cluster of new points.
  window.addEventListener('pointerdown', e => {
    if (e.target.closest('a, button, input, .panel, .column, .contact, .nav, .sheet')) return;
    dropAt(toWorldX(e.clientX), toWorldY(e.clientY));
    redrawIfPaused();
  });
  let rt = 0;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { resize(); updateTerms(); draw(); }, 120); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else schedule(); });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readTheme);
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---------- Boot ----------
  resize(); seedLandscape(); updateTerms(); readTheme();
  initPanel();
  allocParticles(S.n);
  for (let s = 0; s < (reduceMotion ? 600 : 90); s++) tick();
  draw();
  setPlaying(S.playing);
  updateHud();
})();
