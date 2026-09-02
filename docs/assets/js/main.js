/* ═══════════════════════════════════════════════════════════════
   Sound Sparks Motion — Main JS
   ═══════════════════════════════════════════════════════════════ */

/* ── Waveform canvas animation ───────────────────────────────── */
(function initWaveform() {
  const canvas = document.getElementById('waveform-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let raf, time = 0;

  const WAVES = [
    { color: 'rgba(245, 158, 11, 0.25)', freq: 0.012, amp: 28, speed: 0.00055, phase: 0 },
    { color: 'rgba(139, 92, 246, 0.18)', freq: 0.008, amp: 42, speed: 0.00038, phase: Math.PI * 0.7 },
    { color: 'rgba(245, 158, 11, 0.13)', freq: 0.018, amp: 16, speed: 0.00070, phase: Math.PI * 1.4 },
    { color: 'rgba(139, 92, 246, 0.10)', freq: 0.005, amp: 55, speed: 0.00025, phase: Math.PI * 2.1 },
  ];

  function resize() {
    canvas.width  = canvas.offsetWidth  * devicePixelRatio;
    canvas.height = canvas.offsetHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
  }

  function draw() {
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    ctx.clearRect(0, 0, W, H);
    for (const w of WAVES) {
      ctx.beginPath();
      ctx.strokeStyle = w.color;
      ctx.lineWidth = 1.6;
      for (let x = 0; x <= W; x += 2) {
        const y = H * 0.5 + w.amp * Math.sin(w.freq * x + w.phase + time * w.speed * 50);
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      w.phase += w.speed;
    }
    time += 1;
    raf = requestAnimationFrame(draw);
  }

  const visObs = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) { resize(); draw(); }
    else cancelAnimationFrame(raf);
  }, { threshold: 0 });
  visObs.observe(canvas);

  new ResizeObserver(resize).observe(canvas);
})();


/* ── Nav background on scroll ────────────────────────────────── */
(function initNav() {
  const nav = document.getElementById('nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.style.background = window.scrollY > 40
      ? 'rgba(7, 7, 15, 0.92)'
      : 'rgba(7, 7, 15, 0.75)';
  }, { passive: true });
})();


/* ── Fade-in lazy images ─────────────────────────────────────── */
(function initLazyImages() {
  document.querySelectorAll('img[loading="lazy"]').forEach(img => {
    if (img.complete) img.classList.add('loaded');
    else img.addEventListener('load', () => img.classList.add('loaded'));
  });
})();


/* ── Lazy-load videos (data-src) via IntersectionObserver ─────── */
(function initLazyVideos() {
  const videos = document.querySelectorAll('video[data-src]');
  if (!videos.length) return;

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const video = entry.target;
      if (!video.dataset.src) return;

      video.src = video.dataset.src;
      delete video.dataset.src;
      video.load();

      video.addEventListener('canplay', () => {
        const stage = video.closest('.spark-stage');
        const sk = stage && stage.querySelector('.vc-skeleton');
        if (sk) sk.classList.add('hidden');
      }, { once: true });

      obs.unobserve(video);
    });
  }, { rootMargin: '200px' });

  videos.forEach(v => obs.observe(v));
})();


/* ── Spark reveal: source video becomes the edited result in place ── */
(function initSparkCards() {
  const cards = document.querySelectorAll('.comparison-card');
  if (!cards.length) return;

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  cards.forEach((card, idx) => {
    const stage = card.querySelector('.spark-stage');
    const vIn   = card.querySelector('.sv-in');
    const vOut  = card.querySelector('.sv-out');
    const btn   = card.querySelector('[data-spark]');
    const txt   = card.querySelector('[data-spark-text]');
    const hint  = card.querySelector('[data-spark-hint]');
    const chip  = card.querySelector('[data-state-chip]');
    if (!stage || !vIn || !vOut || !btn) return;

    const srcLabel = card.dataset.sourceLabel || 'Source';
    const resLabel = card.dataset.resultLabel || 'Sparked';
    const isTransfer = card.classList.contains('is-transfer');

    let revealed = false;
    let busy     = false;

    /* Pull the result clip in only when it is actually needed. */
    function ensureResult() {
      if (vOut.dataset.outSrc) {
        vOut.src = vOut.dataset.outSrc;
        delete vOut.dataset.outSrc;
        vOut.load();
      }
      if (vOut.readyState >= 3) return Promise.resolve();
      return new Promise(resolve => {
        let done = false;
        const finish = () => { if (!done) { done = true; resolve(); } };
        vOut.addEventListener('canplay', finish, { once: true });
        setTimeout(finish, 5000);          // never block the interaction forever
      });
    }

    /* Warm the result up once the card has been on screen for a moment. */
    const warmObs = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      warmObs.disconnect();
      setTimeout(() => { if (vOut.dataset.outSrc) ensureResult(); }, 1200);
    }, { threshold: 0.3 });
    warmObs.observe(stage);

    /* Only the visible layer plays. */
    const playObs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) (revealed ? vOut : vIn).play().catch(() => {});
      else { vIn.pause(); vOut.pause(); }
    }, { threshold: 0.25 });
    playObs.observe(stage);

    function setPhase(phase) {
      if (phase === 'result') {
        stage.classList.add('revealed');
        card.classList.add('is-revealed');
        if (chip) chip.textContent = resLabel;
      } else {
        stage.classList.remove('revealed');
        card.classList.remove('is-revealed');
        if (chip) chip.textContent = srcLabel;
      }
    }

    function reveal() {
      busy = true;
      btn.disabled = true;
      btn.classList.remove('pulse');
      if (txt) txt.textContent = isTransfer ? 'Transferring…' : 'Sparking…';

      ensureResult().then(() => {
        /* Hand the result the source's playhead so the cut is invisible. */
        const len = vOut.duration || vIn.duration || 0;
        if (len) { try { vOut.currentTime = vIn.currentTime % len; } catch (e) {} }
        vOut.play().catch(() => {});

        stage.classList.add('sparking');
        setTimeout(() => setPhase('result'), reduce ? 0 : 340);
        setTimeout(() => {
          stage.classList.remove('sparking');
          vIn.pause();
          revealed = true;
          busy = false;
          btn.disabled = false;
          if (txt)  txt.textContent  = 'Show the source';
          if (hint) hint.textContent = isTransfer
            ? 'Controls learned elsewhere — applied here'
            : 'Motion applied — nothing else moved';
        }, reduce ? 420 : 1050);
      });
    }

    function restore() {
      busy = true;
      btn.disabled = true;
      const len = vIn.duration || vOut.duration || 0;
      if (len) { try { vIn.currentTime = vOut.currentTime % len; } catch (e) {} }
      vIn.play().catch(() => {});

      stage.classList.add('reverting');
      setPhase('source');
      setTimeout(() => {
        stage.classList.remove('reverting');
        vOut.pause();
        revealed = false;
        busy = false;
        btn.disabled = false;
        if (txt)  txt.textContent  = isTransfer ? 'Transfer the motion!' : 'Spark the motion!';
        if (hint) hint.textContent = 'Same clip — watch the motion appear';
      }, reduce ? 400 : 720);
    }

    if (isTransfer && txt) txt.textContent = 'Transfer the motion!';
    if (idx === 0) btn.classList.add('pulse');

    btn.addEventListener('click', () => {
      if (busy) return;
      revealed ? restore() : reveal();
    });
  });
})();


/* ── Evaluation bar charts ───────────────────────────────────── */
(function initEvalCharts() {
  const charts = document.querySelectorAll('[data-chart]');
  if (!charts.length) return;

  function countUp(el, target, delay) {
    setTimeout(() => {
      const dur = 1100;
      const t0  = performance.now();
      (function tick(now) {
        const p = Math.min((now - t0) / dur, 1);
        const e = 1 - Math.pow(1 - p, 3);
        el.textContent = (target * e).toFixed(1) + '%';
        if (p < 1) requestAnimationFrame(tick);
      })(t0);
    }, delay);
  }

  charts.forEach(chart => {
    const fills = Array.from(chart.querySelectorAll('.bar-fill'));
    const vals  = Array.from(chart.querySelectorAll('.bar-val'));
    if (!fills.length) return;

    /* Bars are scaled to the leader, exactly as in the paper figures. */
    const max = Math.max.apply(null, fills.map(f => parseFloat(f.dataset.pct))) || 1;

    const obs = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      obs.disconnect();
      fills.forEach(f => {
        const pct = parseFloat(f.dataset.pct) || 0;
        f.style.width = (pct > 0 ? Math.max(pct / max * 100, 1.5) : 0.9) + '%';
      });
      vals.forEach((el, i) => countUp(el, parseFloat(el.dataset.val) || 0, i * 80));
    }, { threshold: 0.3 });

    obs.observe(chart);
  });
})();


/* ── Carousel arrow buttons ──────────────────────────────────── */
(function initCarousel() {
  const carousel = document.getElementById('results-carousel');
  const btnLeft  = document.getElementById('arrow-left');
  const btnRight = document.getElementById('arrow-right');
  if (!carousel || !btnLeft || !btnRight) return;

  function scrollAmt() { return Math.min(980, window.innerWidth - 80); }

  function updateArrows() {
    btnLeft.classList.toggle('hidden',  carousel.scrollLeft <= 8);
    btnRight.classList.toggle('hidden', carousel.scrollLeft + carousel.clientWidth >= carousel.scrollWidth - 8);
  }

  btnLeft.addEventListener('click',  () => carousel.scrollBy({ left: -scrollAmt(), behavior: 'smooth' }));
  btnRight.addEventListener('click', () => carousel.scrollBy({ left:  scrollAmt(), behavior: 'smooth' }));
  carousel.addEventListener('scroll', updateArrows, { passive: true });
  updateArrows();
})();


/* ── Scroll-reveal ───────────────────────────────────────────── */
(function initScrollReveal() {
  const targets = document.querySelectorAll(
    '.comparison-card, .method-step, .abstract-text, .teaser-figure, .method-figure, .eval-card, .eval-head'
  );
  if (!targets.length) return;
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.06, rootMargin: '0px 0px -40px 0px' });
  targets.forEach(el => { el.classList.add('reveal'); obs.observe(el); });
})();



/* ── Active nav link on scroll ───────────────────────────────── */
(function initActiveNav() {
  const sections = document.querySelectorAll('section[id]');
  const links    = document.querySelectorAll('.nav-links a');
  if (!sections.length || !links.length) return;
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      links.forEach(a => {
        a.style.color = a.getAttribute('href') === `#${e.target.id}` ? 'var(--text-1)' : '';
      });
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  sections.forEach(s => obs.observe(s));
})();


/* ── BibTeX copy button ──────────────────────────────────────── */
(function initBibtexCopy() {
  const btn  = document.getElementById('bibtex-copy-btn');
  const pre  = document.getElementById('bibtex-entry');
  if (!btn || !pre) return;
  btn.addEventListener('click', () => {
    navigator.clipboard.writeText(pre.textContent.trim()).then(() => {
      const span = btn.querySelector('span');
      btn.classList.add('copied');
      span.textContent = 'Copied!';
      setTimeout(() => { btn.classList.remove('copied'); span.textContent = 'Copy'; }, 2000);
    });
  });
})();
