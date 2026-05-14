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
    { color: 'rgba(245, 158, 11, 0.10)', freq: 0.012, amp: 28, speed: 0.00055, phase: 0 },
    { color: 'rgba(139, 92, 246, 0.07)', freq: 0.008, amp: 42, speed: 0.00038, phase: Math.PI * 0.7 },
    { color: 'rgba(245, 158, 11, 0.05)', freq: 0.018, amp: 16, speed: 0.00070, phase: Math.PI * 1.4 },
    { color: 'rgba(139, 92, 246, 0.04)', freq: 0.005, amp: 55, speed: 0.00025, phase: Math.PI * 2.1 },
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
      ctx.lineWidth = 1.2;
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
        // Hide skeleton once BOTH videos in a vc-wrap are ready
        const wrap = video.closest('.vc-wrap');
        if (wrap) {
          wrap._ready = (wrap._ready || 0) + 1;
          if (wrap._ready >= 2) {
            const sk = wrap.querySelector('.vc-skeleton');
            if (sk) sk.classList.add('hidden');
          }
        }
      }, { once: true });

      obs.unobserve(video);
    });
  }, { rootMargin: '200px' });

  videos.forEach(v => obs.observe(v));
})();


/* ── Video comparison slider ─────────────────────────────────── */
(function initVideoCompare() {
  const wraps = document.querySelectorAll('.vc-wrap');
  if (!wraps.length) return;

  wraps.forEach(wrap => {
    const lClip = wrap.querySelector('.vc-l-clip');
    const lVid  = wrap.querySelector('.vc-l');
    const rVid  = wrap.querySelector('.vc-r');
    const line  = wrap.querySelector('.vc-line');
    const btn   = wrap.querySelector('.vc-btn');
    if (!lClip || !lVid || !rVid || !line) return;

    let pct = 50;
    let dragging = false;

    /* Keep left video full-width and update divider position */
    function setPos(clientX) {
      const rect = wrap.getBoundingClientRect();
      pct = Math.max(3, Math.min(97, ((clientX - rect.left) / rect.width) * 100));
      line.style.left         = pct + '%';
      lClip.style.width       = pct + '%';
      lVid.style.width        = wrap.offsetWidth + 'px';
    }

    /* Reset left-video width after any layout change */
    function syncWidth() {
      lVid.style.width = wrap.offsetWidth + 'px';
      line.style.left  = pct + '%';
      lClip.style.width = pct + '%';
    }

    /* Keep videos in sync while playing */
    function syncTime() {
      if (!lVid.paused && Math.abs(rVid.currentTime - lVid.currentTime) > 0.12) {
        lVid.currentTime = rVid.currentTime;
      }
    }

    function playBoth()  { rVid.play().catch(() => {}); lVid.play().catch(() => {}); }
    function pauseBoth() { rVid.pause(); lVid.pause(); }

    /* ── Mouse ── */
    wrap.addEventListener('mousedown',  e => { dragging = true; setPos(e.clientX); e.preventDefault(); });
    document.addEventListener('mousemove', e => { if (dragging) setPos(e.clientX); });
    document.addEventListener('mouseup',   () => { dragging = false; });

    /* ── Touch: only drag when starting near the divider line ── */
    wrap.addEventListener('touchstart', e => {
      const touch = e.touches[0];
      const lineX = wrap.getBoundingClientRect().left + (wrap.offsetWidth * pct / 100);
      if (Math.abs(touch.clientX - lineX) < 44) {   // 44px touch target
        dragging = true;
        setPos(touch.clientX);
      }
    }, { passive: true });

    document.addEventListener('touchmove', e => {
      if (!dragging) return;
      e.preventDefault();          // stop page/carousel scroll while dragging
      setPos(e.touches[0].clientX);
    }, { passive: false });

    document.addEventListener('touchend',   () => { dragging = false; });
    document.addEventListener('touchcancel',() => { dragging = false; });

    /* ── Play / pause on hover (desktop) ── */
    wrap.addEventListener('mouseenter', playBoth);
    wrap.addEventListener('mouseleave', pauseBoth);

    /* ── Play / pause on tap (mobile) ── */
    let tapping = false;
    wrap.addEventListener('touchstart', () => { tapping = true; }, { passive: true });
    wrap.addEventListener('touchend', () => {
      if (tapping && !dragging) {
        rVid.paused ? playBoth() : pauseBoth();
      }
      tapping = false;
    }, { passive: true });

    /* ── Sync timestamps while playing ── */
    rVid.addEventListener('timeupdate', syncTime);

    /* ── Init + keep width correct on resize ── */
    syncWidth();
    new ResizeObserver(syncWidth).observe(wrap);
    rVid.addEventListener('loadedmetadata', syncWidth);
    lVid.addEventListener('loadedmetadata', syncWidth);
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
    '.comparison-card, .method-step, .abstract-text, .teaser-figure, .method-figure'
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
