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
    { color: 'rgba(245, 158, 11, 0.18)', freq: 0.018, amp: 45, speed: 0.0022, phase: 0 },
    { color: 'rgba(139, 92, 246, 0.12)', freq: 0.013, amp: 65, speed: 0.0015, phase: Math.PI * 0.6 },
    { color: 'rgba(245, 158, 11, 0.08)', freq: 0.024, amp: 28, speed: 0.0028, phase: Math.PI * 1.2 },
    { color: 'rgba(139, 92, 246, 0.07)', freq: 0.009, amp: 80, speed: 0.0010, phase: Math.PI * 1.8 },
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

  // pause when hidden to save CPU
  const observer = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting) { resize(); draw(); }
      else { cancelAnimationFrame(raf); }
    },
    { threshold: 0 }
  );
  observer.observe(canvas);

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
})();


/* ── Nav shrink on scroll ────────────────────────────────────── */
(function initNav() {
  const nav = document.getElementById('nav');
  if (!nav) return;

  let lastY = 0;
  window.addEventListener('scroll', () => {
    const y = window.scrollY;
    nav.style.background = y > 40
      ? 'rgba(7, 7, 15, 0.92)'
      : 'rgba(7, 7, 15, 0.75)';
    lastY = y;
  }, { passive: true });
})();


/* ── Lazy-load images (fade-in on load) ──────────────────────── */
(function initLazyImages() {
  const imgs = document.querySelectorAll('img[loading="lazy"]');

  imgs.forEach(img => {
    if (img.complete) {
      img.classList.add('loaded');
    } else {
      img.addEventListener('load', () => img.classList.add('loaded'));
    }
  });
})();


/* ── Lazy-load videos via IntersectionObserver ───────────────── */
(function initLazyVideos() {
  const videos = document.querySelectorAll('video[data-src]');
  if (!videos.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;

        const video = entry.target;
        const src   = video.dataset.src;
        if (!src) return;

        video.src = src;
        video.load();

        video.addEventListener('canplay', () => {
          // hide skeleton once video is ready
          const skeleton = video.nextElementSibling;
          if (skeleton && skeleton.classList.contains('video-skeleton')) {
            skeleton.classList.add('hidden');
          }
        }, { once: true });

        observer.unobserve(video);
      });
    },
    { rootMargin: '200px' }  // start loading 200px before entering viewport
  );

  videos.forEach(v => observer.observe(v));
})();


/* ── Comparison card hover: play / pause both videos ─────────── */
(function initVideoHover() {
  const cards = document.querySelectorAll('.comparison-card');

  cards.forEach(card => {
    const videos = card.querySelectorAll('video');

    const playAll = () => videos.forEach(v => { if (v.src) v.play().catch(() => {}); });
    const pauseAll = () => videos.forEach(v => v.pause());

    card.addEventListener('mouseenter', playAll);
    card.addEventListener('mouseleave', pauseAll);

    // also play/pause on touch
    card.addEventListener('touchstart', playAll,  { passive: true });
    card.addEventListener('touchend',   pauseAll, { passive: true });
  });
})();


/* ── Scroll-reveal via IntersectionObserver ──────────────────── */
(function initScrollReveal() {
  const targets = document.querySelectorAll(
    '.reveal, .reveal-stagger, .comparison-card, .method-step, .abstract-text, .teaser-figure, .method-figure'
  );
  if (!targets.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
  );

  targets.forEach(el => {
    el.classList.add('reveal');
    observer.observe(el);
  });
})();


/* ── BibTeX copy button ──────────────────────────────────────── */
(function initCopyBibtex() {
  const btn  = document.getElementById('copy-bibtex');
  const code = document.getElementById('bibtex-code');
  if (!btn || !code) return;

  btn.addEventListener('click', async () => {
    const text = code.querySelector('code')?.textContent ?? code.textContent;

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // fallback for older browsers
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }

    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <rect x="4.5" y="4.5" width="8" height="8" rx="1" stroke="currentColor" stroke-width="1.3"/>
        <path d="M2.5 9.5V2.5a1 1 0 0 1 1-1h7" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
      </svg> Copy`;
      btn.classList.remove('copied');
    }, 2000);
  });
})();


/* ── Carousel arrow buttons ──────────────────────────────────── */
(function initCarousel() {
  const carousel = document.getElementById('results-carousel');
  const btnLeft  = document.getElementById('arrow-left');
  const btnRight = document.getElementById('arrow-right');
  if (!carousel || !btnLeft || !btnRight) return;

  const SCROLL_BY = 480;

  function updateArrows() {
    const atStart = carousel.scrollLeft <= 8;
    const atEnd   = carousel.scrollLeft + carousel.clientWidth >= carousel.scrollWidth - 8;
    btnLeft.classList.toggle('hidden', atStart);
    btnRight.classList.toggle('hidden', atEnd);
  }

  btnLeft.addEventListener('click',  () => carousel.scrollBy({ left: -SCROLL_BY, behavior: 'smooth' }));
  btnRight.addEventListener('click', () => carousel.scrollBy({ left:  SCROLL_BY, behavior: 'smooth' }));
  carousel.addEventListener('scroll', updateArrows, { passive: true });

  updateArrows(); // init state
})();


/* ── Active nav link highlight on scroll ─────────────────────── */
(function initActiveNav() {
  const sections = document.querySelectorAll('section[id]');
  const links    = document.querySelectorAll('.nav-links a');
  if (!sections.length || !links.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        links.forEach(a => {
          a.style.color = a.getAttribute('href') === `#${id}`
            ? 'var(--text-1)'
            : '';
        });
      });
    },
    { rootMargin: '-40% 0px -55% 0px' }
  );

  sections.forEach(s => observer.observe(s));
})();
