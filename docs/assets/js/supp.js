/* ═══════════════════════════════════════════════════════════════
   Supplementary additions — transfer cards and per-card sound.
   Kept out of main.js so the original site script is unchanged.
   ═══════════════════════════════════════════════════════════════ */

/* ── Motion transfer: donor ▸ beam ▸ recipient ──────────────────
   Three clips per card. The donor (the video the optimized latent
   was tuned on) plays on the left. The new input plays on the
   right. Clicking sends a pip along the rail, and when it lands
   the recipient crossfades to the transferred result — so the
   causal order the paper describes is what the eye actually sees.
   ─────────────────────────────────────────────────────────────── */
(function initTransferCards() {
  const cards = document.querySelectorAll('.transfer-card');
  if (!cards.length) return;

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  cards.forEach(card => {
    const donor = card.querySelector('.tc-donor video');
    const stage = card.querySelector('.tc-recipient');
    const vIn   = card.querySelector('.tc-recipient .sv-in');
    const vOut  = card.querySelector('.tc-recipient .sv-out');
    const beam  = card.querySelector('.tc-beam');
    const pip   = card.querySelector('.tc-beam .pip');
    const btn   = card.querySelector('[data-xfer]');
    const txt   = card.querySelector('[data-xfer-text]');
    const hint  = card.querySelector('[data-xfer-hint]');
    if (!donor || !stage || !vIn || !vOut || !btn) return;

    let revealed = false;
    let busy     = false;
    /* Same single source of truth as the spark cards (main.js): which clip is
       showing and whether the card is on screen. The donor always plays while
       visible. Before this, a card scrolled away and back mid-transfer could
       return with the visible clip frozen. */
    let showing  = 'in';
    let inView   = false;

    function sync() {
      if (!inView) { donor.pause(); vIn.pause(); vOut.pause(); return; }
      donor.play().catch(() => {});
      const on = showing === 'out' ? vOut : vIn, off = on === vOut ? vIn : vOut;
      on.play().catch(() => {});
      if (busy) off.play().catch(() => {}); else off.pause();
    }

    /* The rail width is only known after layout, so the pip's travel
       distance is set from the measured element rather than guessed. */
    function sizeBeam() {
      if (!beam || !pip) return;
      const w = beam.getBoundingClientRect().width;
      if (w > 20) pip.style.setProperty('--travel', (w - 22) + 'px');
    }
    sizeBeam();
    window.addEventListener('resize', sizeBeam);

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
        setTimeout(finish, 5000);
      });
    }

    /* Only play what is on screen. */
    const obs = new IntersectionObserver(([e]) => {
      inView = e.isIntersecting;
      if (inView) sizeBeam();
      sync();
    }, { threshold: 0.2 });
    obs.observe(card);

    const warm = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      warm.disconnect();
      setTimeout(() => { if (vOut.dataset.outSrc) ensureResult(); }, 1200);
    }, { threshold: 0.3 });
    warm.observe(card);

    function transfer() {
      busy = true;
      showing = 'out';
      btn.disabled = true;
      card.classList.add('is-armed');
      if (txt) txt.textContent = 'Transferring…';

      const travel = reduce ? 0 : 900;
      if (!reduce) { sizeBeam(); card.classList.add('is-beaming'); }

      ensureResult().then(() => {
        setTimeout(() => {
          card.classList.remove('is-beaming');
          card.classList.add('is-landing');

          const len = vOut.duration || vIn.duration || 0;
          if (len) { try { vOut.currentTime = vIn.currentTime % len; } catch (e) {} }
          sync();
          stage.classList.add('revealed');

          setTimeout(() => {
            card.classList.remove('is-landing');
            revealed = true;
            busy = false;
            sync();
            btn.disabled = false;
            if (txt)  txt.textContent  = 'Show the new input';
            if (hint) hint.textContent = '';
          }, reduce ? 0 : 700);
        }, travel);
      });
    }

    function restore() {
      busy = true;
      showing = 'in';
      btn.disabled = true;
      card.classList.remove('is-armed');

      const len = vIn.duration || vOut.duration || 0;
      if (len) { try { vIn.currentTime = vOut.currentTime % len; } catch (e) {} }
      sync();
      stage.classList.remove('revealed');

      setTimeout(() => {
        revealed = false;
        busy = false;
        sync();
        btn.disabled = false;
        if (txt)  txt.textContent  = 'Transfer the motion!';
        if (hint) hint.textContent = '';
      }, reduce ? 0 : 480);
    }

    btn.addEventListener('click', () => {
      if (busy) return;
      revealed ? restore() : transfer();
    });
  });
})();


/* ── Per-card sound (H3 cards only) ─────────────────────────────
   H3 is audio-visual, so its outputs carry a real soundtrack and
   these clips are worth hearing. Only one card plays sound at a
   time — unmuting a second one silences the first.
   ─────────────────────────────────────────────────────────────── */
(function initSoundToggles() {
  const btns = document.querySelectorAll('[data-snd]');
  if (!btns.length) return;

  let active = null;

  function layers(btn) {
    const card = btn.closest('.comparison-card');
    return card ? card.querySelectorAll('video') : [];
  }

  function silence(btn) {
    if (!btn) return;
    layers(btn).forEach(v => { v.muted = true; });
    btn.classList.remove('on');
    btn.setAttribute('aria-label', 'Unmute this clip');
  }

  btns.forEach(btn => {
    btn.setAttribute('aria-label', 'Unmute this clip');
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const on = btn.classList.contains('on');
      if (on) { silence(btn); active = null; return; }
      if (active && active !== btn) silence(active);
      layers(btn).forEach(v => { v.muted = false; });
      btn.classList.add('on');
      btn.setAttribute('aria-label', 'Mute this clip');
      active = btn;
    });
  });
})();


/* ── Scroll cue ─────────────────────────────────────────────────────────────
   The hero fills the first screen, so the cue says there is more below. It
   retires as soon as the reader scrolls — leaving it up would be nagging.
   ─────────────────────────────────────────────────────────────────────────── */
(function initScrollCue() {
  const cue = document.querySelector('[data-scroll-cue]');
  if (!cue) return;
  const hide = () => {
    cue.classList.add('gone');
    window.removeEventListener('scroll', onScroll);
  };
  const onScroll = () => { if (window.scrollY > 40) hide(); };
  window.addEventListener('scroll', onScroll, { passive: true });
  cue.addEventListener('click', () => {
    const first = document.querySelector('#videos');
    if (first) first.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  cue.style.cursor = 'pointer';
})();

/* ── attention cards ───────────────────────────────────────────────────────
   Two clips per card, same length, same first frame. Only the visible one
   plays: the first cut kept both running and seeked the hidden one on every
   toggle, which made the swap stutter and sometimes land on the wrong frame,
   because the reveal happened before the seek had completed. Now the incoming
   clip is seeked first and only shown once it reports `seeked`, so the switch
   is a clean cut at the same moment of the same clip.
   Sources attach only when the card comes into view: sixteen decoders started
   at once made phones stutter badly. */
(function initAttention() {
  const cards = document.querySelectorAll('.att-card');
  if (!cards.length) return;

  cards.forEach(card => {
    const vids = [...card.querySelectorAll('.att-v')];
    const btns = [...card.querySelectorAll('.att-toggle button')];
    let loaded = false, inView = false, swapping = false;

    const active = () => vids.find(v => v.classList.contains('is-on')) || vids[0];

    function load() {
      if (loaded) return;
      loaded = true;
      const sk = card.querySelector('.vc-skeleton');
      vids.forEach(v => {
        // sync() runs the moment the observer fires, which is the same tick the
        // source is attached — play() there has no data yet and is rejected.
        // Retrying on canplay is what actually starts the card.
        v.addEventListener('canplay', () => {
          if (sk) sk.classList.add('hidden');
          sync();
        });
        v.src = v.dataset.attSrc;
        v.load();
      });
    }

    function sync() {
      vids.forEach(v => {
        if (v === active() && inView) v.play().catch(() => {});
        else v.pause();
      });
    }

    function show(mode) {
      const from = active();
      const to = vids.find(v => v.dataset.mode === mode);
      if (!to || to === from || swapping) return;
      swapping = true;

      const reveal = () => {
        vids.forEach(v => v.classList.toggle('is-on', v === to));
        btns.forEach(b => b.classList.toggle('is-on', b.dataset.mode === mode));
        from.pause();
        if (inView) to.play().catch(() => {});
        swapping = false;
      };

      const t = from.currentTime;
      // readyState 1 has metadata but cannot seek yet; swap straight over and
      // let it start from wherever it is rather than hanging on `seeked`.
      if (to.readyState < 2 || Math.abs(to.currentTime - t) < 0.04) return reveal();
      let done = false;
      const once = () => { if (!done) { done = true; reveal(); } };
      to.addEventListener('seeked', once, { once: true });
      setTimeout(once, 400);                 // never leave the card mid-swap
      try { to.currentTime = t; } catch (e) { once(); }
    }

    btns.forEach(b => b.addEventListener('click', () => show(b.dataset.mode)));

    new IntersectionObserver(entries => {
      entries.forEach(e => { inView = e.isIntersecting; if (inView) load(); sync(); });
    }, { rootMargin: '200px 0px', threshold: 0.01 }).observe(card);
  });
})();

/* ── baseline comparison ───────────────────────────────────────────────────
   One scenario at a time. Nothing loads until the section is close: the grid
   used to attach all nine sources at page load, so a reviewer opening the page
   pulled ~1 MB of clips from 3,000 px below the fold. On phones it switches to
   a pair — the chosen clip beside ours — because nine stacked cells ran to
   2,200 px for a single scenario. */
(function initBaselines() {
  const wrap = document.querySelector('.cmp-wrap');
  if (!wrap) return;
  const read = id => {
    const el = document.getElementById(id);
    try { return el ? JSON.parse(el.textContent) : null; } catch (e) { return null; }
  };
  const DATA = read('cmp-data'), LABEL = read('cmp-labels'), ORDER = read('cmp-order');
  if (!DATA || !LABEL || !ORDER) return;

  const grid = wrap.querySelector('[data-cmp-grid]');
  const chips = wrap.querySelector('[data-cmp-chips]');
  const promptEl = wrap.querySelector('[data-cmp-prompt]');
  const playBtn = wrap.querySelector('[data-cmp-play]');
  const playTxt = wrap.querySelector('[data-cmp-play-text]');
  const picks = wrap.querySelectorAll('.cmp-pick');
  const phone = window.matchMedia('(max-width: 560px)');

  let slug = (wrap.querySelector('.cmp-pick.is-on') || picks[0] || {}).dataset?.cmp;
  let other = 'input';          // what ours is compared against in pair mode
  let playing = true, inView = false, started = false;

  const shouldPlay = () => playing && inView;
  const playAll = () => grid.querySelectorAll('video')
    .forEach(v => shouldPlay() ? v.play().catch(() => {}) : v.pause());
  const keysFor = row => ORDER.filter(k => k === 'input' || k === 'ours' || row.methods.includes(k));

  function renderChips(row) {
    chips.innerHTML = '';
    if (!phone.matches) return;
    keysFor(row).filter(k => k !== 'ours').forEach(k => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = LABEL[k] || k;
      b.classList.toggle('is-on', k === other);
      b.setAttribute('aria-pressed', String(k === other));
      b.addEventListener('click', () => { other = k; render(); });
      chips.append(b);
    });
  }

  function render() {
    const row = DATA[slug];
    if (!row) return;
    promptEl.innerHTML = '"' + row.prompt.replace(/\{/g, '<em>').replace(/\}/g, '</em>') + '"';
    grid.querySelectorAll('video').forEach(v => { v.pause(); v.removeAttribute('src'); v.load(); });
    grid.innerHTML = '';
    let keys = keysFor(row);
    if (phone.matches) {
      if (!keys.includes(other)) other = 'input';
      keys = [other, 'ours'];
    }
    grid.classList.toggle('is-pair', phone.matches);
    keys.forEach(k => {
      const fig = document.createElement('figure');
      fig.className = 'cmp-cell' + (k === 'ours' ? ' is-ours' : k === 'input' ? ' is-input' : '');
      const v = document.createElement('video');
      v.loop = true; v.muted = true; v.playsInline = true; v.preload = 'auto';
      v.poster = `assets/posters/cmp_${slug}_${k}.jpg`;
      v.src = `assets/videos/cmp/${slug}_${k}.mp4`;
      v.addEventListener('canplay', () => { if (shouldPlay()) v.play().catch(() => {}); });
      const cap = document.createElement('figcaption');
      cap.textContent = LABEL[k] || k;
      fig.append(v, cap);
      grid.append(fig);
    });
    renderChips(row);
    sync();
  }

  /* Restart together: they are normalised to one length, and letting each
     start when its own data arrived put the grid visibly out of step. */
  function sync() {
    const vids = [...grid.querySelectorAll('video')];
    Promise.all(vids.map(v => v.readyState >= 2 ? Promise.resolve()
      : new Promise(r => v.addEventListener('loadeddata', r, { once: true }))))
      .then(() => { vids.forEach(v => { v.currentTime = 0; }); playAll(); });
  }

  playBtn.addEventListener('click', () => {
    playing = !playing;
    playAll();
    playTxt.textContent = playing ? 'Pause' : 'Play';
  });

  picks.forEach(b => b.addEventListener('click', () => {
    picks.forEach(o => {
      o.classList.toggle('is-on', o === b);
      o.setAttribute('aria-selected', String(o === b));
    });
    slug = b.dataset.cmp;
    if (started) render();
  }));

  phone.addEventListener('change', () => { if (started) render(); });

  // First render only once the section is near; playback only while visible.
  new IntersectionObserver((es, io) => {
    if (es.some(e => e.isIntersecting)) { started = true; render(); io.disconnect(); }
  }, { rootMargin: '700px 0px' }).observe(wrap);
  new IntersectionObserver(es => {
    es.forEach(e => { inView = e.isIntersecting; playAll(); });
  }, { rootMargin: '150px 0px', threshold: 0.01 }).observe(wrap);
})();

/* ── posters, a section at a time ──────────────────────────────────────────
   Every card poster used to load on first paint — 64 images for a page the
   reviewer sees the top of. They now load as their section comes within
   ~900 px, and a whole section at once, so cards further along a carousel
   are ready before they are swiped into view. */
(function initLazyPosters() {
  const set = v => { v.poster = v.dataset.poster; delete v.dataset.poster; };
  const pending = document.querySelectorAll('video[data-poster]');
  if (!pending.length) return;
  if (!('IntersectionObserver' in window)) { pending.forEach(set); return; }
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.querySelectorAll('video[data-poster]').forEach(set);
    io.unobserve(e.target);
  }), { rootMargin: '900px 0px' });
  document.querySelectorAll('section').forEach(sec => io.observe(sec));
})();

/* ── carousel arrows ───────────────────────────────────────────────────────
   Both carousels shipped without any. Step by as many whole cards as fit, so
   snapping never leaves one half-shown. */
(function initCarousels() {
  document.querySelectorAll('.carousel-wrapper').forEach(w => {
    const c = w.querySelector('.results-carousel');
    const L = w.querySelector('.arrow-left'), R = w.querySelector('.arrow-right');
    if (!c || !L || !R) return;
    const step = () => {
      const card = c.querySelector('.comparison-card');
      if (!card) return c.clientWidth * 0.85;
      const gap = parseFloat(getComputedStyle(c).columnGap) || 24;
      const unit = card.getBoundingClientRect().width + gap;
      return unit * Math.max(1, Math.floor((c.clientWidth - 48) / unit));
    };
    const update = () => {
      L.classList.toggle('hidden', c.scrollLeft <= 8);
      R.classList.toggle('hidden', c.scrollLeft + c.clientWidth >= c.scrollWidth - 8);
    };
    L.addEventListener('click', () => c.scrollBy({ left: -step(), behavior: 'smooth' }));
    R.addEventListener('click', () => c.scrollBy({ left: step(), behavior: 'smooth' }));
    c.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  });
})();

/* ── reel chapters ─────────────────────────────────────────────────────────
   The reels use preload="none", so a chapter clicked before the video has
   loaded has to fetch metadata first — seeking a video with no duration is a
   no-op. The current chapter stays highlighted during playback. */
(function initChapters() {
  document.querySelectorAll('.reel').forEach(reel => {
    const v = reel.querySelector('video');
    const btns = [...reel.querySelectorAll('.reel-chapters button')];
    if (!v || !btns.length) return;
    const starts = btns.map(b => parseFloat(b.dataset.t));
    const seek = t => {
      const go = () => { v.currentTime = t; v.play().catch(() => {}); };
      if (v.readyState >= 1) return go();
      v.addEventListener('loadedmetadata', go, { once: true });
      v.preload = 'metadata';
      v.load();
    };
    btns.forEach((b, i) => b.addEventListener('click', () => seek(starts[i])));
    v.addEventListener('timeupdate', () => {
      let cur = 0;
      starts.forEach((t, i) => { if (v.currentTime >= t - 0.05) cur = i; });
      btns.forEach((b, i) => b.classList.toggle('is-on', i === cur));
    });
  });
})();
