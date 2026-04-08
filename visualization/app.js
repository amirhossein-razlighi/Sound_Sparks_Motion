const SLOT_IDS = ["baseline", "ours"];
const THUMBNAIL_TARGET_WIDTH = 110;
const FILMSTRIP_HEIGHT = 84;
const MAX_SYNC_DRIFT_SECONDS = 0.04;
const DEFAULT_FPS = 25;

const playPauseButton = document.querySelector("#playPauseButton");
const pauseButton = document.querySelector("#pauseButton");
const stepBackButton = document.querySelector("#stepBackButton");
const stepForwardButton = document.querySelector("#stepForwardButton");
const resetButton = document.querySelector("#resetButton");
const timelineSlider = document.querySelector("#timelineSlider");
const timeReadout = document.querySelector("#timeReadout");
const statusText = document.querySelector("#statusText");
const loopCheckbox = document.querySelector("#loopCheckbox");
const fpsInput = document.querySelector("#fpsInput");
const filmstripSurface = document.querySelector("#filmstripSurface");
const playhead = document.querySelector("#playhead");

const state = {
  slots: Object.fromEntries(SLOT_IDS.map((slotId) => [slotId, buildSlotState(slotId)])),
  isPlaying: false,
  sharedDuration: 0,
  rafId: 0,
  resumeAfterScrub: false,
  wasDraggingTimeline: false,
};

function buildSlotState(slotId) {
  const shell = document.querySelector(`[data-slot-shell="${slotId}"]`);
  const video = document.querySelector(`[data-slot-video="${slotId}"]`);
  const input = document.querySelector(`[data-slot-input="${slotId}"]`);
  const fileName = document.querySelector(`[data-slot-file-name="${slotId}"]`);
  const duration = document.querySelector(`[data-slot-duration="${slotId}"]`);
  const empty = document.querySelector(`[data-slot-empty="${slotId}"]`);
  const filmstrip = document.querySelector(`[data-filmstrip="${slotId}"]`);

  const slot = {
    id: slotId,
    shell,
    video,
    input,
    fileName,
    duration,
    empty,
    filmstrip,
    objectUrl: "",
    loaded: false,
    filmstripToken: 0,
  };

  bindSlotEvents(slot);
  drawFilmstripPlaceholder(slot, "Upload a video");
  return slot;
}

function bindSlotEvents(slot) {
  slot.video.addEventListener("loadedmetadata", () => handleVideoLoaded(slot));
  slot.video.addEventListener("ended", () => {
    if (!loopCheckbox.checked) {
      pauseBoth();
      seekBoth(state.sharedDuration);
    }
  });

  slot.input.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (file) {
      loadFileIntoSlot(slot, file);
    }
    event.target.value = "";
  });

  for (const eventName of ["dragenter", "dragover"]) {
    slot.shell.addEventListener(eventName, (event) => {
      event.preventDefault();
      slot.shell.classList.add("is-dragover");
    });
  }

  for (const eventName of ["dragleave", "dragend", "drop"]) {
    slot.shell.addEventListener(eventName, (event) => {
      event.preventDefault();
      slot.shell.classList.remove("is-dragover");
    });
  }

  slot.shell.addEventListener("drop", (event) => {
    const [file] = [...(event.dataTransfer?.files || [])];
    if (file && file.type.startsWith("video/")) {
      loadFileIntoSlot(slot, file);
    }
  });
}

function loadFileIntoSlot(slot, file) {
  revokeObjectUrl(slot);

  slot.objectUrl = URL.createObjectURL(file);
  slot.loaded = false;
  slot.fileName.textContent = file.name;
  slot.duration.textContent = "--:--";
  slot.shell.classList.remove("has-video");
  slot.video.pause();
  slot.video.removeAttribute("src");
  slot.video.load();
  slot.video.src = slot.objectUrl;
  slot.video.currentTime = 0;
  slot.video.load();

  drawFilmstripPlaceholder(slot, "Loading thumbnails...");
  updateSharedState();
}

function revokeObjectUrl(slot) {
  if (slot.objectUrl) {
    URL.revokeObjectURL(slot.objectUrl);
    slot.objectUrl = "";
  }
}

function handleVideoLoaded(slot) {
  slot.loaded = Number.isFinite(slot.video.duration) && slot.video.duration > 0;
  slot.duration.textContent = formatClock(slot.video.duration, { ms: false });
  slot.shell.classList.add("has-video");
  slot.video.currentTime = 0;
  slot.video.pause();

  updateSharedState();
  renderFilmstrip(slot);
}

function updateSharedState() {
  const loadedSlots = SLOT_IDS.map((slotId) => state.slots[slotId]).filter((slot) => slot.loaded);

  if (loadedSlots.length < SLOT_IDS.length) {
    pauseBoth();
    state.sharedDuration = 0;
    timelineSlider.disabled = true;
    playPauseButton.disabled = true;
    pauseButton.disabled = true;
    stepBackButton.disabled = true;
    stepForwardButton.disabled = true;
    resetButton.disabled = true;
    timeReadout.textContent = "00:00.000 / 00:00.000";
    statusText.textContent = "Upload both videos to enable synchronized playback.";
    setPlayheadFromProgress(0);
    return;
  }

  const durations = loadedSlots.map((slot) => slot.video.duration);
  const minDuration = Math.min(...durations);
  const maxDuration = Math.max(...durations);
  state.sharedDuration = minDuration;

  const enabled = minDuration > 0;
  timelineSlider.disabled = !enabled;
  playPauseButton.disabled = !enabled;
  pauseButton.disabled = !enabled;
  stepBackButton.disabled = !enabled;
  stepForwardButton.disabled = !enabled;
  resetButton.disabled = !enabled;

  updateTimeUi(Math.min(getCurrentTime(), minDuration));

  const durationNote =
    maxDuration - minDuration > 0.05
      ? `Comparing the shared ${formatClock(minDuration)} window.`
      : `Both videos are aligned across ${formatClock(minDuration)}.`;
  statusText.textContent = durationNote;
}

function getCurrentTime() {
  const baseline = state.slots.baseline.video;
  return Number.isFinite(baseline.currentTime) ? baseline.currentTime : 0;
}

function getFrameStepSeconds() {
  const value = Number(fpsInput.value);
  const fps = Number.isFinite(value) && value > 0 ? value : DEFAULT_FPS;
  return 1 / fps;
}

function pauseBoth() {
  cancelAnimationFrame(state.rafId);
  state.rafId = 0;
  state.isPlaying = false;
  playPauseButton.textContent = "Play";
  for (const slotId of SLOT_IDS) {
    state.slots[slotId].video.pause();
  }
}

async function playBoth() {
  if (!state.sharedDuration) {
    return;
  }

  state.isPlaying = true;
  playPauseButton.textContent = "Playing";
  const startTime = Math.min(getCurrentTime(), state.sharedDuration);
  seekBoth(startTime);

  const playResults = await Promise.allSettled(
    SLOT_IDS.map((slotId) => state.slots[slotId].video.play()),
  );

  const anyRejected = playResults.some((result) => result.status === "rejected");
  if (anyRejected) {
    pauseBoth();
    statusText.textContent = "The browser blocked playback. Interact with the page and try again.";
    return;
  }

  syncLoop();
}

function syncLoop() {
  if (!state.isPlaying) {
    return;
  }

  const baselineVideo = state.slots.baseline.video;
  const oursVideo = state.slots.ours.video;
  const currentTime = Math.min(baselineVideo.currentTime, state.sharedDuration);

  if (Math.abs(oursVideo.currentTime - currentTime) > MAX_SYNC_DRIFT_SECONDS) {
    oursVideo.currentTime = currentTime;
  }

  updateTimeUi(currentTime);

  if (currentTime >= state.sharedDuration - getFrameStepSeconds() * 0.5) {
    if (loopCheckbox.checked) {
      seekBoth(0);
      void Promise.allSettled(SLOT_IDS.map((slotId) => state.slots[slotId].video.play()));
      state.rafId = requestAnimationFrame(syncLoop);
      return;
    }

    pauseBoth();
    seekBoth(state.sharedDuration);
    return;
  }

  state.rafId = requestAnimationFrame(syncLoop);
}

function seekBoth(targetTime) {
  const safeTime = clamp(targetTime, 0, state.sharedDuration || 0);

  for (const slotId of SLOT_IDS) {
    const slot = state.slots[slotId];
    if (!slot.loaded) {
      continue;
    }

    const maxSeek = Math.max(slot.video.duration - 0.001, 0);
    slot.video.currentTime = clamp(safeTime, 0, maxSeek);
  }

  updateTimeUi(safeTime);
}

function updateTimeUi(currentTime) {
  const duration = state.sharedDuration || 0;
  const progress = duration > 0 ? currentTime / duration : 0;

  timelineSlider.value = String(Math.round(progress * 1000));
  timeReadout.textContent = `${formatClock(currentTime)} / ${formatClock(duration)}`;
  setPlayheadFromProgress(progress);
}

function setPlayheadFromProgress(progress) {
  const clamped = clamp(progress, 0, 1);
  const canvasRect = state.slots.baseline.filmstrip.getBoundingClientRect();
  const surfaceRect = filmstripSurface.getBoundingClientRect();
  const x = canvasRect.left - surfaceRect.left + canvasRect.width * clamped;
  playhead.style.left = `${x}px`;
}

function timelineValueToTime(value) {
  return (Number(value) / 1000) * (state.sharedDuration || 0);
}

function drawFilmstripPlaceholder(slot, message) {
  const { ctx, width, height } = prepareCanvas(slot.filmstrip);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b0f15";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#1b2330";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#2c3442";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  ctx.fillStyle = "#9aa7b8";
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
}

async function renderFilmstrip(slot) {
  if (!slot.loaded || !slot.objectUrl) {
    drawFilmstripPlaceholder(slot, "Upload a video");
    return;
  }

  const token = ++slot.filmstripToken;
  drawFilmstripPlaceholder(slot, "Generating thumbnails...");

  const { ctx, width, height } = prepareCanvas(slot.filmstrip);
  const count = Math.max(6, Math.floor(width / THUMBNAIL_TARGET_WIDTH));
  const duration = slot.video.duration;
  const sampleTimes = buildSampleTimes(duration, count);
  const captureVideo = document.createElement("video");
  captureVideo.src = slot.objectUrl;
  captureVideo.muted = true;
  captureVideo.playsInline = true;
  captureVideo.preload = "auto";

  await once(captureVideo, "loadedmetadata");
  if (captureVideo.readyState < 2) {
    await once(captureVideo, "loadeddata");
  }

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0a0d13";
  ctx.fillRect(0, 0, width, height);

  const cellWidth = width / count;

  for (let index = 0; index < sampleTimes.length; index += 1) {
    if (token !== slot.filmstripToken) {
      return;
    }

    const time = sampleTimes[index];
    await seekVideo(captureVideo, time);

    const x = index * cellWidth;
    const y = 0;
    const thumbWidth = cellWidth;
    const thumbHeight = height;
    drawCoverFrame(ctx, captureVideo, x, y, thumbWidth, thumbHeight);

    ctx.fillStyle = "rgba(0, 0, 0, 0.38)";
    ctx.fillRect(x + 4, height - 20, 52, 16);
    ctx.fillStyle = "#f3f6fb";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(formatClock(time, { ms: false }), x + 8, height - 12);
  }

  ctx.strokeStyle = "#2c3442";
  ctx.lineWidth = 1;
  for (let index = 1; index < count; index += 1) {
    const x = index * cellWidth;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();
  }
}

function prepareCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(Math.round(canvas.clientWidth), 320);
  const height = Math.max(Math.round(canvas.clientHeight), FILMSTRIP_HEIGHT);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function buildSampleTimes(duration, count) {
  if (count <= 1 || duration <= 0) {
    return [0];
  }

  const safeEnd = Math.max(duration - 0.04, 0);
  return Array.from({ length: count }, (_, index) => {
    const progress = index / (count - 1);
    return progress * safeEnd;
  });
}

function drawCoverFrame(ctx, video, x, y, width, height) {
  const sourceWidth = video.videoWidth || 1;
  const sourceHeight = video.videoHeight || 1;
  const sourceRatio = sourceWidth / sourceHeight;
  const targetRatio = width / height;

  let sx = 0;
  let sy = 0;
  let sw = sourceWidth;
  let sh = sourceHeight;

  if (sourceRatio > targetRatio) {
    sw = sourceHeight * targetRatio;
    sx = (sourceWidth - sw) / 2;
  } else {
    sh = sourceWidth / targetRatio;
    sy = (sourceHeight - sh) / 2;
  }

  ctx.drawImage(video, sx, sy, sw, sh, x, y, width, height);
}

function once(target, eventName) {
  return new Promise((resolve, reject) => {
    const onResolve = () => {
      cleanup();
      resolve();
    };
    const onReject = () => {
      cleanup();
      reject(new Error(`Failed while waiting for ${eventName}`));
    };
    const cleanup = () => {
      target.removeEventListener(eventName, onResolve);
      target.removeEventListener("error", onReject);
    };

    target.addEventListener(eventName, onResolve, { once: true });
    target.addEventListener("error", onReject, { once: true });
  });
}

function seekVideo(video, time) {
  if (Math.abs(video.currentTime - time) < 0.0005) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Video seek failed"));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };

    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = time;
  });
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatClock(seconds, { ms = true } = {}) {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(seconds, 0) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const wholeSeconds = Math.floor(safeSeconds % 60);
  const millis = Math.round((safeSeconds - Math.floor(safeSeconds)) * 1000);

  if (!ms) {
    return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

playPauseButton.addEventListener("click", () => {
  if (state.isPlaying) {
    pauseBoth();
    return;
  }

  void playBoth();
});

pauseButton.addEventListener("click", () => {
  pauseBoth();
});

resetButton.addEventListener("click", () => {
  pauseBoth();
  seekBoth(0);
});

stepBackButton.addEventListener("click", () => {
  pauseBoth();
  seekBoth(getCurrentTime() - getFrameStepSeconds());
});

stepForwardButton.addEventListener("click", () => {
  pauseBoth();
  seekBoth(getCurrentTime() + getFrameStepSeconds());
});

timelineSlider.addEventListener("pointerdown", () => {
  state.resumeAfterScrub = state.isPlaying;
  state.wasDraggingTimeline = true;
  pauseBoth();
});

timelineSlider.addEventListener("input", () => {
  seekBoth(timelineValueToTime(timelineSlider.value));
});

timelineSlider.addEventListener("pointerup", () => {
  if (state.resumeAfterScrub) {
    void playBoth();
  }
  state.resumeAfterScrub = false;
  state.wasDraggingTimeline = false;
});

timelineSlider.addEventListener("change", () => {
  if (state.wasDraggingTimeline) {
    return;
  }
  seekBoth(timelineValueToTime(timelineSlider.value));
});

filmstripSurface.addEventListener("click", (event) => {
  if (!state.sharedDuration) {
    return;
  }

  const rect = state.slots.baseline.filmstrip.getBoundingClientRect();
  const progress = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  pauseBoth();
  seekBoth(progress * state.sharedDuration);
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) {
    return;
  }

  if (event.code === "Space") {
    event.preventDefault();
    if (state.isPlaying) {
      pauseBoth();
    } else {
      void playBoth();
    }
  } else if (event.code === "ArrowLeft") {
    event.preventDefault();
    pauseBoth();
    seekBoth(getCurrentTime() - getFrameStepSeconds());
  } else if (event.code === "ArrowRight") {
    event.preventDefault();
    pauseBoth();
    seekBoth(getCurrentTime() + getFrameStepSeconds());
  }
});

window.addEventListener("resize", () => {
  for (const slotId of SLOT_IDS) {
    const slot = state.slots[slotId];
    if (slot.loaded) {
      void renderFilmstrip(slot);
    } else {
      drawFilmstripPlaceholder(slot, "Upload a video");
    }
  }
  updateTimeUi(getCurrentTime());
});

window.addEventListener("beforeunload", () => {
  for (const slotId of SLOT_IDS) {
    revokeObjectUrl(state.slots[slotId]);
  }
});

updateSharedState();
