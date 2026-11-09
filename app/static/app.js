"use strict";

(function () {
  const el = (id) => document.getElementById(id);

  async function apiRequest(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    return res.json();
  }

  async function patchConfig(patch) {
    const result = await apiRequest("/api/config", {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    if (result.success && result.data) {
      syncControlsFromConfig(result.data);
      await refreshTelemetryAndEvents();
    } else {
      const cfgRes = await apiRequest("/api/config");
      if (cfgRes.success && cfgRes.data) {
        syncControlsFromConfig(cfgRes.data);
      }
    }
    return result;
  }

  function syncControlsFromConfig(cfg) {
    el("ctrl-source").value = cfg.source;
    el("ctrl-camera-index").value = cfg.camera_index;
    if (el("ctrl-mirror")) {
      el("ctrl-mirror").checked = Boolean(cfg.mirror);
    }
    el("ctrl-scenario").value = cfg.simulator.scenario;
    el("ctrl-width").value = cfg.width;
    el("ctrl-height").value = cfg.height;

    const presetKey = `${cfg.width}x${cfg.height}`;
    const presetSelect = el("ctrl-res-preset");
    if ([...presetSelect.options].some((o) => o.value === presetKey)) {
      presetSelect.value = presetKey;
    }

    el("ctrl-fps").value = cfg.target_fps;
    el("lbl-fps").textContent = `${cfg.target_fps} FPS`;
    el("kpi-resolution").textContent = `${cfg.width} x ${cfg.height}`;
    el("stream-res-label").textContent = `${cfg.width}x${cfg.height} @ ${cfg.target_fps} FPS`;
    document.querySelectorAll(".chip[data-fps]").forEach((btn) => {
      btn.classList.toggle("active", Number(btn.dataset.fps) === cfg.target_fps);
    });

    el("ctrl-sim-speed").value = cfg.simulator.speed_multiplier;
    el("lbl-sim-speed").textContent = `${cfg.simulator.speed_multiplier.toFixed(1)}x`;

    el("ctrl-palette").value = cfg.thermal.palette;
    el("ctrl-temp-min").value = cfg.thermal.min_temp_c;
    el("ctrl-temp-max").value = cfg.thermal.max_temp_c;
    el("ctrl-temp-alarm").value = cfg.thermal.overheat_threshold_c;
    el("ctrl-kinetic-heat").value = cfg.thermal.kinetic_heat_gain;
    el("lbl-kinetic").textContent = `${Math.round(cfg.thermal.kinetic_heat_gain * 100)}%`;
    el("ctrl-show-hotspots").checked = cfg.thermal.show_hotspots;
    el("ctrl-show-isotherms").checked = cfg.thermal.show_isotherms;
    el("ctrl-show-scale").checked = cfg.thermal.show_scale_bar;

    el("ctrl-algorithm").value = cfg.motion.algorithm;
    el("ctrl-sensitivity").value = cfg.motion.sensitivity;
    el("lbl-sensitivity").textContent = `${cfg.motion.sensitivity}%`;
    el("ctrl-min-area").value = cfg.motion.min_area;
    el("ctrl-alarm-pct").value = cfg.motion.alarm_motion_pct;
    el("ctrl-motion-boxes").checked = cfg.motion.show_boxes;
    el("ctrl-motion-trails").checked = cfg.motion.show_trails;
    el("ctrl-roi-enabled").checked = cfg.roi.enabled;

    document.querySelectorAll(".mode-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.mode === cfg.vision_mode);
    });
  }

  function renderTelemetry(tel) {
    el("stream-res-label").textContent = `${tel.width}x${tel.height} @ ${tel.actual_fps.toFixed(1)} FPS`;
    el("kpi-fps").textContent = `${tel.actual_fps.toFixed(1)} / ${tel.target_fps} FPS`;
    el("kpi-latency").textContent = `Processamento: ${tel.processing_ms.toFixed(1)} ms`;
    el("kpi-motion").textContent = `${tel.motion_percentage.toFixed(1)}%`;
    el("kpi-objects").textContent = `${tel.object_count} objeto(s) em movimento`;
    el("kpi-temp-max").textContent = `${tel.max_temp_c.toFixed(1)} °C`;
    el("kpi-temp-range").textContent = `Mín: ${tel.min_temp_c.toFixed(1)}°C | Méd: ${tel.avg_temp_c.toFixed(1)}°C`;
    el("kpi-resolution").textContent = `${tel.width} x ${tel.height}`;
    el("kpi-coords").textContent = `Hotspot: (${tel.hotspot_x}, ${tel.hotspot_y})`;

    const srcBadge = el("badge-source");
    if (tel.hardware_fallback) {
      srcBadge.textContent = "FONTE: FALLBACK (SEM CÂMERA)";
    } else {
      srcBadge.textContent = tel.source_active === "camera" ? "FONTE: CÂMERA PESSOAL" : "FONTE: MODO TESTE";
    }

    const alertBadge = el("badge-alert");
    alertBadge.textContent = `STATUS: ${tel.alert_level}`;
    alertBadge.className = "badge " + (
      tel.alert_level === "CRITICO"
        ? "badge-critical"
        : tel.alert_level === "ATENCAO"
        ? "badge-warn"
        : "badge-normal"
    );
  }

  function renderEvents(items) {
    const container = el("events-list");
    if (!items || items.length === 0) {
      container.innerHTML = `<div class="event-item"><span>Nenhum evento registrado.</span></div>`;
      return;
    }
    container.innerHTML = items
      .slice(0, 15)
      .map((ev) => {
        const timeStr = new Date(ev.timestamp_iso).toLocaleTimeString("pt-BR");
        return `
          <div class="event-item ${ev.severity}">
            <span><strong>[${ev.category}]</strong> ${ev.message}</span>
            <span class="event-time">${timeStr}</span>
          </div>
        `;
      })
      .join("");
  }

  function bindEvents() {
    document.querySelectorAll(".mode-tab").forEach((tab) => {
      tab.addEventListener("click", () => patchConfig({ vision_mode: tab.dataset.mode }));
    });

    document.querySelectorAll("[data-anomaly]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await apiRequest("/api/anomaly", {
          method: "POST",
          body: JSON.stringify({ kind: btn.dataset.anomaly, duration_frames: 50 }),
        });
        refreshTelemetryAndEvents();
      });
    });

    document.querySelectorAll("[data-thermal-profile]").forEach((btn) => {
      btn.addEventListener("click", () =>
        patchConfig({ thermal_profile: btn.dataset.thermalProfile })
      );
    });

    if (el("btn-reconnect-cam")) {
      el("btn-reconnect-cam").addEventListener("click", async () => {
        await apiRequest("/api/camera/reconnect", { method: "POST" });
        const cfgRes = await apiRequest("/api/config");
        if (cfgRes.success && cfgRes.data) {
          syncControlsFromConfig(cfgRes.data);
        }
        refreshTelemetryAndEvents();
      });
    }

    if (el("ctrl-mirror")) {
      el("ctrl-mirror").addEventListener("change", (e) =>
        patchConfig({ mirror: e.target.checked })
      );
    }

    el("btn-recalibrate").addEventListener("click", async () => {
      await apiRequest("/api/reset", { method: "POST" });
      refreshTelemetryAndEvents();
    });

    el("btn-clear-events").addEventListener("click", async () => {
      await apiRequest("/api/events", { method: "DELETE" });
      refreshTelemetryAndEvents();
    });

    el("ctrl-source").addEventListener("change", (e) => patchConfig({ source: e.target.value }));
    el("ctrl-camera-index").addEventListener("change", (e) =>
      patchConfig({ camera_index: Number(e.target.value) })
    );
    el("ctrl-scenario").addEventListener("change", (e) =>
      patchConfig({ simulator: { scenario: e.target.value } })
    );

    el("ctrl-res-preset").addEventListener("change", (e) =>
      patchConfig({ resolution_preset: e.target.value })
    );
    el("ctrl-width").addEventListener("change", (e) =>
      patchConfig({ width: Number(e.target.value), height: Number(el("ctrl-height").value) })
    );
    el("ctrl-height").addEventListener("change", (e) =>
      patchConfig({ width: Number(el("ctrl-width").value), height: Number(e.target.value) })
    );

    el("ctrl-fps").addEventListener("input", (e) => {
      el("lbl-fps").textContent = `${e.target.value} FPS`;
    });
    el("ctrl-fps").addEventListener("change", (e) =>
      patchConfig({ target_fps: Number(e.target.value) })
    );

    document.querySelectorAll(".chip[data-fps]").forEach((chip) => {
      chip.addEventListener("click", () => patchConfig({ target_fps: Number(chip.dataset.fps) }));
    });

    el("ctrl-sim-speed").addEventListener("input", (e) => {
      el("lbl-sim-speed").textContent = `${Number(e.target.value).toFixed(1)}x`;
    });
    el("ctrl-sim-speed").addEventListener("change", (e) =>
      patchConfig({ simulator: { speed_multiplier: Number(e.target.value) } })
    );

    el("ctrl-palette").addEventListener("change", (e) =>
      patchConfig({ thermal: { palette: e.target.value } })
    );
    const updateTempScale = (e) => {
      const minC = Number(el("ctrl-temp-min").value);
      const maxC = Number(el("ctrl-temp-max").value);
      let alarmC = Number(el("ctrl-temp-alarm").value);
      if (
        e &&
        e.target &&
        e.target.id !== "ctrl-temp-alarm" &&
        minC < maxC
      ) {
        alarmC = Math.min(maxC, Math.max(minC, alarmC));
        el("ctrl-temp-alarm").value = String(alarmC);
      }
      return patchConfig({
        thermal: {
          min_temp_c: minC,
          max_temp_c: maxC,
          overheat_threshold_c: alarmC,
        },
      });
    };
    el("ctrl-temp-min").addEventListener("change", updateTempScale);
    el("ctrl-temp-max").addEventListener("change", updateTempScale);
    el("ctrl-temp-alarm").addEventListener("change", updateTempScale);

    el("ctrl-kinetic-heat").addEventListener("input", (e) => {
      el("lbl-kinetic").textContent = `${Math.round(Number(e.target.value) * 100)}%`;
    });
    el("ctrl-kinetic-heat").addEventListener("change", (e) =>
      patchConfig({ thermal: { kinetic_heat_gain: Number(e.target.value) } })
    );

    el("ctrl-show-hotspots").addEventListener("change", (e) =>
      patchConfig({ thermal: { show_hotspots: e.target.checked } })
    );
    el("ctrl-show-isotherms").addEventListener("change", (e) =>
      patchConfig({ thermal: { show_isotherms: e.target.checked } })
    );
    el("ctrl-show-scale").addEventListener("change", (e) =>
      patchConfig({ thermal: { show_scale_bar: e.target.checked } })
    );

    el("ctrl-algorithm").addEventListener("change", (e) =>
      patchConfig({ motion: { algorithm: e.target.value } })
    );
    el("ctrl-sensitivity").addEventListener("input", (e) => {
      el("lbl-sensitivity").textContent = `${e.target.value}%`;
    });
    el("ctrl-sensitivity").addEventListener("change", (e) =>
      patchConfig({ motion: { sensitivity: Number(e.target.value) } })
    );
    el("ctrl-min-area").addEventListener("change", (e) =>
      patchConfig({ motion: { min_area: Number(e.target.value) } })
    );
    el("ctrl-alarm-pct").addEventListener("change", (e) =>
      patchConfig({ motion: { alarm_motion_pct: Number(e.target.value) } })
    );
    el("ctrl-motion-boxes").addEventListener("change", (e) =>
      patchConfig({ motion: { show_boxes: e.target.checked } })
    );
    el("ctrl-motion-trails").addEventListener("change", (e) =>
      patchConfig({ motion: { show_trails: e.target.checked } })
    );
    el("ctrl-roi-enabled").addEventListener("change", (e) =>
      patchConfig({ roi: { enabled: e.target.checked } })
    );
  }

  async function refreshTelemetryAndEvents() {
    try {
      const telRes = await apiRequest("/api/telemetry");
      if (telRes.success && telRes.data) {
        renderTelemetry(telRes.data);
      }
      const evRes = await apiRequest("/api/events?limit=15");
      if (evRes.success && evRes.data) {
        renderEvents(evRes.data.items);
      }
    } catch (_err) {
      // Silencia falha transitória de polling
    }
  }

  async function init() {
    bindEvents();
    const cfgRes = await apiRequest("/api/config");
    if (cfgRes.success && cfgRes.data) {
      syncControlsFromConfig(cfgRes.data);
    }
    await refreshTelemetryAndEvents();
    setInterval(refreshTelemetryAndEvents, 650);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
