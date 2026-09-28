(() => {
  "use strict";

  const config = window.RAIL_AI_CONFIG || {};
  const themesByPort = {
    "8080": { id: "dark", color: "#050608" },
    "8081": { id: "light", color: "#eef2f5" },
    "8082": { id: "reakt", color: "#1c3647" },
    "8083": { id: "sunlight", color: "#ffffff" }
  };
  const configuredTheme = String(config.theme || "").trim().toLowerCase();
  const portTheme = themesByPort[window.location.port] || themesByPort["8080"];
  const configuredEntry = [{ id: "dark", color: "#050608" }, ...Object.values(themesByPort)]
    .find((entry) => entry.id === configuredTheme);
  const activeTheme = configuredEntry || portTheme;
  document.documentElement.dataset.theme = activeTheme.id;
  document.querySelector('meta[name="theme-color"]').setAttribute("content", activeTheme.color);

  const host = config.host || window.location.hostname || "127.0.0.1";
  const apiBase = `${config.apiScheme || "http"}://${host}:${config.apiPort || 5000}`;
  const whepUrl = `${config.whepScheme || "http"}://${host}:${config.whepPort || 8889}/processed/whep`;
  const statusIntervalMs = Math.max(250, config.statusIntervalMs || 1000);
  const metricWindowMs = 10000;
  const debug = ["1", "true", "yes", "on"].includes(String(config.debug || "").trim().toLowerCase());

  const video = document.getElementById("stream");
  const streamStatus = document.getElementById("streamStatus");
  const infoButton = document.getElementById("infoButton");
  const closeButton = document.getElementById("closeButton");
  const infoPanel = document.getElementById("infoPanel");
  const technicalInfo = document.getElementById("technicalInfo");
  let peerConnection = null;
  let sessionUrl = null;
  let reconnectTimer = null;
  let metricSamples = [];

  function setConnection(state, text) {
    document.documentElement.dataset.streamState = state;
    streamStatus.textContent = text;
  }

  function waitForIceGathering(pc) {
    if (pc.iceGatheringState === "complete") {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const timer = window.setTimeout(() => {
        pc.removeEventListener("icegatheringstatechange", listener);
        resolve();
      }, 2000);
      const listener = () => {
        if (pc.iceGatheringState === "complete") {
          window.clearTimeout(timer);
          pc.removeEventListener("icegatheringstatechange", listener);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", listener);
    });
  }

  async function closeSession() {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
    const oldSession = sessionUrl;
    sessionUrl = null;
    if (peerConnection) {
      peerConnection.ontrack = null;
      peerConnection.onconnectionstatechange = null;
      peerConnection.close();
      peerConnection = null;
    }
    video.srcObject = null;
    if (oldSession) {
      try {
        await fetch(oldSession, { method: "DELETE" });
      } catch (_error) {
        // The server can already have removed the WHEP session.
      }
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer) {
      return;
    }
    setConnection("error", "Nicht verfügbar – neuer Versuch …");
    reconnectTimer = window.setTimeout(async () => {
      reconnectTimer = null;
      await connectWebRTC();
    }, 2000);
  }

  async function connectWebRTC() {
    await closeSession();
    setConnection("waiting", "Verbindung wird aufgebaut …");

    const pc = new RTCPeerConnection();
    peerConnection = pc;
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.ontrack = (event) => {
      video.srcObject = event.streams[0];
      const markFirstFrame = () => {
        if (pc === peerConnection && video.videoWidth > 0) {
          setConnection("online", "Live");
        }
      };
      if (typeof video.requestVideoFrameCallback === "function") {
        video.requestVideoFrameCallback(markFirstFrame);
      } else {
        video.addEventListener("playing", markFirstFrame, { once: true });
      }
      video.play().catch(() => {});
    };
    pc.onconnectionstatechange = () => {
      if (pc !== peerConnection) {
        return;
      }
      if (pc.connectionState === "connected") {
        setConnection("waiting", "Verbunden – warte auf Videobild …");
      } else if (["failed", "closed", "disconnected"].includes(pc.connectionState)) {
        scheduleReconnect();
      }
    };

    try {
      await pc.setLocalDescription(await pc.createOffer());
      await waitForIceGathering(pc);
      const response = await fetch(whepUrl, {
        method: "POST",
        headers: {
          "Accept": "application/sdp",
          "Content-Type": "application/sdp"
        },
        body: pc.localDescription.sdp
      });
      if (!response.ok) {
        throw new Error(`WHEP antwortet mit HTTP ${response.status}`);
      }
      const location = response.headers.get("Location");
      if (location) {
        sessionUrl = new URL(location, whepUrl).toString();
      }
      await pc.setRemoteDescription({ type: "answer", sdp: await response.text() });
    } catch (error) {
      console.error(error);
      scheduleReconnect();
    }
  }

  function setText(id, value) {
    document.getElementById(id).textContent = value ?? "–";
  }

  function smoothedMetrics(status) {
    const timestamp = performance.now();
    const latency = status.latency?.pipeline_ms;
    metricSamples.push({
      timestamp,
      fps: Number(status.fps || 0),
      latency: latency == null ? null : Number(latency)
    });
    metricSamples = metricSamples.filter((sample) => timestamp - sample.timestamp <= metricWindowMs);

    const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
    const fpsValues = metricSamples.map((sample) => sample.fps).filter(Number.isFinite);
    const latencyValues = metricSamples.map((sample) => sample.latency).filter(Number.isFinite);
    return {
      fps: fpsValues.length ? mean(fpsValues) : null,
      latency: latencyValues.length ? mean(latencyValues) : null
    };
  }

  async function loadInfo() {
    try {
      const response = await fetch(`${apiBase}/api/v1/info`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Info API HTTP ${response.status}`);
      }
      const info = await response.json();
      const models = Array.isArray(info.models) ? info.models : [];
      document.getElementById("models").replaceChildren(
        ...models.map((model) => {
          const card = document.createElement("article");
          card.className = "model-card";
          const name = document.createElement("strong");
          name.textContent = model.name;
          const details = document.createElement("span");
          details.textContent = `${model.role} · ${model.technology}`;
          card.append(name, details);
          return card;
        })
      );
      setText("release", info.release);
    } catch (error) {
      console.error(error);
      document.getElementById("models").textContent = "Modellinformationen nicht erreichbar.";
    }
  }

  async function loadStatus() {
    try {
      const response = await fetch(`${apiBase}/api/v1/status`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Status API HTTP ${response.status}`);
      }
      const status = await response.json();
      const metrics = smoothedMetrics(status);
      setText("fps", metrics.fps == null ? "–" : `${metrics.fps.toFixed(1)} FPS`);
      setText("latency", metrics.latency == null ? "–" : `${metrics.latency.toFixed(1)} ms`);
      if (debug) {
        setText("platform", status.platform);
        setText("device", status.effective_device);
        setText("precision", `${status.effective_format}/${status.effective_precision}`);
        setText("provider", status.provider_fallback_reason ? `${status.provider} (${status.provider_fallback_reason})` : status.provider);
        setText("decoder", status.decoder_fallback_reason ? `${status.input_decoder} (${status.decoder_fallback_reason})` : status.input_decoder);
        setText("encoder", status.encoder_fallback_reason ? `${status.webrtc_encoder} (${status.encoder_fallback_reason})` : status.webrtc_encoder);
      }
    } catch (error) {
      console.error(error);
      metricSamples = [];
      setText("fps", "API nicht erreichbar");
      setText("latency", "–");
    } finally {
      window.setTimeout(loadStatus, statusIntervalMs);
    }
  }

  function setPanel(open, restoreFocus = false) {
    infoPanel.classList.toggle("info-panel--open", open);
    infoPanel.setAttribute("aria-hidden", String(!open));
    infoPanel.inert = !open;
    infoButton.setAttribute("aria-expanded", String(open));
    if (open) {
      closeButton.focus();
    } else if (restoreFocus) {
      infoButton.focus();
    }
  }

  infoButton.addEventListener("click", () => setPanel(!infoPanel.classList.contains("info-panel--open")));
  closeButton.addEventListener("click", () => setPanel(false, true));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && infoPanel.classList.contains("info-panel--open")) {
      setPanel(false, true);
    }
  });
  window.addEventListener("beforeunload", () => {
    if (sessionUrl) {
      fetch(sessionUrl, { method: "DELETE", keepalive: true }).catch(() => {});
    }
  });

  technicalInfo.hidden = !debug;
  if (debug) {
    loadInfo();
  }
  loadStatus();
  connectWebRTC();
})();
