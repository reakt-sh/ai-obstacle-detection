# Configuration reference (English)

ReaktRailAi reads values from environment variables. Docker Compose loads
these automatically from `.env`. When running `python -m rail_ai` directly,
most values can also be given as CLI arguments; a CLI argument takes
precedence over the corresponding environment variable.

## Docker Compose parameters

These values configure the stack or map host paths to container paths.

| Variable | Default | Meaning |
|---|---:|---|
| `COMPOSE_PROFILES` | – | Exactly one of `cuda`, `rocm`, `jetpack5`, `jetpack6`, `jetpack7`. |
| `RESTART_POLICY` | `unless-stopped` | Docker restart policy. |
| `CAMERA_RTSP_URL` | Example URL | Camera RTSP or RTSPS URL. |
| `MODEL_PATH` | `./model/reaktrailai-yolo26m-seg.pt` | Host path to the bundled PT source model. Mounted read-only; override it to use another model. |
| `INT8_CALIBRATION_PATH` | `.` | Host directory containing the INT8 dataset YAML and its images. |
| `RESULTS_PATH` | `./results` | Host directory for optional status, latency and video files. |
| `WEBUI_AI_HOST` | Empty | AI host address as seen by the browser. If empty, use the host name from the browser URL. |
| `WEBUI_STATUS_INTERVAL_MS` | `1000` | Web interface polling interval in milliseconds. |
| `WEBUI_DEBUG` | `0` | `1` shows additional technical runtime details; `0` hides them. |

The four web interfaces listen on ports `8080` to `8083`. The API and WebRTC
ports are opened directly on the host because of `network_mode: host`.

## Input and runtime

| Environment variable | CLI | Default | Valid values / meaning |
|---|---|---:|---|
| `INPUT_URL` | `--source` | Required | RTSP or RTSPS URL. Compose takes it from `CAMERA_RTSP_URL`. |
| – | `--source-type` | `rtsp` | Currently only `rtsp`. |
| `RAIL_AI_PLATFORM` | `--platform` | `auto` | `auto`, `cpu`, `amd`, `rtx`, `jetson`; set automatically by the Compose profile. |
| `DEVICE` | `--device` | `auto` | `auto`, `cpu`, `rocm:0`, `cuda:0`. |
| `DEVICE_FALLBACK` | `--device-fallback` | `cpu` | Currently only `cpu`. Used when the GPU is unavailable. |
| `INPUT_DECODER` | `--input-decoder` | `auto` | `auto`, `software`, `vaapi`, `cuda`, `nvv4l2`. |
| `VAAPI_DEVICE` | `--vaapi-device` | `auto` | VAAPI device, for example `/dev/dri/renderD128`, or `auto`. |
| `MAX_FRAMES` | `--max-frames` | `0` | Maximum number of frames; `0` means unlimited. |
| `DEBUG_DISPLAY` | `--debug-display` / `--no-debug-display` | `0` | Local OpenCV debug window; normally disabled in a headless container. |

## Model and inference

| Environment variable | CLI | Default | Valid values / meaning |
|---|---|---:|---|
| `MODEL` | `--model` | `/app/model/reaktrailai-yolo26m-seg.pt` in the container | Path to the mounted PT model inside the runtime environment. Normally leave unchanged with Docker; set the host path through `MODEL_PATH`. |
| `MODEL_FORMAT` | `--model-format` | `pytorch` | `pytorch`, `onnx`, `tensorrt`; aliases `pt`, `torch`, `trt`, `engine` are normalized. |
| `MODEL_PRECISION` | `--model-precision` | `fp32` | `fp32`, `fp16`, `int8`. |
| `EXPORT_DIR` | `--export-dir` | `/models/exports` | Cache for exported ONNX or TensorRT models. |
| `IMGSZ` | `--imgsz` | `640` | Square inference image size, at least 32 pixels. |
| `RAIL_CONFIDENCE` | `--rail-confidence` | `0.25` | Minimum confidence for rail segmentation, range `0.0` to `1.0`. |
| `OBJECT_CONFIDENCE` | `--object-confidence` | `0.15` | Minimum confidence for object classes, range `0.0` to `1.0`. |
| `OBJECT_HOLD_SECONDS` | `--object-hold-seconds` | `1.0` | Minimum display time for detected objects, range `0.0` to `60.0` seconds. |
| `TRT_WORKSPACE` | `--export-workspace-gib` | `4` | TensorRT workspace in GiB; greater than 0. |
| `INT8_DATA` | `--int8-data` | `/calibration/data.yaml` in Compose | Container path to the YOLO dataset YAML for representative INT8 calibration images. |
| `INT8_CALIBRATION_SAMPLES` | `--int8-calibration-samples` | `256` | Maximum number of uniformly selected calibration images, at least 1. |
| `MIGRAPHX_DISABLE_REDUCE_FUSION` | – | `1` | ROCm/RDNA4 workaround; ignored outside the ROCm profile. |

Supported combinations:

| Platform | Model format / precision |
|---|---|
| CPU | `pytorch/fp32` |
| AMD | `pytorch/fp32`, `pytorch/fp16`, `onnx/fp16`, `onnx/int8` |
| NVIDIA RTX | `pytorch/fp32`, `pytorch/fp16`, `tensorrt/fp16`, `tensorrt/int8` |
| Jetson | `pytorch/fp32`, `pytorch/fp16`, `tensorrt/fp16`, `tensorrt/int8` |

INT8 always requires representative calibration data. NVIDIA uses TensorRT;
AMD uses ONNX Runtime with MIGraphX. Exported files are cached by model,
platform, precision, image size and runtime version.

## Stream output

| Environment variable | CLI | Default | Valid values / meaning |
|---|---|---:|---|
| `WEBRTC_ENABLED` | `--webrtc` / `--no-webrtc` | `1` | Enables H.264/WebRTC output. |
| `WEBRTC_ENCODER` | `--webrtc-encoder` | `auto` | `auto`, `software`, `vaapi`, `nvenc`, `nvv4l2`. |
| `WEBRTC_RTP_URL` | `--webrtc-rtp-url` | `rtp://127.0.0.1:5004?pkt_size=1200` | Internal RTP target. In the Compose stack, also adjust `mediamtx-webrtc.yml` if you change it. |
| `OUTPUT_FPS` | `--output-fps` | `25` | Target output frame rate; greater than 0. |
| `H264_BITRATE_KBPS` | `--h264-bitrate-kbps` | `30000` | Target H.264 bitrate in kbit/s; greater than 0. |
| `RTP_BUFFER_BYTES` | – | `4194304` | Requested UDP receive buffer for MediaMTX; limited by the host's `net.core.rmem_max`. |

In the Compose stack, the WebRTC endpoint is
`http://<AI-HOST>:8889/processed/whep`.

### Direct parameters of the WebUI container

Compose sets target ports and protocols to match the supplied stack. If the
WebUI image is run separately, its entrypoint also accepts:

| Variable | Default | Meaning |
|---|---:|---|
| `WEBUI_AI_API_PORT` | `5000` | Status API port on the AI host. |
| `WEBUI_AI_WHEP_PORT` | `8889` | WHEP/WebRTC port on the AI host. |
| `WEBUI_AI_API_SCHEME` | `http` | Status API protocol: `http`, or `https` behind a TLS proxy. |
| `WEBUI_AI_WHEP_SCHEME` | `http` | WHEP endpoint protocol: `http`, or `https` behind a TLS proxy. |

In the supplied Compose stack, the WHEP port is also fixed in
`docker/mediamtx-webrtc.yml`. Do not change it in only one place.

## API, diagnostics and recording

| Environment variable | CLI | Default | Valid values / meaning |
|---|---|---:|---|
| `API_ENABLED` | `--serve` / `--no-serve` | `1` | Enables the status and health API. Keep enabled in Compose, as the WebUI and health check use it. |
| `API_HOST` | `--host` | `0.0.0.0` | API bind address. |
| `API_PORT` | `--port` | `5000` | API port between 1 and 65535. |
| `LATENCY_CSV` | `--latency-csv` | Empty | Optional CSV output path, for example `/results/latency.csv` in the container. |
| `STATUS_JSON` | `--status-json` | Empty | Optional current JSON status path, for example `/results/status.json`. |
| `RECORD_DIR` | `--record-dir` | Empty | Optional output directory, for example `/results/recordings`. |
| `RECORD_ANNOTATED` | `--record-annotated` / `--no-record-annotated` | `0` | Writes annotated video; requires `RECORD_DIR`. |

The API also reads `RAIL_AI_RELEASE` from the image to report its version.
For direct use, `RAIL_AI_RELEASE_FILE` can select a path other than
`/app/RELEASE`. Neither is needed for a normal Compose start.

Boolean environment variables accept `1/0`, `true/false`, `yes/no` or
`on/off`.

## Direct start without Compose

The images internally start the same Python entrypoint. You can use it
directly in a suitable Python/GPU environment:

```bash
PYTHONPATH=src python3 -m rail_ai \
  --source rtsp://camera.example:8554/input \
  --model model/reaktrailai-yolo26m-seg.pt \
  --platform rtx \
  --device cuda:0 \
  --model-format pytorch \
  --model-precision fp16 \
  --imgsz 640
```

For the complete CLI help generated by the installed version, run
`PYTHONPATH=src python3 -m rail_ai --help`.

## Docker build arguments

All runtime Dockerfiles accept `RAIL_AI_RELEASE`; its default matches the
`RELEASE` file. The JetPack 7 image also has the coordinated arguments
`JETPACK_APT_RELEASE` and `TENSORRT_DEB_VERSION`. Change these two only
together with a tested JetPack base image, because PyTorch, CUDA and
TensorRT must be compatible.

`ORT_PROVIDER` and `PIP_CONSTRAINT` appear in the source code but are not
user parameters: the runtime manager sets `ORT_PROVIDER` according to the
detected hardware. `PIP_CONSTRAINT` comes from the NVIDIA base image and
protects its matched Python/CUDA stack during the image build.

# Konfigurationsreferenz (Deutsch)

ReaktRailAi liest die Werte aus Umgebungsvariablen. Docker Compose lädt sie
automatisch aus `.env`. Beim direkten Aufruf von `python -m rail_ai` können die
meisten Werte alternativ als CLI-Argumente angegeben werden; ein CLI-Argument
hat dabei Vorrang vor der gleichnamigen Umgebungsvariable.

## Docker-Compose-Parameter

Diese Werte konfigurieren den Stack oder übersetzen Hostpfade in Containerpfade.

| Variable | Standard | Bedeutung |
|---|---:|---|
| `COMPOSE_PROFILES` | – | Genau eines von `cuda`, `rocm`, `jetpack5`, `jetpack6`, `jetpack7`. |
| `RESTART_POLICY` | `unless-stopped` | Docker-Neustartrichtlinie. |
| `CAMERA_RTSP_URL` | Beispieladresse | RTSP- oder RTSPS-Adresse der Kamera. |
| `MODEL_PATH` | `./model/reaktrailai-yolo26m-seg.pt` | Hostpfad zum mitgelieferten PT-Ausgangsmodell. Wird schreibgeschützt eingebunden und kann zum Modellwechsel überschrieben werden. |
| `INT8_CALIBRATION_PATH` | `.` | Hostordner mit der INT8-Dataset-YAML und deren Bildern. |
| `RESULTS_PATH` | `./results` | Hostordner für optionale Status-, Latenz- und Videodateien. |
| `WEBUI_AI_HOST` | leer | Adresse des AI-Hosts aus Sicht des Browsers. Leer übernimmt den Hostnamen der Browser-URL. |
| `WEBUI_STATUS_INTERVAL_MS` | `1000` | Abfrageintervall der Weboberfläche in Millisekunden. |
| `WEBUI_DEBUG` | `0` | `1` zeigt zusätzliche technische Laufzeitdetails, `0` blendet sie aus. |

Die vier Weboberflächen hören auf den Ports `8080` bis `8083`. API-Port und
WebRTC-Port werden wegen `network_mode: host` direkt am Host geöffnet.

## Eingabe und Laufzeit

| Umgebungsvariable | CLI | Standard | Gültige Werte / Bedeutung |
|---|---|---:|---|
| `INPUT_URL` | `--source` | erforderlich | RTSP- oder RTSPS-Adresse. Compose setzt sie aus `CAMERA_RTSP_URL`. |
| – | `--source-type` | `rtsp` | Derzeit ausschließlich `rtsp`. |
| `RAIL_AI_PLATFORM` | `--platform` | `auto` | `auto`, `cpu`, `amd`, `rtx`, `jetson`; im Compose-Profil automatisch gesetzt. |
| `DEVICE` | `--device` | `auto` | `auto`, `cpu`, `rocm:0`, `cuda:0`. |
| `DEVICE_FALLBACK` | `--device-fallback` | `cpu` | Derzeit ausschließlich `cpu`. Bei nicht verfügbarer GPU wird darauf zurückgefallen. |
| `INPUT_DECODER` | `--input-decoder` | `auto` | `auto`, `software`, `vaapi`, `cuda`, `nvv4l2`. |
| `VAAPI_DEVICE` | `--vaapi-device` | `auto` | VAAPI-Gerät, zum Beispiel `/dev/dri/renderD128`, oder `auto`. |
| `MAX_FRAMES` | `--max-frames` | `0` | Maximale Framezahl; `0` bedeutet unbegrenzt. |
| `DEBUG_DISPLAY` | `--debug-display` / `--no-debug-display` | `0` | Lokales OpenCV-Debugfenster. Im Headless-Container normalerweise deaktiviert. |

## Modell und Inferenz

| Umgebungsvariable | CLI | Standard | Gültige Werte / Bedeutung |
|---|---|---:|---|
| `MODEL` | `--model` | `/app/model/reaktrailai-yolo26m-seg.pt` im Container | Pfad zum eingebundenen PT-Modell innerhalb der Laufzeitumgebung. Für Docker gewöhnlich nicht ändern; Hostpfad über `MODEL_PATH`. |
| `MODEL_FORMAT` | `--model-format` | `pytorch` | `pytorch`, `onnx`, `tensorrt`; Aliase `pt`, `torch`, `trt`, `engine` werden normalisiert. |
| `MODEL_PRECISION` | `--model-precision` | `fp32` | `fp32`, `fp16`, `int8`. |
| `EXPORT_DIR` | `--export-dir` | `/models/exports` | Cache für exportierte ONNX- oder TensorRT-Modelle. |
| `IMGSZ` | `--imgsz` | `640` | Quadratische Inferenzgröße, mindestens 32 Pixel. |
| `RAIL_CONFIDENCE` | `--rail-confidence` | `0.25` | Mindestkonfidenz der Gleissegmentierung, Bereich `0.0` bis `1.0`. |
| `OBJECT_CONFIDENCE` | `--object-confidence` | `0.15` | Mindestkonfidenz der Objektklassen, Bereich `0.0` bis `1.0`. |
| `OBJECT_HOLD_SECONDS` | `--object-hold-seconds` | `1.0` | Mindestanzeigedauer erkannter Objekte, Bereich `0.0` bis `60.0` Sekunden. |
| `TRT_WORKSPACE` | `--export-workspace-gib` | `4` | TensorRT-Workspace in GiB, größer als 0. |
| `INT8_DATA` | `--int8-data` | `/calibration/data.yaml` in Compose | Containerpfad zur YOLO-Dataset-YAML für repräsentative INT8-Kalibrierbilder. |
| `INT8_CALIBRATION_SAMPLES` | `--int8-calibration-samples` | `256` | Maximale Zahl gleichmäßig ausgewählter Kalibrierbilder, mindestens 1. |
| `MIGRAPHX_DISABLE_REDUCE_FUSION` | – | `1` | ROCm/RDNA4-Workaround; wird außerhalb des ROCm-Profils ignoriert. |

Unterstützte Kombinationen:

| Plattform | Modellformat / Präzision |
|---|---|
| CPU | `pytorch/fp32` |
| AMD | `pytorch/fp32`, `pytorch/fp16`, `onnx/fp16`, `onnx/int8` |
| NVIDIA RTX | `pytorch/fp32`, `pytorch/fp16`, `tensorrt/fp16`, `tensorrt/int8` |
| Jetson | `pytorch/fp32`, `pytorch/fp16`, `tensorrt/fp16`, `tensorrt/int8` |

INT8 benötigt immer repräsentative Kalibrierdaten. NVIDIA verwendet TensorRT,
AMD verwendet ONNX Runtime mit MIGraphX. Die exportierten Dateien werden anhand
von Modell, Plattform, Präzision, Bildgröße und Laufzeitversion gecacht.

## Streamausgabe

| Umgebungsvariable | CLI | Standard | Gültige Werte / Bedeutung |
|---|---|---:|---|
| `WEBRTC_ENABLED` | `--webrtc` / `--no-webrtc` | `1` | Aktiviert die H.264-/WebRTC-Ausgabe. |
| `WEBRTC_ENCODER` | `--webrtc-encoder` | `auto` | `auto`, `software`, `vaapi`, `nvenc`, `nvv4l2`. |
| `WEBRTC_RTP_URL` | `--webrtc-rtp-url` | `rtp://127.0.0.1:5004?pkt_size=1200` | Internes RTP-Ziel. Im Compose-Stack nicht ändern, ohne zugleich `mediamtx-webrtc.yml` anzupassen. |
| `OUTPUT_FPS` | `--output-fps` | `25` | Zielbildrate der Ausgabe, größer als 0. |
| `H264_BITRATE_KBPS` | `--h264-bitrate-kbps` | `30000` | H.264-Zielbitrate in kbit/s, größer als 0. |
| `RTP_BUFFER_BYTES` | – | `4194304` | Angeforderter UDP-Empfangspuffer für MediaMTX. Wird auf das Hostlimit `net.core.rmem_max` begrenzt. |

Der WebRTC-Endpunkt lautet im Compose-Stack
`http://<AI-HOST>:8889/processed/whep`.

### Direkte Parameter des WebUI-Containers

Compose setzt die Zielports und Protokolle passend zum mitgelieferten Stack.
Wird das WebUI-Image eigenständig gestartet, akzeptiert dessen Entrypoint außerdem:

| Variable | Standard | Bedeutung |
|---|---:|---|
| `WEBUI_AI_API_PORT` | `5000` | Status-API-Port des AI-Hosts. |
| `WEBUI_AI_WHEP_PORT` | `8889` | WHEP-/WebRTC-Port des AI-Hosts. |
| `WEBUI_AI_API_SCHEME` | `http` | Protokoll der Status-API, `http` oder bei vorgeschaltetem TLS-Proxy `https`. |
| `WEBUI_AI_WHEP_SCHEME` | `http` | Protokoll des WHEP-Endpunkts, `http` oder bei vorgeschaltetem TLS-Proxy `https`. |

Im mitgelieferten Compose-Stack ist der WHEP-Port zugleich in
`docker/mediamtx-webrtc.yml` festgelegt. Er darf dort nicht einseitig geändert
werden.

## API, Diagnose und Aufzeichnung

| Umgebungsvariable | CLI | Standard | Gültige Werte / Bedeutung |
|---|---|---:|---|
| `API_ENABLED` | `--serve` / `--no-serve` | `1` | Aktiviert Status- und Health-API. Im Compose-Stack eingeschaltet lassen, da WebUI und Healthcheck sie verwenden. |
| `API_HOST` | `--host` | `0.0.0.0` | Bind-Adresse der API. |
| `API_PORT` | `--port` | `5000` | API-Port zwischen 1 und 65535. |
| `LATENCY_CSV` | `--latency-csv` | leer | Optionaler CSV-Ausgabepfad, im Container beispielsweise `/results/latency.csv`. |
| `STATUS_JSON` | `--status-json` | leer | Optionaler Pfad für einen aktuellen JSON-Status, beispielsweise `/results/status.json`. |
| `RECORD_DIR` | `--record-dir` | leer | Optionaler Ausgabeordner, beispielsweise `/results/recordings`. |
| `RECORD_ANNOTATED` | `--record-annotated` / `--no-record-annotated` | `0` | Schreibt annotierte Videos; benötigt zwingend `RECORD_DIR`. |

Die API liest für ihre Versionsangabe außerdem `RAIL_AI_RELEASE` aus dem Image.
Mit `RAIL_AI_RELEASE_FILE` kann beim direkten Betrieb ein anderer Pfad als
`/app/RELEASE` angegeben werden. Beides ist für einen normalen Compose-Start
nicht erforderlich.

Boolesche Umgebungsvariablen akzeptieren `1/0`, `true/false`, `yes/no` oder
`on/off`.

## Direkter Start ohne Compose

Die Images starten intern denselben Python-Einstiegspunkt. In einer passend
eingerichteten Python-/GPU-Umgebung kann er auch direkt verwendet werden:

```bash
PYTHONPATH=src python3 -m rail_ai \
  --source rtsp://camera.example:8554/input \
  --model model/reaktrailai-yolo26m-seg.pt \
  --platform rtx \
  --device cuda:0 \
  --model-format pytorch \
  --model-precision fp16 \
  --imgsz 640
```

Die vollständige, von der installierten Version erzeugte CLI-Hilfe ist mit
`PYTHONPATH=src python3 -m rail_ai --help` verfügbar.

## Docker-Buildargumente

Alle Laufzeit-Dockerfiles kennen `RAIL_AI_RELEASE`; dessen Standard entspricht
der Datei `RELEASE`. Das JetPack-7-Image besitzt zusätzlich die fest aufeinander
abgestimmten Argumente `JETPACK_APT_RELEASE` und `TENSORRT_DEB_VERSION`. Diese
beiden Werte sollten nur gemeinsam mit einem geprüften JetPack-Basisimage geändert
werden, weil PyTorch, CUDA und TensorRT zueinander passen müssen.

`ORT_PROVIDER` und `PIP_CONSTRAINT` tauchen im Quellcode auf, sind jedoch keine
Benutzerparameter: Der Runtime-Manager setzt `ORT_PROVIDER` passend zur erkannten
Hardware selbst. `PIP_CONSTRAINT` stammt aus dem NVIDIA-Basisimage und schützt
beim Image-Build dessen abgestimmten Python-/CUDA-Stack.
