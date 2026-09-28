# ReaktRailAi Labeling Tool

**English** | [Deutsch](#deutsch)

This directory contains a standalone utility for creating and correcting YOLO
segmentation labels. It consists of two components:

1. `prepare_candidates.py` analyzes a video with an existing model, selects
   noteworthy frames, and creates pre-labels.
2. `server.py` starts a local browser-based editor for these frames.

## Prerequisites

- Python 3.11 or newer
- a CUDA or ROCm build of PyTorch compatible with the machine
- an Ultralytics YOLO segmentation model with a class named `rail`
- a local video from which candidates will be generated

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/runtime.txt
python3 -m pip install -r labeling-tool/requirements.txt
```

On a GPU machine, install the PyTorch build matching the installed CUDA or ROCm
driver first. Otherwise, `pip` may select an unsuitable default build.

## 1. Generate candidates from a video

The target directory must not already contain a completed `candidates.json`
data set. The following basic run selects all object tracks with no more than
four consecutive detections:

```bash
python3 labeling-tool/prepare_candidates.py \
  --video /path/to/input.mp4 \
  --model model/reaktrailai-yolo26m-seg.pt \
  --output-dir /path/to/labeling-data/run-01 \
  --device 0 \
  --selection-mode all-short \
  --short-track-max-length 4 \
  --cache-annotations
```

`--cache-annotations` transfers the masks calculated during the scan directly
to the editor and avoids a second model inference while building the data set.

Use `balanced` for a mixed selection of short object detections, unstable rail
masks, and, optionally, an important time range:

```bash
python3 labeling-tool/prepare_candidates.py \
  --video /path/to/input.mp4 \
  --model model/reaktrailai-yolo26m-seg.pt \
  --output-dir /path/to/labeling-data/run-02 \
  --device 0 \
  --selection-mode balanced \
  --switch-start 70.0 \
  --switch-end 95.0 \
  --switch-stride 10 \
  --cache-annotations
```

The range referred to as `switch` is simply a freely selectable priority time
range. If `--switch-start` and `--switch-end` are omitted, no fixed time range
is prioritized. Both arguments must be specified together. The option was
originally used to mark and focus on a switch area in a video, but it is also
useful for adding masks to otherwise unremarkable frames.

### Candidate generation parameters

| Parameter | Default | Description |
|---|---:|---|
| `--stage` | `all` | Run `scan`, `build`, or both steps with `all`. |
| `--video` | required | Input video. |
| `--model` | required | PT segmentation model used for scanning and pre-labeling. |
| `--output-dir` | required | New working directory for the manifest, frames, and labels. |
| `--device` | `0` | Ultralytics device, for example `0` or `cuda:0`. |
| `--imgsz` | `640` | Inference size in pixels. |
| `--precision` | `fp16` | `fp16` or `fp32`. |
| `--object-confidence` | `0.25` | Regular object confidence threshold. |
| `--mining-object-confidence` | `0.15` | Lower mining threshold; must not exceed `--object-confidence`. |
| `--rail-confidence` | `0.25` | Confidence threshold for rail masks. |
| `--annotation-confidence` | `0.15` | Minimum confidence threshold for pre-labels created in the editor. |
| `--switch-start` | not set | Start of an optional priority range in seconds. |
| `--switch-end` | not set | End of an optional priority range in seconds. |
| `--switch-stride` | `10` | Select a candidate from every nth frame in the priority range. |
| `--max-short-object-frames` | `120` | Maximum number of short-object frames in `balanced` mode. |
| `--max-rail-hard-frames` | `60` | Maximum number of noteworthy rail frames in `balanced` mode. |
| `--max-frames` | `0` | Stop scanning after this number of frames; `0` means unlimited. |
| `--jpeg-quality` | `95` | JPEG quality of working copies, internally limited to a range of 70 to 100. |
| `--polygon-epsilon` | `2.0` | Simplification of model polygons in pixels. |
| `--scan-batch` | `8` | Batch size used while scanning the video. |
| `--selection-mode` | `balanced` | Use `balanced` or only short tracks with `all-short`. |
| `--short-track-max-length` | `3` | Maximum number of consecutive detections in a short object track. |
| `--cache-annotations` | off | Cache scan masks for the build step. |
| `--dataset-name` | generic name | Display name shown in the editor. |

A large `--scan-batch` increases GPU memory usage. A “short detection” is
defined by raw model detections in directly consecutive frames, not by the
retention time used in the live pipeline.

## 2. Start the editor

For local-only use:

```bash
python3 labeling-tool/server.py \
  --dataset /path/to/labeling-data/run-01 \
  --host 127.0.0.1 \
  --port 8090
```

Then open `http://127.0.0.1:8090`.

## Editing

- Click a mask, or select it unambiguously from the mask list on the right.
- Drag vertices with the mouse.
- Double-click a polygon edge to add a vertex.
- Right-click a vertex to remove it.
- Select `Neue Maske` (New Mask), place the points, and press `Enter` to finish.
- Merge two masks of the same class with `Masken verbinden` (Merge Masks). Small
  gaps of up to two pixels are closed in the process.
- `Auswahl sperren` (Lock Selection) keeps the selected mask active even when
  polygons overlap.
- Copy a selected mask with `Ctrl+C` and paste it in the same position with
  `Ctrl+V`. The pasted copy is then selected.
- Delete a false positive with `Delete`. If the image is correct afterward,
  select `Geprüft & übernehmen` (Review & Accept).
- `Frame verwerfen` (Discard Frame) excludes the entire image from the export;
  it is not the same as a valid negative example.

Important keyboard shortcuts:

| Key | Action |
|---|---|
| `←` / `→` | previous / next frame |
| `N` | new mask |
| `M` | merge two masks |
| `Ctrl+C` / `Ctrl+V` | copy / paste mask |
| `Delete` | delete the selected mask |
| `Ctrl+Z` / `Ctrl+Y` | undo / redo |
| `R` | accept reviewed frame |
| `X` | discard the entire frame |
| `F` | fit image to view |
| `Ctrl+S` | save |
| Mouse wheel | zoom |
| Shift+drag or middle mouse button | pan the view |

## Saving and exporting

Changes are saved in the specified data set directory:

```text
run-01/
├── annotations/       # editable JSON annotations
├── images/            # selected frames
├── labels/            # confirmed YOLO segmentation labels
├── candidates.json    # manifest and review status
└── scan-*.json*       # scan metadata and detected tracks
```

The `Geprüfte Labels exportieren` (Export Reviewed Labels) button creates a ZIP
containing only confirmed, non-discarded frames, their YOLO labels, the JSON
annotations, and a `data.yaml` file. The ZIP is a working data set and is
intentionally kept outside version control.

Before making extensive edits, it is advisable to create a regular file-system
copy of the complete data set directory. The editor writes changes directly to
disk.

---

<a id="deutsch"></a>

# ReaktRailAi Labeling-Werkzeug

[English](#reaktrailai-labeling-tool) | **Deutsch**

Dieses Verzeichnis ist ein eigenständiges Hilfswerkzeug zum Erzeugen und
Korrigieren von YOLO-Segmentierungslabels. Es besteht aus zwei Teilen:

1. `prepare_candidates.py` untersucht ein Video mit einem vorhandenen Modell,
   wählt auffällige Frames aus und erstellt Vorlabels.
2. `server.py` startet einen lokalen Browsereditor für diese Frames.


## Voraussetzungen

- Python 3.11 oder neuer
- eine zum Rechner passende CUDA- oder ROCm-Version von PyTorch
- ein Ultralytics-YOLO-Segmentierungsmodell mit einer Klasse namens `rail`
- ein lokales Video, aus dem Kandidaten erzeugt werden sollen

Virtuelle Umgebung vom Repository-Stamm aus anlegen:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/runtime.txt
python3 -m pip install -r labeling-tool/requirements.txt
```

Bei einem GPU-Rechner sollte zuerst die zum installierten CUDA-/ROCm-Treiber
passende PyTorch-Ausgabe installiert werden. Andernfalls kann `pip` eine
ungeeignete Standardausgabe auswählen.

## 1. Kandidaten aus einem Video erzeugen

Der Zielordner darf noch keinen fertigen `candidates.json`-Datensatz enthalten.
Ein einfacher Lauf, der alle Objektspuren mit höchstens vier zusammenhängenden
Erkennungen auswählt:

```bash
python3 labeling-tool/prepare_candidates.py \
  --video /pfad/zum/eingang.mp4 \
  --model model/reaktrailai-yolo26m-seg.pt \
  --output-dir /pfad/zu/labeling-data/run-01 \
  --device 0 \
  --selection-mode all-short \
  --short-track-max-length 4 \
  --cache-annotations
```

`--cache-annotations` übernimmt exakt die bereits beim Scan berechneten Masken
in den Editor und vermeidet eine zweite Modellinferenz beim Aufbau des
Datensatzes.

Für eine gemischte Auswahl aus kurzen Objekterkennungen, instabilen Gleismasken
und optional einem wichtigen Zeitbereich wird `balanced` verwendet:

```bash
python3 labeling-tool/prepare_candidates.py \
  --video /pfad/zum/eingang.mp4 \
  --model model/reaktrailai-yolo26m-seg.pt \
  --output-dir /pfad/zu/labeling-data/run-02 \
  --device 0 \
  --selection-mode balanced \
  --switch-start 70.0 \
  --switch-end 95.0 \
  --switch-stride 10 \
  --cache-annotations
```

Der als `switch` bezeichnete Bereich ist lediglich ein frei wählbarer
Prioritätszeitraum. Ohne `--switch-start` und `--switch-end` wird kein fester
Zeitbereich bevorzugt. Beide Argumente müssen gemeinsam angegeben werden.
(ursprünglich verwendet, um einen weichenbereich im Video zu makierne um diesen zu fokusieren ist aber auch hilfreich um bilder mit masken zu versehen, welche eigentlich unauffälig waren)

### Parameter der Kandidatenerstellung

| Parameter | Standard | Bedeutung |
|---|---:|---|
| `--stage` | `all` | `scan`, `build` oder beide Schritte mit `all`. |
| `--video` | erforderlich | Eingabevideo. |
| `--model` | erforderlich | PT-Segmentierungsmodell für Scan und Vorlabels. |
| `--output-dir` | erforderlich | Neuer Arbeitsordner für Manifest, Frames und Labels. |
| `--device` | `0` | Ultralytics-Gerät, zum Beispiel `0` oder `cuda:0`. |
| `--imgsz` | `640` | Inferenzgröße in Pixeln. |
| `--precision` | `fp16` | `fp16` oder `fp32`. |
| `--object-confidence` | `0.25` | Reguläre Objektschwelle. |
| `--mining-object-confidence` | `0.15` | Niedrigere Suchschwelle; darf nicht größer als `--object-confidence` sein. |
| `--rail-confidence` | `0.25` | Schwelle für Gleismasken. |
| `--annotation-confidence` | `0.15` | Mindestschwelle der im Editor angelegten Vorlabels. |
| `--switch-start` | nicht gesetzt | Beginn eines optionalen Prioritätsbereichs in Sekunden. |
| `--switch-end` | nicht gesetzt | Ende des optionalen Prioritätsbereichs in Sekunden. |
| `--switch-stride` | `10` | Aus jedem n-ten Frame des Prioritätsbereichs einen Kandidaten nehmen. |
| `--max-short-object-frames` | `120` | Obergrenze kurzer Objektframes im Modus `balanced`. |
| `--max-rail-hard-frames` | `60` | Obergrenze auffälliger Gleisframes im Modus `balanced`. |
| `--max-frames` | `0` | Scan nach dieser Framezahl beenden; `0` ist unbegrenzt. |
| `--jpeg-quality` | `95` | JPEG-Qualität der Arbeitskopien, intern auf 70 bis 100 begrenzt. |
| `--polygon-epsilon` | `2.0` | Vereinfachung der Modellpolygone in Pixeln. |
| `--scan-batch` | `8` | Batchgröße beim Videoscan. |
| `--selection-mode` | `balanced` | `balanced` oder ausschließlich kurze Spuren mit `all-short`. |
| `--short-track-max-length` | `3` | Maximale zusammenhängende Trefferzahl einer kurzen Objektspur. |
| `--cache-annotations` | aus | Masken des Scans für den Build-Schritt zwischenspeichern. |
| `--dataset-name` | generischer Name | Anzeigename im Editor. |

Ein großer `--scan-batch` erhöht den GPU-Speicherbedarf. Die Definition einer
„kurzen Erkennung“ bezieht sich auf rohe, unmittelbar aufeinanderfolgende
Modelltreffer und nicht auf die Haltezeit der Live-Pipeline.

## 2. Editor starten

Für eine ausschließlich lokale Verwendung:

```bash
python3 labeling-tool/server.py \
  --dataset /pfad/zu/labeling-data/run-01 \
  --host 127.0.0.1 \
  --port 8090
```

Danach `http://127.0.0.1:8090` öffnen.

## Bearbeitung

- Eine Maske anklicken oder eindeutig in der rechten Maskenliste auswählen.
- Eckpunkte mit der Maus ziehen.
- Doppelklick auf eine Polygonkante fügt einen Eckpunkt hinzu.
- Rechtsklick auf einen Eckpunkt entfernt ihn.
- `Neue Maske` wählen, Punkte setzen und mit `Enter` abschließen.
- Zwei Masken derselben Klasse mit `Masken verbinden` vereinigen. Kleine Spalten
  bis zwei Pixel werden dabei geschlossen.
- Mit `Auswahl sperren` bleibt die gewählte Maske auch bei überlappenden Polygonen
  aktiv.
- Eine gewählte Maske mit `Strg+C` kopieren und mit `Strg+V` deckungsgleich
  einfügen. Danach ist die Kopie ausgewählt.
- Einen Fehlalarm mit `Entf` löschen. Wenn das Bild danach korrekt ist,
  `Geprüft & übernehmen` wählen.
- `Frame verwerfen` schließt das gesamte Bild aus dem Export aus; es ist nicht
  gleichbedeutend mit einem gültigen Negativbeispiel.

Wichtige Kurzbefehle:

| Taste | Aktion |
|---|---|
| `←` / `→` | vorheriger / nächster Frame |
| `N` | neue Maske |
| `M` | zwei Masken verbinden |
| `Strg+C` / `Strg+V` | Maske kopieren / einfügen |
| `Entf` | ausgewählte Maske löschen |
| `Strg+Z` / `Strg+Y` | rückgängig / wiederholen |
| `R` | geprüft übernehmen |
| `X` | gesamten Frame verwerfen |
| `F` | Bild einpassen |
| `Strg+S` | speichern |
| Mausrad | zoomen |
| Umschalt+Ziehen oder mittlere Maustaste | Ansicht verschieben |

## Speichern und Exportieren

Änderungen werden im angegebenen Dataset-Ordner gespeichert:

```text
run-01/
├── annotations/       # editierbare JSON-Annotationen
├── images/            # ausgewählte Frames
├── labels/            # bestätigte YOLO-Segmentierungslabels
├── candidates.json    # Manifest und Prüfstatus
└── scan-*.json*       # Scanmetadaten und erkannte Spuren
```

Der Button `Geprüfte Labels exportieren` erzeugt ein ZIP mit ausschließlich
bestätigten und nicht verworfenen Frames, den YOLO-Labels, den JSON-Annotationen
und einer `data.yaml`. Das ZIP ist ein Arbeitsdatensatz und bleibt bewusst
außerhalb der Versionsverwaltung.

Vor größeren Bearbeitungen empfiehlt sich eine normale Dateisystemkopie des
kompletten Dataset-Ordners. Der Editor schreibt Änderungen direkt auf die Platte.
