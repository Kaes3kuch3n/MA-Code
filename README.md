# Masterarbeit
Im Folgenden wird der Inhalt dieses Repositorys sowie die Verwendung der einzelnen Teile beschrieben.

## Notebooks
Der Ordner `notebooks` enthält Jupyter Notebooks, die jeweils verschiedene Teile des in der MA implementierten ersten Ansatzes beinhalten.
Am relevantesten ist das Notebook `transcribe_vocals`, welches die vollständige Transcription-Pipeline enthält.
Zum Verwenden müssen lediglich alle Zellen nacheinander ausgeführt werden.
In der Zelle mit dem Titel "Vocals Transcription Execution Cell" findet die eigentliche Transkription statt.
Dort kann auch die zu transkribierende Datei sowie der Ausgabepfad angegeben werden.
Die beiden darauf folgenden Zellen erlauben das Speichern des Ergebnisses als MIDI- und WAVE-Datei sowie die Visualisierung als Piano Roll.

## USDX Dataset
Der Ordner `usdx_dataset` enthält die Skripte, die zur Erstellung des Datensatzes benötigt werden.
Das Skript `prepare_data.py` nimmt den dort konfigurierten Ordnerpfad und wandelt alle in den Unterordnern liegenden USDX-Songfiles in MIDI-Dateien um.
Die neben den MIDI-Dateien liegenden MP3-Dateien werden in WAVE-Dateien konvertiert.
Der Eingabeordner sollte also folgende Struktur haben:
```
input_folder
|--- song1
|    |--- song1.mp3
|    |--- song1.txt
|--- song2
|    |--- song2.mp3
|    |--- song2.txt
|--- ...
```
Das Skript `create_usdx_dataset.py` nimmt den von `prepare_data.py` erzeugten Ordner mit den MIDI- und WAVE-Dateien entgegen und erstellt daraus den eigentlichen Datensatz in Form von TFRecord-Dateien.

## Weitere Dateien

### `optimize_transcription.py`
In diesem Skript ist die Optimierung der Transkription durch Anpassung der Parameter für die Audioaufbereitung und die Transkription mittels Basic Pitch implementiert.
Oben in der Datei lassen sich der Eingabepfad, die Anzahl der Iterationen und die Batchgröße pro Iteration anpassen.
Das Skript verwendet die Datei `transcription.py` zur Transkription des Gesangs.
Der Inhalt von `transcription.py` ist dabei quasi identisch mit dem Inhalt des Notebooks `transcribe_vocals`.
Da Notebooks aber nicht ohne weiteres aus einem Python-Skript verwendet werden können, wurde die Pipeline hier noch einmal gesammelt als Skript implementiert.

### `mt3-changes.patch`
Diese Patch-Datei enthält die Änderungen, die an der MT3-Bibliothek vorgenommen wurden.
Durch die Anwendung per `git apply` auf das MT3-Repository können diese Änderungen wiederhergestellt werden.
Die Änderungen werden in Form einer Patch-Datei bereitgestellt, damit einfach ersichtlich ist, welche Änderungen gegenüber der originalen Implementierung vorgenommen wurden.

### `main.py`
Dieses Skript enthält Wrapper-Methoden zur Bequemlichkeit für die Audio-Separation mit Demucs und die Transkription mit Basic Pitch.

## MT3
Die Verwendung von MT3 teilt sich auf zwei Schritte auf: das Training und die Verwendung des trainierten Modells.

### Training
Zum Trainieren eines neuen Modells muss zunächst die Trainingsumgebung eingerichtet werden:

#### Vorbereitung
- FFMPEG und fluidsynth (für TensorBoard logs) installieren
- Miniconda installieren: https://docs.conda.io/en/latest/miniconda.html
- Conda Environment erstellen und aktivieren: `conda create -n mt3 && conda activate mt3`
- Benötigte Repos klonen
    - https://github.com/google-research/t5x.git
    - https://github.com/magenta/mt3.git
- Patch anwenden: `cd mt3 && git apply <Pfad zu mt3-changes.patch>`
- Conda Environment einrichten
    - Conda install python==3.9
    - Cd mt3 && pip install .
    - Cd t5x && pip install 
        - Für TPU Support mit -e ‘.[tpu]’ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        - Für GPU Support Nvidia Rosetta Docker Image benutzen (siehe https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/t5x) und Nvidia Container Toolkit installieren, damit Docker die GPU mit CUDA nutzen kann

Für das Training mit einer Batch Size von 8 werden mindestens 12 GB VRAM benötigt.
Bei größeren Batch Sizes wird entsprechend mehr VRAM benötigt.

#### Task Caching
- (Eventuell) Conda Environment aktivieren: `conda activate mt3`
- (Eventuell) Pfad zum Datensatz USDX_VOCALS in `mt3/datasets.py` anpassen
- Cache generieren:
```shell
seqio_cache_tasks \
    --tasks=usdx_vocals_notes_ties_vb1_train,usdx_vocals_notes_ties_vb1_eval_train,usdx_vocals_notes_ties_vb1_validation \
    --output_cache_dir=/path/to/cache_dir \
    --module_import=mt3.tasks \
    --alsologtostderr
```

#### Training
- (Eventuell) Conda Environment aktivieren: `conda activate mt3`
- (Eventuell) Pfad zum Datensatz USDX_VOCALS in `mt3/datasets.py` anpassen
- Training starten:
```shell
python <t5x-dir>/t5x/train.py \
    --gin_file="<mt3-dir>/mt3/gin/model.gin" \
    --gin_file="<mt3-dir>/mt3/gin/train.gin" \
    --gin_file="<mt3-dir>/mt3/gin/vocals.gin" \
    --gin.MODEL_DIR=\"<model-output-dir>\" \
    --gin.USE_CACHED_TASKS=False  # falls noch keine cached tasks generiert wurden, sonst weglassen
```

### Transkription
Für die Transkription kann das Notebook `mt3/colab/music_transcription_with_transformers.ipynb` aus dem MT3-Repository verwendet werden.
Dort muss lediglich der Pfad zum trainierten Modell-Checkpoint angepasst werden (Code-Zelle mit dem Titel "Load Model", Variable `checkpoint_path`).
Am einfachsten ist die Verwendung des Notebooks in Google Colab, da dort automatisch eine entsprechende Laufzeitumgebung bereitgestellt wird, die alle Anforderungen erfüllt.