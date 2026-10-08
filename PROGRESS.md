# PROGRESS

One entry per session, newest first. Structured fields in English; the reflection in Spanish.

---

## 2026-10-08 · Session 1 · From Colab notebooks to a pipeline

**Objective.** Turn the three Colab notebooks into six numbered scripts with one command, run them end to end on a tiny dataset, and record what the Colab runs actually measured.

**Prediction.** The scripts run on the test dataset in under 2 minutes on CPU, and the patient split rebuilt from public label files reproduces Colab's 27 706 / 5 541 / 5 540 images exactly.

**Test dataset.** `python tests/make_test_data.py` → 80 synthetic fundus-like images (40 in the APTOS layout, 20 patients × 2 eyes in the EyePACS layout), 1.2 MB in `tests/data/raw`.

**Result.**

| Item | Predicted | Outcome |
|---|---|---|
| `DRY_RUN=1 bash run.sh` | prints six stages | done: each command printed and logged, nothing ran |
| `bash run.sh` on `tests/` | < 2 min | 40 s on CPU (torch 2.13.0, timm 1.0.30); two clean runs gave identical losses and AUROCs |
| Stage 05 resume | resumes after an interruption | weights deleted, rerun resumed at epoch 3; with `--epochs 3` it refused the checkpoint |
| `model_v1.pt` in the new code | loads | `load_state_dict(strict=True)`: all keys matched |
| Split rebuilt from public labels (`notebooks/check_split.py`) | 27 706 / 5 541 / 5 540 | 27 705 / 5 541 / 5 541, referable 22% vs 21% in train: **not reproduced** |
| EyePACS copy `cristbalmontoyabao/atros-dataset` | downloadable | `403 Client Error: Forbidden for url: https://api.kaggle.com/v1/datasets.DatasetApiService/GetDatasetMetadata` |
| Colab results extracted (`notebooks/extract_runs.py`) | three runs | model_v1 and model_v3 kept; model_v2 discarded (see Session 0) |

Errors met during the session, verbatim, each fixed in its own commit:

- `bash: line 1: unzip: command not found` while extracting `trainLabels.csv.zip` in WSL → `unzip` added to `environment.yml` (stage 01 needs it for APTOS).
- `specificity  0.0% (95% CI -0.0%-43.5%)` on the test set → Wilson bounds clipped to [0, 1]. The notebook had the same code.
- Evaluation figure with the y tick labels cut and the axis label over the matrix → constrained layout, label placed inside the axes.

**Costliest error.** Di por hecho que la copia de EyePACS de Cristóbal era pública y que sus etiquetas eran las del concurso de 2015. Las dos cosas fallaron: Kaggle devuelve 403 con mi cuenta y la división reconstruida con `trainLabels.csv` se desvía en una imagen por conjunto. Me costó unos 20 minutos entre probar descargas, montar `check_split.py` y entender que el desvío viene de las etiquetas y no del código, porque `StratifiedGroupKFold` solo depende del conjunto de pares (paciente, grado). Lección: un dataset que solo existe en la cuenta de un compañero no es reproducible; antes de escribir el README hay que pedir que lo publique o tener su `etiquetas.csv` en el repositorio.

**Next step.** Ask C. Montoya to make the 384 px copy public or to share `etiquetas.csv`, rerun `check_split.py` until all three sets match, then run `CONFIG=config/config.sh bash run.sh` once on a Colab GPU to confirm the scripts reproduce model_v3. Port the IDRiD cross-device notebook as stage 07.

---

## 2026-09-22 to 2026-10-07 · Session 0 · Colab runs (reconstructed on 2026-10-08)

Rebuilt from the archived notebooks and `results/model_v1.json`; no prediction was written at the time.

**Objective.** First real model for referable DR (ICDR ≥ 2) on APTOS 2019, then add EyePACS and split by patient.

**Result.**

| Run | Data, split | Test images | AUROC | Sensitivity | Specificity | Status |
|---|---|---|---|---|---|---|
| model_v1, 2026-09-22 | APTOS, per image | 550 | 0.981 | 94.2% | 91.4% | kept |
| model_v2, 2026-10-06 | APTOS + EyePACS, per patient | 5 018 | 0.924 | 90.1% | 73.2% | discarded |
| model_v3, 2026-10-07 | APTOS + EyePACS, per patient | 5 540 | 0.930 | 90.5% | 74.0% | kept |

model_v3 trained 12 epochs in 75.8 min on a Colab T4; validation AUROC peaked at 0.9265 in epoch 6. In `RD_02_APTOS_EyePACS_v3.ipynb` the training cell failed with `NameError: name 'CONFIG' is not defined`: the kernel no longer held the configuration cell. That notebook is superseded by the checkpoint version and is not archived here.

**Costliest error.** Reusé `RD_01_primer_modelo_APTOS.ipynb` para la corrida v2 en vez de copiarlo. Se perdió el código exacto de model_v1 (solo queda su ficha JSON y los pesos) y la corrida v2 quedó con estado mezclado: la división impresa dice 5 540 imágenes de prueba, pero el examen se hizo sobre 5 018, con un umbral distinto (0.021 frente a 0.057) aunque el registro de entrenamiento es idéntico al de v3. Esos números no se pueden defender ante nadie. Costó un entrenamiento entero de 73 minutos en T4 que no sirve como evidencia. Lección: un cuaderno por corrida, y celdas que lean del disco y no de variables vivas; eso es lo que hacen ahora los scripts numerados.

**Next step.** Move the code out of Colab into versioned scripts (done in Session 1).
