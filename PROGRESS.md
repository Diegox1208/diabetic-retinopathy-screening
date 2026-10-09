# PROGRESS

One entry per session, newest first. Structured fields in English, the reflection in Spanish.

---

## Session 1 · From Colab notebooks to a pipeline

### Objective

Turn the three Colab notebooks into six numbered scripts with one command, run them end to end on a tiny dataset, and record what the Colab runs actually measured.

### Prediction

The scripts run on the test dataset in under 2 minutes on CPU, and the patient split rebuilt from public label files reproduces Colab's 27 706 / 5 541 / 5 540 images exactly.

### Test dataset

`python tests/make_test_data.py` → 80 synthetic fundus-like images, 40 in the APTOS layout and 20 patients × 2 eyes in the EyePACS layout, 1.2 MB in `tests/data/raw`.

### Result

<table>
<thead><tr><th>Item</th><th>Predicted</th><th>Outcome</th></tr></thead>
<tbody><tr><td><code>DRY_RUN=1 bash run.sh</code></td><td>prints six stages</td><td>done, each command printed and logged, nothing ran</td></tr></tbody>
<tbody><tr><td><code>bash run.sh</code> on <code>tests/</code></td><td>&lt; 2 min</td><td>40 s on CPU with torch 2.13.0 and timm 1.0.30, two clean runs gave identical losses and AUROCs</td></tr></tbody>
<tbody><tr><td>Stage 05 resume</td><td>resumes after an interruption</td><td>weights deleted, rerun resumed at epoch 3, with <code>--epochs 3</code> it refused the checkpoint</td></tr></tbody>
<tbody><tr><td><code>model_v1.pt</code> in the new code</td><td>loads</td><td><code>load_state_dict(strict=True)</code> matched all keys</td></tr></tbody>
<tbody><tr><td>Split rebuilt from public labels by <code>notebooks/check_split.py</code></td><td>27 706 / 5 541 / 5 540</td><td>27 705 / 5 541 / 5 541, referable 22% vs 21% in train, not reproduced</td></tr></tbody>
<tbody><tr><td>EyePACS copy <code>cristbalmontoyabao/atros-dataset</code></td><td>downloadable</td><td><code>403 Client Error: Forbidden for url: https://api.kaggle.com/v1/datasets.DatasetApiService/GetDatasetMetadata</code></td></tr></tbody>
<tbody><tr><td>Colab results extracted by <code>notebooks/extract_runs.py</code></td><td>three runs</td><td>model_v1 and model_v3 kept, model_v2 discarded, see Session 0</td></tr></tbody>
</table>

Errors met during the session, quoted verbatim, each fixed in its own commit.

- `bash: line 1: unzip: command not found` while extracting `trainLabels.csv.zip` in WSL → `unzip` added to `environment.yml`, since stage 01 needs it for APTOS.
- `specificity  0.0% (95% CI -0.0%-43.5%)` on the test set → Wilson bounds clipped to [0, 1]. The notebook had the same code.
- Evaluation figure with the y tick labels cut and the axis label over the matrix → constrained layout, label placed inside the axes.

### Costliest error

Di por hecho que la copia de EyePACS de Cristóbal era pública y que sus etiquetas eran las del concurso original. Las dos cosas fallaron. Kaggle devuelve 403 con mi cuenta y la división reconstruida con `trainLabels.csv` se desvía en una imagen por conjunto. Me costó unos 20 minutos entre probar descargas, montar `check_split.py` y entender que el desvío viene de las etiquetas y no del código, porque `StratifiedGroupKFold` solo depende del conjunto de pares paciente-grado. La lección es que un dataset que solo existe en la cuenta de un compañero no es reproducible, y que antes de escribir el README hay que pedir que lo publique o tener su `etiquetas.csv` en el repositorio.

### Next step

Ask C. Montoya to make the 384 px copy public or to share `etiquetas.csv`, rerun `check_split.py` until all three sets match, then run `CONFIG=config/config.sh bash run.sh` once on a Colab GPU to confirm the scripts reproduce model_v3. Port the IDRiD cross-device notebook as stage 07.

---

## Session 0 · Colab runs, reconstructed in Session 1

Rebuilt from the archived notebooks and `results/model_v1.json`. No prediction was written at the time.

### Objective

First real model for referable DR, ICDR grade 2 or above, on APTOS 2019, then add EyePACS and split by patient.

### Result

<table>
<thead><tr><th>Run</th><th>Data, split</th><th>Test images</th><th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>Status</th></tr></thead>
<tbody><tr><td>model_v1</td><td>APTOS, per image</td><td>550</td><td>0.981</td><td>94.2%</td><td>91.4%</td><td>kept</td></tr></tbody>
<tbody><tr><td>model_v2</td><td>APTOS + EyePACS, per patient</td><td>5 018</td><td>0.924</td><td>90.1%</td><td>73.2%</td><td>discarded</td></tr></tbody>
<tbody><tr><td>model_v3</td><td>APTOS + EyePACS, per patient</td><td>5 540</td><td>0.930</td><td>90.5%</td><td>74.0%</td><td>kept</td></tr></tbody>
</table>

model_v3 trained 12 epochs in 75.8 min on a Colab T4, and validation AUROC peaked at 0.9265 in epoch 6. In `RD_02_APTOS_EyePACS_v3.ipynb` the training cell failed with `NameError: name 'CONFIG' is not defined` because the kernel no longer held the configuration cell. That notebook is superseded by the checkpoint version and is not archived here.

### Costliest error

Reusé `RD_01_primer_modelo_APTOS.ipynb` para la corrida v2 en vez de copiarlo. Se perdió el código exacto de model_v1, del que solo quedan su ficha JSON y los pesos, y la corrida v2 quedó con estado mezclado. La división impresa dice 5 540 imágenes de prueba, pero el examen se hizo sobre 5 018, con un umbral distinto, 0.021 frente a 0.057, aunque el registro de entrenamiento es idéntico al de v3. Esos números no se pueden defender ante nadie. Costó un entrenamiento entero de 73 minutos en T4 que no sirve como evidencia. La lección es un cuaderno por corrida, con celdas que lean del disco y no de variables vivas. Eso es lo que hacen ahora los scripts numerados.

### Next step

Move the code out of Colab into versioned scripts, done in Session 1.
