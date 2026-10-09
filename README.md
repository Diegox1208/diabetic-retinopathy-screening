# diabetic-retinopathy-screening

An EfficientNet-B0 fine-tuned on APTOS 2019 and EyePACS flags referable diabetic retinopathy, ICDR grade 2 or above, at 90.5% sensitivity and 74.0% specificity on 5 540 test images from patients it never saw, with an AUROC of 0.930.

Diego G. Salas · data from APTOS 2019 Blindness Detection on Kaggle with 3 662 images, and from EyePACS, the train set of the Kaggle Diabetic Retinopathy Detection challenge with 35 126 images, used through a 384 px copy prepared by C. Montoya

## Run it

```bash
micromamba create -n diabetic-retinopathy-screening -f environment.yml && micromamba activate diabetic-retinopathy-screening
bash run.sh
```

Runs on `tests/data`, 80 synthetic images in 1.2 MB, in 40 s on a laptop CPU. `DRY_RUN=1` prints the commands. `CONFIG=config/config.sh bash run.sh` runs the full data with the model_v3 parameters. It needs `~/.kaggle/kaggle.json`, access to the EyePACS copy and a GPU. The 12 epochs took 75.8 min on a Colab T4.

<table>
<thead><tr><th>Stage</th><th>Script</th><th>Output</th></tr></thead>
<tbody><tr><td>01</td><td><code>workflow/01_download.sh</code></td><td>Kaggle downloads in <code>data/raw/&lt;name&gt;</code></td></tr></tbody>
<tbody><tr><td>02</td><td><code>workflow/02_manifest.py</code></td><td>one schema for all datasets, images with a missing file dropped</td></tr></tbody>
<tbody><tr><td>03</td><td><code>workflow/03_preprocess.py</code></td><td>black border cropped, padded to square, resized, optional CLAHE</td></tr></tbody>
<tbody><tr><td>04</td><td><code>workflow/04_split.py</code></td><td>train / validation / test by patient, stratified by grade</td></tr></tbody>
<tbody><tr><td>05</td><td><code>workflow/05_train.py</code></td><td>best-validation weights, checkpoint per epoch, resumable</td></tr></tbody>
<tbody><tr><td>06</td><td><code>workflow/06_evaluate.py</code></td><td>threshold chosen on validation, metrics with 95% CI on test, model card</td></tr></tbody>
</table>

## Main result

<table>
<thead><tr><th>Run</th><th>Data</th><th>Split unit</th><th>Test images</th><th>Referable</th><th>Threshold</th><th>AUROC [95% CI]</th><th>Sensitivity [95% CI]</th><th>Specificity [95% CI]</th></tr></thead>
<tbody><tr><td>model_v1</td><td>APTOS</td><td>image</td><td>550</td><td>223</td><td>0.445</td><td>0.981 [0.972–0.989]</td><td>94.2% [90.3–96.6]</td><td>91.4% [87.9–94.0]</td></tr></tbody>
<tbody><tr><td>model_v3</td><td>APTOS + EyePACS</td><td>patient</td><td>5 540</td><td>1 226</td><td>0.057</td><td>0.930 [0.920–0.939]</td><td>90.5% [88.8–92.1]</td><td>74.0% [72.7–75.3]</td></tr></tbody>
</table>

The source is `results/colab_runs.tsv`, extracted by `python notebooks/extract_runs.py` from `results/model_v1.json` and the outputs of `notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb`. The threshold is the one that reaches 90% sensitivity on validation, and test is evaluated once. Sensitivity and specificity carry Wilson intervals, AUROC a 1 000-resample bootstrap.

The two rows are not a like-for-like comparison. model_v1 is tested on 550 APTOS images split per image, model_v3 on 5 540 images, most of them EyePACS, split per patient. At the same 90% sensitivity target, the second setting lets 26.0% of non-referable images through as referrals, 1 121 of 4 314, against 8.6% in the first, 28 of 327.

![model_v3 confusion matrix and ROC on test](results/figures/model_v3_confusion_roc.png)

*model_v3 on test, figure archived from the Colab run. The legend reads "model_v1" because the notebook hard-coded that label, but the counts and the AUROC are model_v3's. The training curve is in `results/figures/model_v3_training_curve.png`, where validation AUROC peaked at 0.9265 in epoch 6 of 12.*

## Confirmed vs not confirmed

<table>
<thead><tr><th>Claim</th><th>Status</th><th>Basis</th></tr></thead>
<tbody><tr><td>model_v3 reaches AUROC 0.930, sensitivity 90.5%, specificity 74.0%</td><td>single run, one seed</td><td>printed output of the archived notebook, weights not in this repository</td></tr></tbody>
<tbody><tr><td>model_v1 reaches AUROC 0.981, sensitivity 94.2%, specificity 91.4%</td><td>single run, one seed</td><td><code>results/model_v1.json</code>, weights load into stage 06 with all keys matched, the training code was overwritten</td></tr></tbody>
<tbody><tr><td>model_v2 reached AUROC 0.924, specificity 73.2%</td><td>discarded</td><td>its test had 5 018 images while the split printed in the same notebook had 5 540, see <code>PROGRESS.md</code></td></tr></tbody>
<tbody><tr><td>The six scripts run end to end</td><td>confirmed on the synthetic test set only</td><td>two clean runs with identical losses and AUROCs, see Session 1 in <code>PROGRESS.md</code></td></tr></tbody>
<tbody><tr><td>Stage 05 resumes from its checkpoint and refuses a changed configuration</td><td>confirmed on the synthetic test set only</td><td>weights deleted, rerun resumed at epoch 3, <code>--epochs 3</code> stopped with an error, see Session 1 in <code>PROGRESS.md</code></td></tr></tbody>
<tbody><tr><td>The scripts rebuild the model_v3 split</td><td>not confirmed</td><td>the public label files give 27 705 / 5 541 / 5 541 images against 27 706 / 5 541 / 5 540 in Colab, see <code>results/split_check.tsv</code>. An exact check needs the labels of the 384 px copy</td></tr></tbody>
<tbody><tr><td>The scripts reproduce the model_v3 metrics</td><td>not run</td><td>needs a GPU and access to the EyePACS copy</td></tr></tbody>
</table>

## Limitations

- APTOS has no patient identifier, so each APTOS image counts as its own patient. In model_v1, which used APTOS alone, the per-image split cannot rule out the same eye in train and test.
- The 90%-sensitivity threshold of model_v3 is 0.057, so an image is referred when the model gives it 5.7% probability or more. Calibration was not measured.
- EyePACS dominates model_v3 with 35 125 of 38 787 images. Stage 06 now computes metrics per dataset in `<version>_by_source.tsv`, but the Colab run did not.
- The EyePACS copy used for training, `cristbalmontoyabao/atros-dataset`, returned 403 Forbidden to the Kaggle account in `~/.kaggle` of the machine that built this repository, and the public `trainLabels.csv` does not reproduce its split exactly. A third party cannot rebuild model_v3 from this repository today.
- All images come from desktop fundus cameras. Nothing here measures performance on smartphone or other low-cost cameras.
- Synthetic test images check that every stage runs. They say nothing about retinopathy.

## Not done on purpose

- No cross-device evaluation on IDRiD or mBRSET in this repository. It lives in a separate Colab notebook and will be ported as a stage 07.
- No Grad-CAM or other saliency maps.
- No CLAHE run. The switch `useClahe` exists but every reported run used `0`.
- No model weights in git, since `models/` is ignored. `model_v1.pt` stays local at 16 MB, and the notebook saved `model_v3.pt` to the Google Drive folder `Retinopatia_modelos LOCAL/`.
- No hyperparameter search. One seed, the parameters of the Colab notebook.
