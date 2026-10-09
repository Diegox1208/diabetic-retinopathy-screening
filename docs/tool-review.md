# Tool review of EfficientNet-B0 from timm, fine-tuned with PyTorch and split with scikit-learn StratifiedGroupKFold

Diego G. Salas · versions tested are torch 2.13.0 (CPU), timm 1.0.30 and scikit-learn 1.9.1 on the synthetic test set, while the Colab runs did not print their versions · dataset APTOS 2019 + EyePACS (Kaggle)

## What it does

`timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)` loads a 4.0 M-parameter CNN with ImageNet weights and replaces its classifier with one logit. Training minimises binary cross-entropy with AdamW (weight decay 1e-4) and a cosine learning-rate schedule stepped per batch, in mixed precision on a GPU. The epoch with the best validation AUROC is kept. `StratifiedGroupKFold` assigns whole patients to folds while keeping the ICDR grade distribution. One fold of 7 becomes test, then one of 6 of the rest becomes validation (about 14% each).

## Parameters that change the result

| Parameter | Tested values | Effect observed | Evidence |
|---|---|---|---|
| Training data and split unit | APTOS per image (model_v1) vs APTOS + EyePACS per patient (model_v3) | test AUROC 0.981 vs 0.930, specificity at the 90%-sensitivity threshold 91.4% vs 74.0%, threshold 0.445 vs 0.057. Test sets differ, so this is not a controlled comparison | `results/colab_runs.tsv` |
| Labels fed to the split | EyePACS labels of the 384 px copy vs the public `trainLabels.csv`, same seed | train / val / test 27 706 / 5 541 / 5 540 vs 27 705 / 5 541 / 5 541, referable fraction 21% vs 22% in train | `results/split_check.tsv` |
| `pretrained` | 1 (all Colab runs) vs 0 (synthetic test only) | not compared on the same data, 0 exists so the test runs offline | `tests/config.test.sh` |

## Where it fails

- **Notebook state.** The model_v2 run printed a split with 5 540 test images and then evaluated 5 018. Its training log is identical to model_v3's to four decimals, yet its validation threshold differs (0.021 vs 0.057), so the evaluation cell ran on objects left by an earlier execution. Numbered scripts that read their inputs from disk remove this class of error.
- **Scheduler warning with mixed precision.** PyTorch warns that `lr_scheduler.step()` ran before `optimizer.step()`. `GradScaler` skips the optimiser step when the first batch overflows in float16, so the cosine schedule runs one step ahead. Over 10 392 steps (866 × 12) the effect is negligible, and the order was kept to stay faithful to the Colab runs.
- **Low operating threshold.** Reaching 90% sensitivity on validation required a threshold of 0.057 in model_v3. The metric is valid, but the probabilities are not calibrated as risks, and calibration was not measured.
- **Pretrained weights need the network.** timm fetches them from the Hugging Face Hub (21.4 MB, unauthenticated warning in Colab). Offline machines need a cached copy or `pretrained=0`.
- **Group split with one image per patient.** APTOS publishes no patient id, so grouping protects only EyePACS eyes. For APTOS the split is still per image.

## Alternative

RETFound (Zhou et al., Nature 2023), a foundation model pre-trained on 1.6 M retinal images, is the natural next comparison for the same split. Not run here.

## Source

- Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. ICML 2019.
- timm documentation for `timm.create_model`, https://huggingface.co/docs/timm
- scikit-learn documentation for `StratifiedGroupKFold`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- PyTorch documentation for automatic mixed precision and `GradScaler`, https://pytorch.org/docs/stable/amp.html
