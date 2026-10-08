#!/usr/bin/env python3
#### created by Diego G. Salas, 2026-10-08
#### adapted from: notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb, cells "Paso 6" and "Paso 7"
#
# Stage 06: choose the threshold on validation for a target sensitivity, then evaluate once on test.
# Sensitivity and specificity carry Wilson 95% intervals; AUROC a percentile bootstrap 95% interval.
# Writes the model card (JSON), per-dataset metrics, test predictions and the confusion/ROC figure.

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

import retina

##### Getopts #####
#-----------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--split", required=True, help="split table from stage 04")
parser.add_argument("--model-dir", required=True, help="folder with <version>.pt and <version>_params.json")
parser.add_argument("--prefix", required=True, help="output prefix, e.g. results/model_v3")
parser.add_argument("--version", required=True)
parser.add_argument("--architecture", default="efficientnet_b0")
parser.add_argument("--target-sensitivity", type=float, default=0.90)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--device", default="auto")
args = parser.parse_args()

weights_file = Path(args.model_dir) / f"{args.version}.pt"
for f in (Path(args.split), weights_file):
    if not f.is_file():
        print(f"/!\\ Error : {f} does not exist; run stages 04 and 05 first", file=sys.stderr)
        sys.exit(1)

##### Functions #####
#-----------------------------------------------------------
def metrics_at(y, p, threshold, n_boot):
    # Confusion counts, Wilson intervals and bootstrap AUROC at a fixed threshold.
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    sens, spec = retina.wilson(tp, tp + fn), retina.wilson(tn, tn + fp)
    auc = retina.auroc_bootstrap(y, p, n=n_boot) if len(np.unique(y)) == 2 else (float("nan"),) * 3
    return dict(n=int(len(y)), referable=int(y.sum()),
                auroc=[round(float(v), 4) for v in auc],
                sensitivity=[round(float(v), 4) for v in sens],
                specificity=[round(float(v), 4) for v in spec],
                confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))


def plot_confusion_roc(m, y, p, version, file):
    # Left: confusion matrix on a one-hue ramp. Right: ROC with the operating point labelled.
    blue, ink, muted, surface = "#2a78d6", "#1f1f1e", "#6b6b68", "#fcfcfb"
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#104281"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=surface)
    c = m["confusion"]
    cells = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    a1.imshow(cells, cmap=ramp)
    names = [["true negative", "false positive"], ["false negative", "true positive"]]
    for i in range(2):
        for j in range(2):
            dark = cells[i, j] > cells.max() / 2
            a1.text(j, i, f"{cells[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=10,
                    color="white" if dark else ink)
    a1.set_xticks([0, 1], ["predicted not referable", "predicted referable"], color=ink)
    a1.set_yticks([0, 1], ["not referable", "referable"], color=ink)
    a1.set_title(f"Confusion matrix, test (n = {m['n']})", color=ink, loc="left")

    fpr, tpr, _ = roc_curve(y, p)
    a2.plot([0, 1], [0, 1], "--", color=muted, linewidth=1)
    a2.plot(fpr, tpr, color=blue, linewidth=2)
    x0, y0 = 1 - m["specificity"][0], m["sensitivity"][0]
    a2.scatter([x0], [y0], s=64, color="#104281", edgecolor=surface, linewidth=2, zorder=3)
    a2.annotate(f"threshold: sens {y0:.1%}, spec {1 - x0:.1%}", (x0, y0), xytext=(12, -18),
                textcoords="offset points", color=ink, fontsize=9)
    a2.set_xlabel("1 − specificity", color=ink)
    a2.set_ylabel("sensitivity", color=ink)
    a2.set_title(f"ROC, test · {version} · AUROC {m['auroc'][0]:.3f}", color=ink, loc="left")
    a2.grid(color="#e6e6e3", linewidth=0.8)
    for a in (a1, a2):
        a.set_facecolor(surface)
        for s in a.spines.values():
            s.set_color("#d0d0cc")
    fig.tight_layout()
    fig.savefig(file, dpi=120, facecolor=surface)
    plt.close(fig)

##### Data files #####
#-----------------------------------------------------------
device = retina.resolve_device(args.device)
with retina.Timer("Data files"):
    df = pd.read_csv(args.split, sep="\t", dtype={"image_id": str, "patient_id": str})
    val, test = df[df["split"] == "val"], df[df["split"] == "test"]
    loader = lambda d: DataLoader(retina.RetinaDataset(d, retina.build_transforms(False)),
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = timm.create_model(args.architecture, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)

##### Analysis #####
#-----------------------------------------------------------
with retina.Timer("Threshold on validation"):
    p_val, y_val = retina.predict(model, loader(val), device)
    fpr_v, tpr_v, thresholds_v = roc_curve(y_val, p_val)
    idx = np.where(tpr_v >= args.target_sensitivity)[0][0]
    threshold = float(min(thresholds_v[idx], 1.0))      # roc_curve puts inf as the first threshold
    print(f"threshold\t{threshold:.4f}\t(validation sensitivity >= {args.target_sensitivity})")

with retina.Timer("Test"):
    p_test, y_test = retina.predict(model, loader(test), device)
    overall = metrics_at(y_test, p_test, threshold, args.bootstrap)
    by_source = []
    for source in sorted(test["source_dataset"].unique()):
        mask = (test["source_dataset"] == source).to_numpy()
        m = metrics_at(y_test[mask], p_test[mask], threshold, args.bootstrap)
        by_source.append(dict(source_dataset=source, n=m["n"], referable=m["referable"],
                              auroc=m["auroc"][0], auroc_lo=m["auroc"][1], auroc_hi=m["auroc"][2],
                              sensitivity=m["sensitivity"][0], sens_lo=m["sensitivity"][1], sens_hi=m["sensitivity"][2],
                              specificity=m["specificity"][0], spec_lo=m["specificity"][1], spec_hi=m["specificity"][2],
                              **m["confusion"]))

##### Table #####
#-----------------------------------------------------------
a, s, e = overall["auroc"], overall["sensitivity"], overall["specificity"]
print(f"{args.version} test: n {overall['n']} (referable {overall['referable']})")
print(f"  AUROC        {a[0]:.3f} (95% CI {a[1]:.3f}-{a[2]:.3f})")
print(f"  sensitivity  {s[0]:.1%} (95% CI {s[1]:.1%}-{s[2]:.1%})")
print(f"  specificity  {e[0]:.1%} (95% CI {e[1]:.1%}-{e[2]:.1%})")
print(pd.DataFrame(by_source).to_string(index=False))

##### Saving data #####
#-----------------------------------------------------------
params_file = Path(args.model_dir) / f"{args.version}_params.json"
train_info = json.loads(params_file.read_text()) if params_file.is_file() else {}
card = dict(
    version=args.version,
    date=datetime.datetime.now().isoformat(timespec="seconds"),
    task="binary: referable (ICDR >= 2) vs not referable",
    data=" + ".join(sorted(df["source_dataset"].unique())),
    split="per patient, StratifiedGroupKFold on ICDR grade",
    images_per_split={k: int(v) for k, v in df["split"].value_counts().items()},
    train_sources={k: int(v) for k, v in df[df["split"] == "train"]["source_dataset"].value_counts().items()},
    test_sources={k: int(v) for k, v in test["source_dataset"].value_counts().items()},
    threshold=threshold,
    target_sensitivity=args.target_sensitivity,
    test=overall,
    **train_info,
)
Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
with open(f"{args.prefix}_card.json", "w") as f:
    json.dump(card, f, indent=2)
pd.DataFrame(by_source).to_csv(f"{args.prefix}_by_source.tsv", sep="\t", index=False)
test.assign(prob=p_test.round(5), predicted=(p_test >= threshold).astype(int))[
    ["image_id", "source_dataset", "patient_id", "icdr_grade", "referable", "prob", "predicted"]
].to_csv(f"{args.prefix}_test_predictions.tsv", sep="\t", index=False)
plot_confusion_roc(overall, y_test, p_test, args.version, f"{args.prefix}_confusion_roc.png")
print(f"Saved {args.prefix}_card.json, _by_source.tsv, _test_predictions.tsv, _confusion_roc.png")
