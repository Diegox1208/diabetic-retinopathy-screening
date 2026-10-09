#!/usr/bin/env python3
#### created by Diego G. Salas

# Extracts the metrics, training history and figures of the three Colab runs
# (model_v1, model_v2, model_v3) into results/. Standard library only.
#   python notebooks/extract_runs.py

import argparse
import base64
import csv
import json
import re
import sys
from pathlib import Path

#---- ##-- Configuration ----
##### Set variables
repo_dir = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Extract Colab run results into results/.")
parser.add_argument("--notebooks", default=str(repo_dir / "notebooks"), help="folder with the archived notebooks")
parser.add_argument("--results", default=str(repo_dir / "results"), help="output folder")
args = parser.parse_args()

nb_dir = Path(args.notebooks)
results_dir = Path(args.results)
fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Each Colab run: which notebook printed it, and what data and split it used.
runs = [
    dict(version="model_v1", source="results/model_v1.json", notebook=None,
         data="APTOS2019", split="per image, 70/15/15, stratified by grade"),
    dict(version="model_v2", source="notebooks/RD_01_primer_modelo_APTOS.ipynb",
         notebook="RD_01_primer_modelo_APTOS.ipynb",
         data="APTOS2019 + EyePACS", split="per patient, StratifiedGroupKFold 6/7 then 5/6"),
    dict(version="model_v3", source="notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb",
         notebook="RD_02_APTOS_EyePACS_v3_checkpoint.ipynb",
         data="APTOS2019 + EyePACS", split="per patient, StratifiedGroupKFold 6/7 then 5/6"),
]

#---- ##-- Functions ----
def cell_outputs(cell):
    # Concatenated text of every stream output of a code cell.
    return "".join("".join(o.get("text", "")) for o in cell.get("outputs", []) if o.get("output_type") == "stream")


def cell_images(cell):
    return [o["data"]["image/png"] for o in cell.get("outputs", [])
            if o.get("output_type") in ("display_data", "execute_result") and "image/png" in o.get("data", {})]


def pct(s):
    return round(float(s) / 100, 4)


def parse_test_block(text):
    # Parses the "RESULTADO REAL" block printed by the evaluation cell.
    num = r"([\d.]+)"
    m = dict(
        threshold=re.search(r"Umbral elegido \(en validación\): " + num, text),
        n=re.search(r"Fotos en el examen: (\d+)\s+\(referibles: (\d+)\)", text),
        auroc=re.search(r"AUROC\s+: " + num + r"\s+\(IC95% " + num + "–" + num + r"\)", text),
        sens=re.search(r"Sensibilidad\s+: " + num + r"%\s+\(IC95% " + num + "%–" + num + r"%\)\s+→ detectó (\d+) de (\d+)", text),
        spec=re.search(r"Especificidad : " + num + r"%\s+\(IC95% " + num + "%–" + num + r"%\)\s+→ acertó (\d+) de (\d+)", text),
    )
    missing = [k for k, v in m.items() if v is None]
    if missing:
        return None
    tp, pos = int(m["sens"].group(4)), int(m["sens"].group(5))
    tn, neg = int(m["spec"].group(4)), int(m["spec"].group(5))
    return dict(
        threshold=float(m["threshold"].group(1)),
        n_test=int(m["n"].group(1)), referable_test=int(m["n"].group(2)),
        auroc=float(m["auroc"].group(1)), auroc_lo=float(m["auroc"].group(2)), auroc_hi=float(m["auroc"].group(3)),
        sens=pct(m["sens"].group(1)), sens_lo=pct(m["sens"].group(2)), sens_hi=pct(m["sens"].group(3)),
        spec=pct(m["spec"].group(1)), spec_lo=pct(m["spec"].group(2)), spec_hi=pct(m["spec"].group(3)),
        tp=tp, fn=pos - tp, tn=tn, fp=neg - tn,
    )


def parse_history(text):
    pattern = r"Época\s+(\d+)/\d+ \| pérdida train ([\d.]+) \| AUROC val ([\d.]+) \| (\d+)s"
    return [dict(epoch=int(e), train_loss=float(l), val_auroc=float(a), seconds=int(s))
            for e, l, a, s in re.findall(pattern, text)]


def parse_split(text):
    # "Entrenamiento : 27706 fotos | referibles: 21%" -> {"train": (27706, 0.21), ...}
    names = {"Entrenamiento": "train", "Validación": "val", "Prueba": "test"}
    found = re.findall(r"(Entrenamiento|Validación|Prueba)\s*:\s*(\d+) fotos \| referibles: (\d+)%", text)
    return {names[k]: (int(n), int(r) / 100) for k, n, r in found}

#---- ##-- Data files ----
summary_rows, history_rows = [], []

for run in runs:
    ##### model_v1: the model card saved by the notebook is the only record
    if run["notebook"] is None:
        card = json.loads((results_dir / "model_v1.json").read_text(encoding="utf-8"))
        t = card["test"]
        row = dict(threshold=card["umbral"], n_test=t["n"], referable_test=t["referibles"],
                   auroc=t["auroc"][0], auroc_lo=t["auroc"][1], auroc_hi=t["auroc"][2],
                   sens=t["sensibilidad"][0], sens_lo=t["sensibilidad"][1], sens_hi=t["sensibilidad"][2],
                   spec=t["especificidad"][0], spec_lo=t["especificidad"][1], spec_hi=t["especificidad"][2],
                   tp=t["matriz"]["tp"], fn=t["matriz"]["fn"], tn=t["matriz"]["tn"], fp=t["matriz"]["fp"])
        history = [dict(epoch=h["epoca"], train_loss=h["perdida_train"], val_auroc=h["auc_val"], seconds="")
                   for h in card["historial"]]
        split = {}
    ##### model_v2 and model_v3: parse the printed outputs
    else:
        nb = json.loads((nb_dir / run["notebook"]).read_text(encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        text = "\n".join(cell_outputs(c) for c in code)
        row = parse_test_block(text)
        if row is None:
            print(f"/!\\ Error : no test block in {run['notebook']}", file=sys.stderr)
            sys.exit(1)
        history = parse_history(text)
        split = parse_split(text)
        ##### Figures, located by the code that drew them
        tag = run["version"]
        for c in code:
            src = "".join(c["source"])
            images = cell_images(c)
            if not images:
                continue
            if "matriz = np.array" in src:
                name = f"{tag}_confusion_roc.png"
            elif "pd.DataFrame(historial)" in src:
                name = f"{tag}_training_curve.png"
            elif "nombres = [" in src:
                name = "grade_examples.png"
            else:
                continue
            (fig_dir / name).write_bytes(base64.b64decode(images[0]))
            print(f"figure\t{fig_dir / name}")

    best = max(history, key=lambda h: h["val_auroc"])
    summary_rows.append(dict(version=run["version"], data=run["data"], split=run["split"],
                             **{f"printed_{k}_images": split.get(k, ("", ""))[0] for k in ("train", "val", "test")},
                             **{f"printed_{k}_referable": split.get(k, ("", ""))[1] for k in ("train", "val", "test")},
                             **row,
                             best_val_auroc=round(best["val_auroc"], 4), best_epoch=best["epoch"],
                             source=run["source"]))
    history_rows += [dict(version=run["version"], **h) for h in history]

#---- ##-- Saving data ----
def write_tsv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"table\t{path}")

write_tsv(results_dir / "colab_runs.tsv", summary_rows)
write_tsv(results_dir / "colab_history.tsv", history_rows)
