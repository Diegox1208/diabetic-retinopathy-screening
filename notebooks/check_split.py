#!/usr/bin/env python3
#### created by Diego G. Salas
#
# Checks that workflow/retina.py rebuilds the model_v3 patient split from labels alone.
# StratifiedGroupKFold depends only on the set of (patient, grade) pairs, so no image is needed.
#   python notebooks/check_split.py --aptos <dir with train.csv> --eyepacs <dir with trainLabels.csv>

import argparse
import sys
from pathlib import Path

import pandas as pd

repo_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_dir / "workflow"))
import retina  # noqa: E402

#---- ##-- Configuration ----
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aptos", required=True, help="folder with APTOS train.csv")
parser.add_argument("--eyepacs", required=True, help="folder with EyePACS etiquetas.csv or trainLabels.csv")
parser.add_argument("--drop", nargs="*", default=["EyePACS:16028_left"],
                    help="dataset:image_id dropped in Colab because the file was missing")
parser.add_argument("--colab", default=str(repo_dir / "results" / "colab_runs.tsv"))
parser.add_argument("--version", default="model_v3")
parser.add_argument("--out", default=str(repo_dir / "results" / "split_check.tsv"))
args = parser.parse_args()

##### Data files #####
#-----------------------------------------------------------
df = retina.unify({"APTOS2019": retina.read_aptos(args.aptos), "EyePACS": retina.read_eyepacs(args.eyepacs)})
for item in args.drop:
    source, image_id = item.split(":", 1)
    df = df[~((df["source_dataset"] == source) & (df["image_id"] == image_id))]
df = df.reset_index(drop=True)
colab = pd.read_csv(args.colab, sep="\t").set_index("version").loc[args.version]

##### Analysis #####
#-----------------------------------------------------------
with retina.Timer("Split"):
    parts = retina.split_by_patient(df)

rows = []
for name, p in parts.items():
    rows.append(dict(split=name,
                     images_rebuilt=len(p), images_colab=int(colab[f"printed_{name}_images"]),
                     referable_rebuilt=round(p["referable"].mean(), 2), referable_colab=colab[f"printed_{name}_referable"],
                     aptos=int((p["source_dataset"] == "APTOS2019").sum()), eyepacs=int((p["source_dataset"] == "EyePACS").sum()),
                     referable_images=int(p["referable"].sum()),
                     fingerprint=retina.fingerprint(p["image_id"])))
out = pd.DataFrame(rows)
out["match"] = (out["images_rebuilt"] == out["images_colab"]) & (out["referable_rebuilt"] == out["referable_colab"])

##### Saving data #####
#-----------------------------------------------------------
print(out.to_string(index=False))
out.to_csv(args.out, sep="\t", index=False)
print(f"Saved {args.out}; all match: {bool(out['match'].all())}")
