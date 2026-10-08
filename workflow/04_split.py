#!/usr/bin/env python3
#### created by Diego G. Salas, 2026-10-08
#### adapted from: notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb, cell "Paso 3 — Dividir"
#
# Stage 04: split train / validation / test by patient, stratified by ICDR grade.
# No patient appears in two sets, so both eyes of one person stay together.

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import retina

##### Getopts #####
#-----------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", required=True, help="manifest with prep_path, from stage 03")
parser.add_argument("--out", required=True, help="output table with a split column, TSV")
parser.add_argument("--summary", required=True, help="output table of images per split and dataset, TSV")
parser.add_argument("--test-folds", type=int, default=7, help="1 of n folds becomes test")
parser.add_argument("--val-folds", type=int, default=6, help="1 of n remaining folds becomes validation")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if not Path(args.manifest).is_file():
    print(f"/!\\ Error : {args.manifest} does not exist; run stage 03 first", file=sys.stderr)
    sys.exit(1)

##### Functions #####
#-----------------------------------------------------------
def split_off(d, n_splits, seed):
    # Returns (rest, one fold) of a StratifiedGroupKFold on grade, grouped by patient.
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, fold_idx = next(sgkf.split(d, d["icdr_grade"], groups=d["patient_id"]))
    return d.iloc[rest_idx], d.iloc[fold_idx]

##### Data files #####
#-----------------------------------------------------------
df = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str, "patient_id": str})

##### Analysis #####
#-----------------------------------------------------------
with retina.Timer("Split"):
    trainval, test = split_off(df, args.test_folds, args.seed)
    train, val = split_off(trainval, args.val_folds, args.seed)
    parts = {"train": train, "val": val, "test": test}

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(parts[a]["patient_id"]) & set(parts[b]["patient_id"]):
            print(f"/!\\ Error : patients shared between {a} and {b}", file=sys.stderr)
            sys.exit(1)
    if sum(len(p) for p in parts.values()) != len(df):
        print("/!\\ Error : the three sets do not add up to the manifest", file=sys.stderr)
        sys.exit(1)
    sources = set(df["source_dataset"])
    for name, p in parts.items():
        if set(p["source_dataset"]) != sources:
            print(f"/!\\ Error : {name} lacks a dataset: {p['source_dataset'].value_counts().to_dict()}", file=sys.stderr)
            sys.exit(1)

    out = pd.concat([p.assign(split=name) for name, p in parts.items()], ignore_index=True)

##### Table #####
#-----------------------------------------------------------
summary = (out.groupby(["split", "source_dataset"])
              .agg(images=("image_id", "size"), patients=("patient_id", "nunique"), referable=("referable", "sum"))
              .reset_index())
summary["referable_fraction"] = (summary["referable"] / summary["images"]).round(3)
print(summary.to_string(index=False))

##### Saving data #####
#-----------------------------------------------------------
out.to_csv(args.out, sep="\t", index=False)
summary.to_csv(args.summary, sep="\t", index=False)
print(f"Saved {args.out} and {args.summary}")
