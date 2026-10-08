#!/usr/bin/env python3
#### created by Diego G. Salas, 2026-10-08
#### adapted from: notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb, cells "Mirar los datos" to "Control de seguridad"
#
# Stage 02: put every dataset in one schema and drop images whose file is missing.
# Schema: image_id, source_dataset, file_path, icdr_grade, referable, patient_id, eye, device.

import argparse
import sys
from pathlib import Path

import pandas as pd

import retina

##### Getopts #####
#-----------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw", required=True, help="folder with one subfolder per dataset name")
parser.add_argument("--info", required=True, help="datasets.tsv with columns dataset name accession source")
parser.add_argument("--out", required=True, help="output manifest, TSV")
parser.add_argument("--summary", required=True, help="output table of images per dataset and grade, TSV")
args = parser.parse_args()

raw_dir = Path(args.raw)
if not raw_dir.is_dir():
    print(f"/!\\ Error : {raw_dir} does not exist; run stage 01 first", file=sys.stderr)
    sys.exit(1)

##### Data files #####
#-----------------------------------------------------------
with retina.Timer("Data files"):
    info = pd.read_csv(args.info, sep="\t")
    tables = {}
    for row in info.itertuples():
        if row.dataset not in retina.READERS:
            print(f"/!\\ Error : no reader for dataset {row.dataset}", file=sys.stderr)
            sys.exit(1)
        folder = raw_dir / row.name
        if not folder.is_dir():
            print(f"[02_manifest] missing {folder}, skipping {row.dataset}", file=sys.stderr)
            continue
        tables[row.dataset] = retina.READERS[row.dataset](folder)
    if not tables:
        print("/!\\ Error : no dataset found", file=sys.stderr)
        sys.exit(1)
    df = retina.unify(tables)

##### Analysis #####
#-----------------------------------------------------------
with retina.Timer("Analysis"):
    exists = df["file_path"].map(lambda p: Path(p).is_file())
    for r in df[~exists].itertuples():
        print(f"[02_manifest] missing file, dropped: {r.source_dataset} {r.image_id}", file=sys.stderr)
    dropped = df[~exists].groupby("source_dataset").size()
    df = df[exists].reset_index(drop=True)

    if df.duplicated(["source_dataset", "image_id"]).any():
        print("/!\\ Error : duplicated images in the manifest", file=sys.stderr)
        sys.exit(1)
    if df[["image_id", "icdr_grade", "patient_id", "referable"]].isna().any().any():
        print("/!\\ Error : missing values in key columns", file=sys.stderr)
        sys.exit(1)

##### Table #####
#-----------------------------------------------------------
summary = (df.pivot_table(index="source_dataset", columns="icdr_grade", values="image_id",
                          aggfunc="count", fill_value=0)
             .rename(columns=lambda g: f"grade_{g}"))
summary["images"] = summary.sum(axis=1)
summary["referable"] = df.groupby("source_dataset")["referable"].sum()
summary["patients"] = df.groupby("source_dataset")["patient_id"].nunique()
summary["dropped_missing_file"] = dropped.reindex(summary.index, fill_value=0)
print(summary.to_string())

##### Saving data #####
#-----------------------------------------------------------
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, sep="\t", index=False)
summary.reset_index().to_csv(args.summary, sep="\t", index=False)
print(f"Saved {args.out} ({len(df)} images) and {args.summary}")
