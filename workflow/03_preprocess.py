#!/usr/bin/env python3
#### created by Diego G. Salas
#### adapted from notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb, cell "Paso 2. Preprocesar las fotos"
#
# Stage 03 crops the black border, pads to a square and resizes every image once, with optional CLAHE.
# Images already on disk are not redone, so an interrupted run resumes where it stopped.

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm.auto import tqdm

import retina

##### Getopts #####
#-----------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", required=True, help="manifest from stage 02")
parser.add_argument("--out-dir", required=True, help="folder for the preprocessed images")
parser.add_argument("--out", required=True, help="output manifest with a prep_path column")
parser.add_argument("--size", type=int, default=384)
parser.add_argument("--clahe", type=int, default=0, choices=[0, 1])
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

if not Path(args.manifest).is_file():
    print(f"/!\\ Error : {args.manifest} does not exist; run stage 02 first", file=sys.stderr)
    sys.exit(1)

##### Data files #####
#-----------------------------------------------------------
df = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str, "patient_id": str})
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
# File names encode the parameters that change the pixels, size and CLAHE.
suffix = f"_{args.size}px" + ("_clahe" if args.clahe else "")
df["prep_path"] = [str(out_dir / f"{s}_{i.replace('/', '_')}{suffix}.jpg")
                   for s, i in zip(df["source_dataset"], df["image_id"])]

##### Functions #####
#-----------------------------------------------------------
def process_and_save(row):
    img = retina.preprocess_image(row.file_path, args.size, bool(args.clahe))
    cv2.imwrite(row.prep_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

##### Analysis #####
#-----------------------------------------------------------
with retina.Timer("Preprocessing"):
    pending = [r for r in df.itertuples() if not Path(r.prep_path).is_file()]
    print(f"{len(pending)} of {len(df)} images to preprocess")
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(process_and_save, pending), total=len(pending), desc="Preprocessing"))
    missing = [p for p in df["prep_path"] if not Path(p).is_file()]
    if missing:
        print(f"/!\\ Error : {len(missing)} preprocessed images missing, e.g. {missing[:3]}", file=sys.stderr)
        sys.exit(1)

##### Saving data #####
#-----------------------------------------------------------
df.to_csv(args.out, sep="\t", index=False)
print(f"Saved {args.out}")
