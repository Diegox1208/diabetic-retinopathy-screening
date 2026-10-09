#!/usr/bin/env python3
#### created by Diego G. Salas
#
# Builds tests/data/raw with synthetic fundus-like images in the exact file layout of the two Kaggle
# downloads, so every stage runs on a laptop without credentials. They are not real retinas, so no metric
# computed on them says anything about diabetic retinopathy.
#   python tests/make_test_data.py

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

#---- ##-- Configuration ----
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "raw"))
parser.add_argument("--per-grade", type=int, default=8, help="APTOS-like images per ICDR grade")
parser.add_argument("--patients", type=int, default=20, help="EyePACS-like patients, two eyes each")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
out = Path(args.out)

##### Functions #####
#-----------------------------------------------------------
def fundus(grade, width=150, height=118):
    # Orange disc on a black, non-square canvas (so the crop and pad code is exercised),
    # dark vessels, a pale optic disc, red dots from grade 1 and yellow exudates from grade 3.
    img = np.zeros((height, width, 3), np.uint8)
    cx, cy, r = width // 2, height // 2, int(height * 0.45)
    yy, xx = np.mgrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    inside = dist <= 1
    shade = np.clip(1.1 - 0.5 * dist, 0, 1)
    img[inside] = (np.stack([30 * shade, 90 * shade, 200 * shade], -1)[inside]).astype(np.uint8)   # BGR
    disc = (cx + int(r * 0.45 * rng.choice([-1, 1])), cy)
    for _ in range(4):
        end = (int(cx + rng.uniform(-r, r) * 0.8), int(cy + rng.uniform(-r, r) * 0.8))
        cv2.line(img, disc, end, (20, 40, 120), 1)
    cv2.circle(img, disc, max(3, r // 7), (150, 210, 240), -1)
    for _ in range({0: 0, 1: 2, 2: 6, 3: 10, 4: 14}[grade]):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.8) * r
        cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))), int(rng.integers(1, 3)), (10, 10, 150), -1)
    for _ in range(6 if grade >= 3 else 0):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.7) * r
        cv2.circle(img, (int(cx + d * np.cos(a)), int(cy + d * np.sin(a))), 2, (90, 220, 240), -1)
    noise = rng.normal(0, 4, img.shape)
    img = np.where(inside[..., None], np.clip(img + noise, 0, 255), img).astype(np.uint8)
    return img

#---- ##-- APTOS layout, train.csv (id_code, diagnosis) + train_images/<id>.png ----
aptos = out / "aptos"
(aptos / "train_images").mkdir(parents=True, exist_ok=True)
rows = []
for grade in range(5):
    for k in range(args.per_grade):
        id_code = f"{rng.integers(16**11, 16**12):012x}"
        cv2.imwrite(str(aptos / "train_images" / f"{id_code}.png"), fundus(grade))
        rows.append(dict(id_code=id_code, diagnosis=grade))
pd.DataFrame(rows).to_csv(aptos / "train.csv", index=False)

#---- ##-- EyePACS layout, eyepacs_384/etiquetas.csv + <patient>_<eye>.jpg, nested as Kaggle unzips it ----
eyepacs = out / "eyepacs" / "eyepacs_384"
eyepacs.mkdir(parents=True, exist_ok=True)
rows = []
for p in range(args.patients):
    patient_id = 1000 + p
    grade = p % 5
    for eye in ("left", "right"):
        image_id = f"{patient_id}_{eye}"
        cv2.imwrite(str(eyepacs / f"{image_id}.jpg"), fundus(grade), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(dict(image_id=image_id, icdr_grade=grade, patient_id=patient_id, ojo=eye))
pd.DataFrame(rows).to_csv(eyepacs / "etiquetas.csv", index=False)

total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
print(f"wrote {args.per_grade * 5} APTOS-like and {args.patients * 2} EyePACS-like images to {out} ({total / 1e6:.2f} MB)")
