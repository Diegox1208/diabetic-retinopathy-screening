#### created by Diego G. Salas
#### adapted from the Colab notebook notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb
#
# Functions shared by the numbered stages for preprocessing, dataset, prediction and metrics.
# Imported as `import retina` by the scripts in workflow/.

import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


##### Timing #####
#-----------------------------------------------------------
class Timer:
    def __init__(self, block):
        self.block = block

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        print(f"{self.block}\tdone in {round(time.time() - self.t)}s", flush=True)


##### Reproducibility #####
#-----------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def fingerprint(image_ids):
    return hashlib.md5("|".join(sorted(map(str, image_ids))).encode()).hexdigest()


##### Dataset readers #####
#-----------------------------------------------------------
def read_aptos(folder):
    t = pd.read_csv(Path(folder) / "train.csv")
    return pd.DataFrame({
        "image_id": t["id_code"].astype(str),
        "file_path": [str(Path(folder) / "train_images" / f"{i}.png") for i in t["id_code"]],
        "icdr_grade": t["diagnosis"].astype(int),
        "patient_id": t["id_code"].astype(str),
        "eye": "unknown",
    })


def read_eyepacs(folder):
    folder = Path(folder)
    copy = sorted(folder.rglob("etiquetas.csv"))
    if copy:
        t = pd.read_csv(copy[0])
        return pd.DataFrame({
            "image_id": t["image_id"].astype(str),
            "file_path": [str(copy[0].parent / f"{i}.jpg") for i in t["image_id"]],
            "icdr_grade": t["icdr_grade"].astype(int),
            "patient_id": t["patient_id"].astype(str),
            "eye": t["ojo"].astype(str),
        })
    original = sorted(folder.rglob("trainLabels.csv"))
    if original:
        t = pd.read_csv(original[0])
        parts = t["image"].str.rsplit("_", n=1, expand=True)
        return pd.DataFrame({
            "image_id": t["image"].astype(str),
            "file_path": [str(original[0].parent / "train" / f"{i}.jpeg") for i in t["image"]],
            "icdr_grade": t["level"].astype(int),
            "patient_id": parts[0],
            "eye": parts[1],
        })
    raise FileNotFoundError(f"no etiquetas.csv or trainLabels.csv under {folder}")


READERS = {"APTOS2019": read_aptos, "EyePACS": read_eyepacs}


def unify(tables):
    df = pd.concat([t.assign(source_dataset=name) for name, t in tables.items()], ignore_index=True)
    df["patient_id"] = df["source_dataset"] + "_" + df["patient_id"].astype(str)
    df["referable"] = (df["icdr_grade"] >= 2).astype(int)
    df["device"] = "desktop_camera"
    return df[["image_id", "source_dataset", "file_path", "icdr_grade", "referable", "patient_id", "eye", "device"]]


##### Split #####
#-----------------------------------------------------------
def split_off(d, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, fold_idx = next(sgkf.split(d, d["icdr_grade"], groups=d["patient_id"]))
    return d.iloc[rest_idx], d.iloc[fold_idx]


def split_by_patient(df, test_folds=7, val_folds=6, seed=42):
    trainval, test = split_off(df, test_folds, seed)
    train, val = split_off(trainval, val_folds, seed)
    return {"train": train, "val": val, "test": test}


##### Preprocessing #####
#-----------------------------------------------------------
def preprocess_image(path, size=384, use_clahe=False, tolerance=10):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > tolerance
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = img
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


##### Dataset #####
#-----------------------------------------------------------
def build_transforms(train):
    if not train:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RetinaDataset(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        img = Image.open(row.prep_path).convert("RGB")
        return self.transform(img), torch.tensor(row.referable, dtype=torch.float32)


##### Prediction #####
#-----------------------------------------------------------
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(device)).float().squeeze(1)).cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


##### Metrics #####
#-----------------------------------------------------------
def wilson(successes, total, z=1.96):
    if total == 0:
        return (float("nan"),) * 3
    p = successes / total
    centre = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def auroc_bootstrap(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            values.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(values, 2.5), np.percentile(values, 97.5)
