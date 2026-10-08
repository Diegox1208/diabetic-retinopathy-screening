#### created by Diego G. Salas, 2026-10-08
#### adapted from: notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb (Colab, 2026-10-07)
#
# Functions shared by the numbered stages: preprocessing, dataset, prediction, metrics.
# Imported as `import retina` by the scripts in workflow/.

import hashlib
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


##### Timing #####
#-----------------------------------------------------------
class Timer:
    # Prints "<block>\tdone in Ns" when the block ends, the R `t = Sys.time()` idiom.
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
    # MD5 of the sorted image ids: proves a resumed run sees the same split.
    return hashlib.md5("|".join(sorted(map(str, image_ids))).encode()).hexdigest()


##### Preprocessing #####
#-----------------------------------------------------------
def preprocess_image(path, size=384, use_clahe=False, tolerance=10):
    # Crops the black border, pads to a square, resizes; optional CLAHE on the L channel.
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
    # Augmentation on train only; val and test are seen as they are.
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
    # Yields (image tensor, referable label) from a table with columns prep_path, referable.
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
    # Returns (probabilities, labels) as numpy arrays.
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
    # Proportion with its Wilson 95% interval.
    if total == 0:
        return (float("nan"),) * 3
    p = successes / total
    centre = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return p, centre - margin, centre + margin


def auroc_bootstrap(y, p, n=1000, seed=0):
    # AUROC with a percentile bootstrap 95% interval; resamples with one class are skipped.
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            values.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(values, 2.5), np.percentile(values, 97.5)
