#!/usr/bin/env python3
#### created by Diego G. Salas
#### adapted from notebooks/RD_02_APTOS_EyePACS_v3_checkpoint.ipynb, cells "Paso 4" and "Paso 5"
#
# Stage 05 fine-tunes an ImageNet CNN from timm for referable DR, ICDR >= 2, with one sigmoid output.
# Keeps the weights of the epoch with the best validation AUROC. Saves a checkpoint every epoch,
# and a rerun resumes from it only if the split and the parameters are identical.

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import retina

##### Getopts #####
#-----------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--split", required=True, help="split table from stage 04")
parser.add_argument("--model-dir", required=True, help="folder for <version>.pt and <version>_checkpoint.pt")
parser.add_argument("--history", required=True, help="output table of loss and validation AUROC per epoch, TSV")
parser.add_argument("--version", required=True)
parser.add_argument("--architecture", default="efficientnet_b0")
parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1])
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--device", default="auto")
args = parser.parse_args()

if not Path(args.split).is_file():
    print(f"/!\\ Error : {args.split} does not exist; run stage 04 first", file=sys.stderr)
    sys.exit(1)
weights_file = Path(args.model_dir) / f"{args.version}.pt"
ckpt_file = Path(args.model_dir) / f"{args.version}_checkpoint.pt"
if weights_file.is_file():
    print(f"[05_train] {weights_file} exists, skipping; change version to train a new model", file=sys.stderr)
    sys.exit(0)
Path(args.model_dir).mkdir(parents=True, exist_ok=True)

params = {k: v for k, v in vars(args).items() if k not in ("num_workers", "device", "history", "model_dir", "split")}

##### Libraries #####
#-----------------------------------------------------------
retina.set_seed(args.seed)
device = retina.resolve_device(args.device)
use_amp = device.type == "cuda"
print(f"device\t{device}\ttorch {torch.__version__}\ttimm {timm.__version__}")

##### Data files #####
#-----------------------------------------------------------
with retina.Timer("Data files"):
    df = pd.read_csv(args.split, sep="\t", dtype={"image_id": str, "patient_id": str})
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    loader_train = DataLoader(retina.RetinaDataset(train, retina.build_transforms(True)), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=use_amp)
    loader_val = DataLoader(retina.RetinaDataset(val, retina.build_transforms(False)), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    fingerprints = {name: retina.fingerprint(df.loc[df["split"] == name, "image_id"]) for name in ("train", "val", "test")}
    print(f"train {len(train)} images, {len(loader_train)} batches per epoch; val {len(val)} images")

##### Model #####
#-----------------------------------------------------------
model = timm.create_model(args.architecture, pretrained=bool(args.pretrained), num_classes=1).to(device)
print(f"{args.architecture}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader_train))
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

history, best_auc, best_weights, first_epoch = [], -1.0, None, 1
if ckpt_file.is_file():
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    if ckpt["fingerprints"] != fingerprints:
        print("/!\\ Error : the split changed since the checkpoint; delete it or check the data", file=sys.stderr)
        sys.exit(1)
    if ckpt["params"] != params:
        print("/!\\ Error : parameters changed since the checkpoint; use another version or delete it", file=sys.stderr)
        sys.exit(1)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    history, best_auc, best_weights = ckpt["history"], ckpt["best_auc"], ckpt["best_weights"]
    first_epoch = ckpt["epoch"] + 1
    print(f"resuming from epoch {first_epoch}, best validation AUROC so far {best_auc:.4f}")

##### Training #####
#-----------------------------------------------------------
for epoch in range(first_epoch, args.epochs + 1):
    t = time.time()
    model.train()
    loss_sum = 0.0
    for x, y in loader_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x).squeeze(1), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_sum += loss.item() * len(y)
    train_loss = loss_sum / len(loader_train.dataset)

    p_val, y_val = retina.predict(model, loader_val, device)
    val_auc = roc_auc_score(y_val, p_val)
    seconds = round(time.time() - t)
    history.append(dict(epoch=epoch, train_loss=train_loss, val_auroc=val_auc, seconds=seconds))
    mark = ""
    if val_auc > best_auc:
        best_auc, best_weights, mark = val_auc, copy.deepcopy(model.state_dict()), "\tbest"
    print(f"epoch {epoch:2d}/{args.epochs}\ttrain loss {train_loss:.4f}\tval AUROC {val_auc:.4f}\t{seconds}s{mark}", flush=True)

    ##### Checkpoint, written to a temporary file first so an interruption never leaves it corrupt
    tmp = str(ckpt_file) + ".tmp"
    torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
                    best_auc=best_auc, best_weights=best_weights, fingerprints=fingerprints, params=params), tmp)
    os.replace(tmp, ckpt_file)

##### Saving data #####
#-----------------------------------------------------------
torch.save(best_weights, weights_file)
pd.DataFrame(history).assign(version=args.version).to_csv(args.history, sep="\t", index=False)
with open(Path(args.model_dir) / f"{args.version}_params.json", "w") as f:
    json.dump(dict(params=params, fingerprints=fingerprints, best_val_auroc=best_auc), f, indent=2)
print(f"Saved {weights_file} (best validation AUROC {best_auc:.4f}) and {args.history}")
