#!/usr/bin/env python
# coding: utf-8

import os
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import CachedDTIDataset
from model import GPCRQuadnet
from train import Trainer
from utils import cached_collate_fn, mkdir, set_seed


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_loader(dataset, cfg, shuffle, drop_last):
    params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": drop_last,
        "collate_fn": cached_collate_fn,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if cfg.SOLVER.NUM_WORKERS > 0:
        params["prefetch_factor"] = 4
    return DataLoader(dataset, **params)


def main():
    cfg_path = os.path.join(BASE_DIR, "configs", "GPCRQuadnet.yaml")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.RESULT.OUTPUT_DIR = os.path.join(BASE_DIR, "results", "training")
    cfg.freeze()

    warnings.filterwarnings("ignore")
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Config yaml: {cfg_path}")
    print(f"Running on: {device}")

    data_dir = os.path.join(BASE_DIR, "data")
    embedding_dir = os.path.join(BASE_DIR, "preprocessing", "training")

    df_train = pd.read_csv(os.path.join(data_dir, "train_tmp.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "valid_tmp.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test_tmp.csv"))

    train_dataset = CachedDTIDataset(
        df_train,
        os.path.join(embedding_dir, "drug_train"),
        os.path.join(embedding_dir, "protein_train"),
    )
    val_dataset = CachedDTIDataset(
        df_val,
        os.path.join(embedding_dir, "drug_valid"),
        os.path.join(embedding_dir, "protein_valid"),
    )
    test_dataset = CachedDTIDataset(
        df_test,
        os.path.join(embedding_dir, "drug_test"),
        os.path.join(embedding_dir, "protein_test"),
    )

    train_loader = make_loader(train_dataset, cfg, shuffle=True, drop_last=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False, drop_last=False)
    test_loader = make_loader(test_dataset, cfg, shuffle=False, drop_last=False)

    model = GPCRQuadnet(**cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=0.01)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, optim, device, train_loader, val_loader, test_loader, experiment=None, **cfg)
    trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
