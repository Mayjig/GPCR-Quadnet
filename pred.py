#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import CachedDTIDataset
from model import GPCRQuadnet
from utils import cached_collate_fn, set_seed


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPCR-Quadnet prediction.")
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "configs", "GPCRQuadnet.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join(BASE_DIR, "results", "best_model.pth"))
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "data", "evaluation_tmp.csv"))
    parser.add_argument("--drug-dir", default=os.path.join(BASE_DIR, "preprocessing", "evaluation", "drug"))
    parser.add_argument("--protein-dir", default=os.path.join(BASE_DIR, "preprocessing", "evaluation", "protein"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "results", "predictions", "predictions.csv"))
    return parser.parse_args()


def strip_dataparallel_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    model = GPCRQuadnet(**cfg).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(strip_dataparallel_prefix(state_dict))
    model.eval()
    print("Loaded model from:", args.checkpoint)

    df_test = pd.read_csv(args.csv)
    test_dataset = CachedDTIDataset(df_test, args.drug_dir, args.protein_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        collate_fn=cached_collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )

    predictions = {task: [] for task in ("binding", "site", "function", "modulator")}
    with torch.no_grad():
        for ligand_features, gpcr_features, _ in test_loader:
            ligand_features = ligand_features.to(device)
            gpcr_features = gpcr_features.to(device)
            _, _, _, outputs = model(ligand_features, gpcr_features)

            for task in predictions:
                probs = torch.sigmoid(outputs[task]).cpu().numpy().reshape(-1)
                predictions[task].extend(probs.tolist())

    for task, values in predictions.items():
        df_test[f"pred_{task}"] = values[: len(df_test)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_test.to_csv(args.output, index=False)
    print("Saved predictions to:", args.output)


if __name__ == "__main__":
    main()
