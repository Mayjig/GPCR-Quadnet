import os

import pandas as pd
import torch
import torch.utils.data as data


class CachedDTIDataset(data.Dataset):
    def __init__(self, df, ligand_dir, gpcr_dir):
        self.df = df
        self.ligand_dir = ligand_dir
        self.gpcr_dir = gpcr_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ligand_feat = torch.load(os.path.join(self.ligand_dir, f"{index}.pt"))
        gpcr_feat = torch.load(os.path.join(self.gpcr_dir, f"{index}.pt"))
        row = self.df.iloc[index]

        labels = {
            "binding": torch.tensor(row["Y"], dtype=torch.float32),
            "site": optional_label(row["site"]),
            "function": optional_label(row["function"]),
            "modulator": optional_label(row["modulator"]),
        }
        return ligand_feat, gpcr_feat, labels


def optional_label(value):
    if pd.isna(value):
        return torch.tensor(-1, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)
