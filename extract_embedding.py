#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import esm
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cached ChemBERTa and ESM2 embeddings.")
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "data", "evaluation_tmp.csv"))
    parser.add_argument("--drug-dir", default=os.path.join(BASE_DIR, "preprocessing", "evaluation", "drug"))
    parser.add_argument("--protein-dir", default=os.path.join(BASE_DIR, "preprocessing", "evaluation", "protein"))
    parser.add_argument("--chemberta-dir", default=os.path.join(BASE_DIR, "chemberta"))
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.drug_dir, exist_ok=True)
    os.makedirs(args.protein_dir, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.chemberta_dir)
    chemberta = RobertaModel.from_pretrained(args.chemberta_dir).eval().to(device)

    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm_model)
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    df = pd.read_csv(args.csv)
    for index in tqdm(range(len(df)), mininterval=60):
        smiles = df.iloc[index]["SMILES"]
        protein = df.iloc[index]["Protein"]

        drug_path = os.path.join(args.drug_dir, f"{index}.pt")
        protein_path = os.path.join(args.protein_dir, f"{index}.pt")

        if not os.path.exists(drug_path):
            chem_inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
            chem_inputs = {key: value.to(device) for key, value in chem_inputs.items()}
            with torch.no_grad():
                chem_output = chemberta(**chem_inputs).last_hidden_state.cpu()
            torch.save(chem_output.squeeze(0), drug_path)

        if not os.path.exists(protein_path):
            sequence_batch = [(f"seq{index}", protein)]
            _, _, tokens = batch_converter(sequence_batch)
            tokens = tokens.to(device)
            with torch.no_grad():
                esm_output = esm_model(tokens, repr_layers=[33])["representations"][33][:, 1:-1].cpu()
            torch.save(esm_output.squeeze(0), protein_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
