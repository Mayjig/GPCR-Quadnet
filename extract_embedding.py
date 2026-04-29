#!/usr/bin/env python
# coding: utf-8

import os

import esm
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

csv_path = os.path.join(BASE_DIR, "data", "evaluation_tmp.csv")
#drug_dir = os.path.join(BASE_DIR, "preprocessing", "training", "drug_train")
#drug_dir = os.path.join(BASE_DIR, "preprocessing", "training", "drug_valid")
#drug_dir = os.path.join(BASE_DIR, "preprocessing", "training", "drug_test")
drug_dir = os.path.join(BASE_DIR, "preprocessing", "evaluation", "drug")
#protein_dir = os.path.join(BASE_DIR, "preprocessing", "training", "protein_train")
#protein_dir = os.path.join(BASE_DIR, "preprocessing", "training", "protein_valid")
#protein_dir = os.path.join(BASE_DIR, "preprocessing", "training", "protein_test")
protein_dir = os.path.join(BASE_DIR, "preprocessing", "evaluation", "protein")
chemberta_dir = os.path.join(PROJECT_DIR, "chemberta")

os.makedirs(drug_dir, exist_ok=True)
os.makedirs(protein_dir, exist_ok=True)


tokenizer = RobertaTokenizer.from_pretrained(chemberta_dir)
chemberta = RobertaModel.from_pretrained(chemberta_dir).eval().cuda()

esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
esm_model.eval().cuda()
batch_converter = alphabet.get_batch_converter()

df = pd.read_csv(csv_path)

for index in tqdm(range(len(df)), mininterval=60):
    smiles = df.iloc[index]["SMILES"]
    protein = df.iloc[index]["Protein"]

    drug_path = os.path.join(drug_dir, f"{index}.pt")
    protein_path = os.path.join(protein_dir, f"{index}.pt")

    if os.path.exists(drug_path) and os.path.exists(protein_path):
        continue

    if not os.path.exists(drug_path):
        chem_inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        chem_inputs = {key: value.cuda() for key, value in chem_inputs.items()}
        with torch.no_grad():
            chem_output = chemberta(**chem_inputs).last_hidden_state.cpu()
        torch.save(chem_output.squeeze(0), drug_path)

    if not os.path.exists(protein_path):
        sequence_batch = [(f"seq{index}", protein)]
        _, _, tokens = batch_converter(sequence_batch)
        tokens = tokens.cuda()
        with torch.no_grad():
            esm_output = esm_model(tokens, repr_layers=[33])["representations"][33][:, 1:-1].cpu()
        torch.save(esm_output.squeeze(0), protein_path)

    torch.cuda.empty_cache()
