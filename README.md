# GPCR-Quadnet

GPCR-Quadnet is a multitask ligand-GPCR classification model. It combines cached
ChemBERTa ligand embeddings and ESM2 GPCR sequence embeddings with a bilinear
attention fusion layer, then predicts four related tasks:

- `binding`: binder / non-binder
- `site`: orthosteric / allosteric, evaluated for binders
- `function`: functional activity label for orthosteric binders
- `modulator`: allosteric modulation label for allosteric binders

## Repository Layout

```text
.
├── configs/GPCRQuadnet.yaml        # training and model configuration
├── data/                           # CSV input files and small template/demo CSVs
├── preprocessing/                  # cached ChemBERTa and ESM2 embeddings
├── results/                        # trained checkpoints, predictions, and logs
├── ban.py                          # bilinear attention network layer
├── model.py                        # GPCRQuadnet model and multitask loss
├── extract_embedding.py            # ChemBERTa/ESM2 embedding extraction
├── run_demo.py                     # training/evaluation entry point
├── pred.py                         # prediction entry point
├── env.yml                         # conda environment
└── README.md
```

## Environment

Create the conda environment from `env.yml`:

```bash
conda env create -f env.yml
conda activate gpcr_quadnet
```

The environment includes PyTorch, CUDA-related packages, RDKit, scikit-learn,
`fair-esm`, `transformers`, and other dependencies used by the preprocessing,
training, and prediction scripts.

If your CUDA version differs from the one in `env.yml`, install the PyTorch build
that matches your system before running training.

## External Models

### ESM2

The project uses `esm2_t33_650M_UR50D` through the `fair-esm` package:

```python
esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
```

The first run may download the ESM2 weights automatically to the local cache.

### ChemBERTa

`extract_embedding.py` expects a local ChemBERTa model directory by default:

```text
chemberta/
├── config.json
├── merges.txt
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.json
```

Place this directory at the repository root, or pass a custom path:

```bash
python extract_embedding.py --chemberta-dir /path/to/chemberta
```

## Data Format

Input CSV files should contain the following columns:

| Column | Description |
| --- | --- |
| `SMILES` | ligand SMILES string |
| `Protein` | GPCR amino-acid sequence |
| `Y` | binding label, `0.0` or `1.0` |
| `site` | site label, `0.0` orthosteric, `1.0` allosteric, `-1.0` missing |
| `function` | function label for orthosteric binders, `-1.0` missing |
| `modulator` | allosteric modulation label, `-1.0` missing |

The `*_tmp.csv` files in `data/` are small template/demo files for quick checks:

- `train_tmp.csv`
- `valid_tmp.csv`
- `test_tmp.csv`
- `evaluation_tmp.csv`

For a new dataset, keep the same columns and use `-1.0` for unavailable
downstream labels.

## Generate Embeddings

Before training or prediction, generate cached embeddings from the CSV files.
Each row creates one ligand embedding and one protein embedding with matching
integer filenames such as `0.pt`, `1.pt`, and so on.

Training split:

```bash
python extract_embedding.py \
  --csv data/train_tmp.csv \
  --drug-dir preprocessing/training/drug_train \
  --protein-dir preprocessing/training/protein_train
```

Validation split:

```bash
python extract_embedding.py \
  --csv data/valid_tmp.csv \
  --drug-dir preprocessing/training/drug_valid \
  --protein-dir preprocessing/training/protein_valid
```

Test split:

```bash
python extract_embedding.py \
  --csv data/test_tmp.csv \
  --drug-dir preprocessing/training/drug_test \
  --protein-dir preprocessing/training/protein_test
```

Prediction/evaluation split:

```bash
python extract_embedding.py \
  --csv data/evaluation_tmp.csv \
  --drug-dir preprocessing/evaluation/drug \
  --protein-dir preprocessing/evaluation/protein
```

## Train

`run_demo.py` trains GPCR-Quadnet using:

- `data/train_tmp.csv`
- `data/valid_tmp.csv`
- `data/test_tmp.csv`
- embeddings from `preprocessing/training/`
- model settings from `configs/GPCRQuadnet.yaml`

Run:

```bash
python run_demo.py
```

Training outputs are written to:

```text
results/training/
```

This includes checkpoints, metric tables, model architecture text, and loss/AUROC
plots.

## Prediction

The default prediction command uses:

- checkpoint: `results/best_model.pth`
- CSV: `data/evaluation_tmp.csv`
- embeddings: `preprocessing/evaluation/drug` and `preprocessing/evaluation/protein`
- output: `results/predictions/predictions.csv`

Run:

```bash
python pred.py
```

Or specify custom files:

```bash
python pred.py \
  --checkpoint results/best_model.pth \
  --csv data/evaluation_tmp.csv \
  --drug-dir preprocessing/evaluation/drug \
  --protein-dir preprocessing/evaluation/protein \
  --output results/predictions/predictions.csv
```

The output CSV appends:

- `pred_binding`
- `pred_site`
- `pred_function`
- `pred_modulator`

## Notes

- `model.py` keeps `DrugBAN = GPCRQuadnet` as a compatibility alias for older
  checkpoints/scripts, but the model in this repository is GPCR-Quadnet.
- Large CSV, `.pt`, and `.pth` files are tracked with Git LFS.
- If you use different embedding models, update `LIGAND_EMBED_DIM`,
  `PROTEIN.EMBED_DIM`, or the related model configuration accordingly.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
and [NOTICE](NOTICE).
