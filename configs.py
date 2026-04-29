from yacs.config import CfgNode as CN


_C = CN()

# Ligand feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_EMBEDDING = 768

# GPCR sequence feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.ESM_MODEL = "esm2_t33_650M_UR50D"
_C.PROTEIN.EMBED_DIM = 1280

# Bilinear attention fusion
_C.BCN = CN()
_C.BCN.HEADS = 2

# Multitask decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.MULTI_TASK = True

# Solver
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.SEED = 2048

# Results
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./results"
_C.RESULT.SAVE_MODEL = True

# Comet config
_C.COMET = CN()
_C.COMET.WORKSPACE = "pz-white"
_C.COMET.PROJECT_NAME = "GPCR-Quadnet"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()
