"""GPCR-Quadnet model and multitask training losses.

The model consumes cached ChemBERTa ligand embeddings and ESM GPCR sequence
embeddings, fuses them with bilinear attention, and predicts four related
binary tasks:

1. binding: binder / non-binder
2. site: orthosteric / allosteric, only for binders
3. function: agonist-like function label, only for orthosteric binders
4. modulator: allosteric modulation label, only for allosteric binders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

try:
    from .ban import BANLayer
except ImportError:
    from ban import BANLayer


TASKS = ("binding", "site", "function", "modulator")
DEFAULT_TASK_WEIGHTS = {
    "binding": 0.30,
    "site": 0.25,
    "function": 0.25,
    "modulator": 0.20,
}
DEFAULT_POS_WEIGHTS = {
    "binding": 1.75,
    "site": 20.0,
    "function": 0.74,
    "modulator": 1.90,
}


class MultiTaskDecoder(nn.Module):
    """Shared decoder with one binary head per GPCR-Quadnet task."""

    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.binding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.site_head = self._make_head(hidden_dim, dropout)
        self.function_head = self._make_head(hidden_dim, dropout)
        self.modulator_head = self._make_head(hidden_dim, dropout)

    @staticmethod
    def _make_head(hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        shared = self.shared_fc(x)
        return {
            "binding": self.binding_head(shared),
            "site": self.site_head(shared),
            "function": self.function_head(shared),
            "modulator": self.modulator_head(shared),
        }


class BinaryFocalLoss(nn.Module):
    """Binary focal loss operating on logits, with optional positive weight."""

    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = (1.0 - pt).pow(self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class GPCRQuadnet(nn.Module):
    """Bilinear-attention multitask model for GPCR ligand classification."""

    def __init__(self, **config):
        super().__init__()
        decoder_cfg = config.get("DECODER", {})
        bcn_cfg = config.get("BCN", {})

        mlp_in_dim = decoder_cfg.get("IN_DIM", 256)
        mlp_hidden_dim = decoder_cfg.get("HIDDEN_DIM", 256)
        ban_heads = bcn_cfg.get("HEADS", 2)

        ligand_dim = config.get("LIGAND_EMBED_DIM", config.get("DRUG", {}).get("NODE_IN_EMBEDDING", 768))
        gpcr_dim = config.get("GPCR_EMBED_DIM", config.get("PROTEIN", {}).get("EMBED_DIM", 1280))

        self.bcn = weight_norm(
            BANLayer(v_dim=ligand_dim, q_dim=gpcr_dim, h_dim=mlp_in_dim, h_out=ban_heads),
            name="h_mat",
            dim=None,
        )
        self.mlp_classifier = MultiTaskDecoder(mlp_in_dim, mlp_hidden_dim)

    @property
    def fusion(self):
        return self.bcn

    @property
    def decoder(self):
        return self.mlp_classifier

    def forward(self, ligand_features, gpcr_features, mode="train"):
        fused_features, attention = self.bcn(ligand_features, gpcr_features)
        scores = self.mlp_classifier(fused_features)

        if mode == "train":
            return None, None, fused_features, scores
        return None, None, scores, attention


# Backward compatibility for scripts that still import the original DrugBAN name.
DrugBAN = GPCRQuadnet


def multitask_loss(
    outputs,
    labels,
    loss_module=None,
    task_weights=None,
    pos_weights=None,
):
    """Compute hierarchical GPCR-Quadnet loss.

    Missing downstream labels are encoded as -1. Downstream tasks are only
    trained where their parent task is positive and the corresponding label
    exists.
    """

    task_weights = task_weights or DEFAULT_TASK_WEIGHTS
    pos_weights = pos_weights or DEFAULT_POS_WEIGHTS

    binding_output = outputs["binding"].view(-1)
    device = binding_output.device
    labels = {key: value.to(device) for key, value in labels.items()}

    binding_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weights["binding"]], device=device),
        reduction="mean",
    )
    site_loss_fn = BinaryFocalLoss(
        gamma=2.0,
        pos_weight=torch.tensor([pos_weights["site"]], device=device),
        reduction="mean",
    )
    function_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weights["function"]], device=device),
        reduction="mean",
    )
    modulator_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weights["modulator"]], device=device),
        reduction="mean",
    )

    loss_binding = binding_loss_fn(binding_output, labels["binding"].float())

    valid_site = (labels["binding"] == 1) & (labels["site"] != -1)
    loss_site = _masked_binary_loss(site_loss_fn, outputs["site"], labels["site"], valid_site)

    orthosteric = (labels["binding"] == 1) & (labels["site"] == 0) & (labels["function"] != -1)
    loss_function = _masked_binary_loss(
        function_loss_fn,
        outputs["function"],
        labels["function"],
        orthosteric,
    )

    allosteric = (labels["binding"] == 1) & (labels["site"] == 1) & (labels["modulator"] != -1)
    loss_modulator = _masked_binary_loss(
        modulator_loss_fn,
        outputs["modulator"],
        labels["modulator"],
        allosteric,
    )

    losses = [loss_binding, loss_site, loss_function, loss_modulator]
    if loss_module is not None:
        total_loss, _ = loss_module(losses)
    else:
        total_loss = sum(task_weights[task] * loss for task, loss in zip(TASKS, losses))

    return total_loss, {task: loss.item() for task, loss in zip(TASKS, losses)}


def _masked_binary_loss(loss_fn, logits, labels, mask):
    if mask.any():
        return loss_fn(logits[mask].view(-1), labels[mask].float())
    return logits.new_tensor(0.0)
