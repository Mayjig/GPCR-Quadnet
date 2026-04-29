import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def set_seed(seed=1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cached_collate_fn(batch):
    ligand_batch, gpcr_batch, label_batch = zip(*batch)
    ligand_tensor = pad_sequence(ligand_batch, batch_first=True)
    gpcr_tensor = pad_sequence(gpcr_batch, batch_first=True)
    label_dict = {key: torch.stack([lb[key] for lb in label_batch]) for key in label_batch[0].keys()}
    return ligand_tensor, gpcr_tensor, label_dict

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_multitask_model(model, dataloader, device):
    model.eval()

    all_preds = {task: [] for task in ['binding', 'site', 'function', 'modulator']}
    all_labels = {task: [] for task in ['binding', 'site', 'function', 'modulator']}

    metrics = {
        'binding': {'correct': 0, 'total': 0},
        'site': {'correct': 0, 'total': 0},
        'function': {'correct': 0, 'total': 0},
        'modulator': {'correct': 0, 'total': 0}
    }

    with torch.no_grad():
        for ligand_features, gpcr_features, labels in dataloader:
            ligand_features = ligand_features.to(device)
            gpcr_features = gpcr_features.to(device)

            for key in labels:
                labels[key] = labels[key].to(device)

            _, _, _, outputs = model(ligand_features, gpcr_features)

            binding_preds = torch.sigmoid(outputs['binding'].view(-1)) > 0.5
            binding_true = labels['binding']
            metrics['binding']['correct'] += (binding_preds == binding_true).sum().item()
            metrics['binding']['total'] += binding_true.size(0)
            all_preds['binding'].extend(binding_preds.cpu().numpy())
            all_labels['binding'].extend(binding_true.cpu().numpy())

            site = (binding_true == 1) & (labels['site'] != -1)
            if site.sum() > 0:
                site_preds = torch.sigmoid(outputs['site'][site].view(-1)) > 0.5
                site_true = labels['site'][site]
                metrics['site']['correct'] += (site_preds == site_true).sum().item()
                metrics['site']['total'] += site.sum().item()
                all_preds['site'].extend(site_preds.cpu().numpy())
                all_labels['site'].extend(site_true.cpu().numpy())

            ortho = (binding_true == 1) & (labels['site'] == 0) & (labels['function'] != -1)
            if ortho.sum() > 0:
                func_preds = torch.sigmoid(outputs['function'][ortho].view(-1)) > 0.5
                func_true = labels['function'][ortho]
                metrics['function']['correct'] += (func_preds == func_true).sum().item()
                metrics['function']['total'] += ortho.sum().item()
                all_preds['function'].extend(func_preds.cpu().numpy())
                all_labels['function'].extend(func_true.cpu().numpy())

            allo = (binding_true == 1) & (labels['site'] == 1) & (labels['modulator'] != -1)
            if allo.sum() > 0:
                mod_preds = torch.sigmoid(outputs['modulator'][allo].view(-1)) > 0.5
                mod_true = labels['modulator'][allo]
                metrics['modulator']['correct'] += (mod_preds == mod_true).sum().item()
                metrics['modulator']['total'] += allo.sum().item()
                all_preds['modulator'].extend(mod_preds.cpu().numpy())
                all_labels['modulator'].extend(mod_true.cpu().numpy())


    metrics_dict = {}
    for task in metrics:
        if metrics[task]['total'] > 0:
            acc = metrics[task]['correct'] / metrics[task]['total']
            metrics_dict[f'{task}_accuracy'] = acc

            y_true = np.array(all_labels[task])
            y_pred = np.array(all_preds[task])

            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[False, True])
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp + 1e-6)

            metrics_dict[f'{task}_f1'] = f1
            metrics_dict[f'{task}_precision'] = precision
            metrics_dict[f'{task}_sensitivity'] = recall
            metrics_dict[f'{task}_specificity'] = specificity
        else:
            for k in ['accuracy', 'f1', 'precision', 'sensitivity', 'specificity']:
                metrics_dict[f'{task}_{k}'] = 0.0

    return metrics_dict
