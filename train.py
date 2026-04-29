import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

from model import multitask_loss
from utils import evaluate_multitask_model


TASKS = ("binding", "site", "function", "modulator")


class Trainer:
    def __init__(
        self,
        model,
        optim,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        opt_da=None,
        discriminator=None,
        experiment=None,
        **config,
    ):
        self.model = model.to(device)
        self.optim = optim
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.experiment = experiment
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0.0
        self.test_metrics = {}

        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.train_task_losses = {task: [] for task in TASKS}
        self.val_task_losses = {task: [] for task in TASKS}
        self.train_task_aurocs = {task: [] for task in TASKS}
        self.val_task_aurocs = {task: [] for task in TASKS}

        self.val_table = PrettyTable(
            ["# Epoch", "AUROC_binding", "AUPRC_binding", "AUROC_site", "AUROC_function", "AUROC_modulator", "Val_loss"]
        )
        self.train_table = PrettyTable(
            ["# Epoch", "AUROC_binding", "AUPRC_binding", "AUROC_site", "AUROC_function", "AUROC_modulator", "Train_loss"]
        )
        self.test_table = PrettyTable(
            [
                "# Epoch",
                "bind_auroc",
                "bind_auprc",
                "bind_f1",
                "bind_sensitivity",
                "bind_specificity",
                "bind_accuracy",
                "bind_precision",
                "binding_threshold",
                "site_auroc",
                "site_auprc",
                "site_f1",
                "site_sensitivity",
                "site_specificity",
                "site_accuracy",
                "site_precision",
                "function_auroc",
                "function_auprc",
                "function_f1",
                "function_sensitivity",
                "function_specificity",
                "function_accuracy",
                "function_precision",
                "modulator_auroc",
                "modulator_auprc",
                "modulator_f1",
                "modulator_sensitivity",
                "modulator_specificity",
                "modulator_accuracy",
                "modulator_precision",
                "test_loss",
            ]
        )

        self._print_model_size()

    def _print_model_size(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")
        print(f"Non-trainable parameters: {num_params - num_trainable:,}")
        print(f"Model size: {param_size / 1024**2:.2f} MB")

    def train(self, scheduler=None):
        for _ in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            self.train_loss_epoch.append(train_loss)
            self._log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)

            val_metrics = self.test(dataloader="val")
            train_metrics = self.test(dataloader="train")
            self._record_epoch_metrics("val", val_metrics)
            self._record_epoch_metrics("train", train_metrics)
            self._add_epoch_rows(train_metrics, val_metrics, train_loss)

            if scheduler is not None:
                scheduler.step(val_metrics["loss"])

            if val_metrics["binding_auroc"] >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_auroc = val_metrics["binding_auroc"]
                self.best_epoch = self.current_epoch

            self._print_validation_summary(val_metrics)
            if self.current_epoch % 5 == 0 or self.current_epoch == self.epochs:
                test_metrics = self.test(dataloader="test")
                self.test_table.add_row(self._format_test_row(test_metrics, self.current_epoch))
                self._save_plots()
                if self.config["RESULT"]["SAVE_MODEL"]:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))

        final_metrics = self.test(dataloader="test")
        self.test_table.add_row(self._format_test_row(final_metrics, self.best_epoch))
        self.test_metrics.update(final_metrics)
        self.test_metrics["best_epoch"] = self.best_epoch
        self.save_result()
        return self.test_metrics

    def train_epoch(self):
        self.model.train()
        task_loss_totals = {task: 0.0 for task in TASKS}
        total_loss = 0.0

        for ligand_features, gpcr_features, labels in tqdm(self.train_dataloader, mininterval=600):
            self.step += 1
            ligand_features = ligand_features.to(self.device)
            gpcr_features = gpcr_features.to(self.device)
            labels = {key: value.to(self.device) for key, value in labels.items()}

            self.optim.zero_grad()
            _, _, _, outputs = self.model(ligand_features, gpcr_features)
            loss, loss_dict = multitask_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()

            total_loss += loss.item()
            for task in TASKS:
                task_loss_totals[task] += loss_dict[task]

            self._log_metric("train_step total loss", loss.item(), step=self.step)
            for task, value in loss_dict.items():
                self._log_metric(f"train_step {task} loss", value, step=self.step)

        num_batches = max(1, len(self.train_dataloader))
        for task in TASKS:
            self.train_task_losses[task].append(task_loss_totals[task] / num_batches)
        return total_loss / num_batches

    def test(self, dataloader="test"):
        data_loader = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader,
        }[dataloader]

        model_to_eval = self.model
        if dataloader == "test" and self.best_model is not None:
            model_to_eval = copy.deepcopy(self.model)
            model_to_eval.load_state_dict(self.best_model)

        model_to_eval.eval()
        task_loss_totals = {task: 0.0 for task in TASKS}
        total_loss = 0.0
        y_true = {task: [] for task in TASKS}
        y_pred = {task: [] for task in TASKS}

        with torch.no_grad():
            for ligand_features, gpcr_features, labels in data_loader:
                ligand_features = ligand_features.to(self.device)
                gpcr_features = gpcr_features.to(self.device)
                labels = {key: value.to(self.device) for key, value in labels.items()}

                _, _, _, outputs = model_to_eval(ligand_features, gpcr_features)
                loss, loss_dict = multitask_loss(outputs, labels)
                total_loss += loss.item()
                for task in TASKS:
                    task_loss_totals[task] += loss_dict[task]

                self._collect_predictions(outputs, labels, y_true, y_pred)

        num_batches = max(1, len(data_loader))
        metrics = self._score_predictions(y_true, y_pred)
        metrics["loss"] = total_loss / num_batches
        for task in TASKS:
            metrics[f"{task}_loss"] = task_loss_totals[task] / num_batches

        if dataloader == "test":
            metrics.update(evaluate_multitask_model(model_to_eval, data_loader, self.device))
            self.test_metrics.update(metrics)
            self._log_test_metrics(metrics)
            self._print_test_metrics(metrics)

        return metrics

    @staticmethod
    def _collect_predictions(outputs, labels, y_true, y_pred):
        masks = {
            "binding": torch.ones_like(labels["binding"], dtype=torch.bool),
            "site": (labels["binding"] == 1) & (labels["site"] != -1),
            "function": (labels["binding"] == 1) & (labels["site"] == 0) & (labels["function"] != -1),
            "modulator": (labels["binding"] == 1) & (labels["site"] == 1) & (labels["modulator"] != -1),
        }
        for task, mask in masks.items():
            if mask.any():
                logits = outputs[task][mask].view(-1)
                y_true[task].extend(labels[task][mask].cpu().numpy())
                y_pred[task].extend(torch.sigmoid(logits).cpu().numpy())

    @staticmethod
    def _score_predictions(y_true, y_pred):
        metrics = {}
        for task in TASKS:
            labels = np.array(y_true[task])
            preds = np.array(y_pred[task])
            metrics[f"{task}_auroc"], metrics[f"{task}_auprc"] = safe_auc_pr(labels, preds)

        labels = np.array(y_true["binding"])
        preds = np.array(y_pred["binding"])
        metrics["binding_optimal_threshold"] = optimal_threshold(labels, preds)
        return metrics

    def _record_epoch_metrics(self, split, metrics):
        target_aurocs = self.val_task_aurocs if split == "val" else self.train_task_aurocs

        if split == "val":
            self.val_loss_epoch.append(metrics["loss"])
            for task in TASKS:
                self.val_task_losses[task].append(metrics[f"{task}_loss"])

        for task in TASKS:
            target_aurocs[task].append(metrics[f"{task}_auroc"])

    def _add_epoch_rows(self, train_metrics, val_metrics, train_loss):
        float2str = lambda x: f"{x:.4f}"
        self.train_table.add_row(
            ["epoch " + str(self.current_epoch)]
            + list(
                map(
                    float2str,
                    [
                        train_metrics["binding_auroc"],
                        train_metrics["binding_auprc"],
                        train_metrics["site_auroc"],
                        train_metrics["function_auroc"],
                        train_metrics["modulator_auroc"],
                        train_loss,
                    ],
                )
            )
        )
        self.val_table.add_row(
            ["epoch " + str(self.current_epoch)]
            + list(
                map(
                    float2str,
                    [
                        val_metrics["binding_auroc"],
                        val_metrics["binding_auprc"],
                        val_metrics["site_auroc"],
                        val_metrics["function_auroc"],
                        val_metrics["modulator_auroc"],
                        val_metrics["loss"],
                    ],
                )
            )
        )

    def _print_validation_summary(self, metrics):
        print(
            f"Binding Validation at Epoch {self.current_epoch} with validation loss {metrics['binding_loss']} "
            f"AUROC {metrics['binding_auroc']} AUPRC {metrics['binding_auprc']}"
        )
        print(f"Site Validation at Epoch {self.current_epoch} with validation loss {metrics['site_loss']} AUROC {metrics['site_auroc']}")
        print(
            f"Function Validation at Epoch {self.current_epoch} with validation loss {metrics['function_loss']} "
            f"AUROC {metrics['function_auroc']}"
        )
        print(
            f"Allosteric Validation at Epoch {self.current_epoch} with validation loss {metrics['modulator_loss']} "
            f"AUROC {metrics['modulator_auroc']}"
        )

    def _save_plots(self):
        plt.rcParams.update({
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 20,
        })

        self._plot_lines(
            {"Training Loss": self.train_loss_epoch, "Validation Loss": self.val_loss_epoch},
            "Loss",
            "Training and Validation Loss Curve",
            f"loss_curve_{self.current_epoch}.png",
        )
        self._plot_lines(
            {f"{task} Loss": self.train_task_losses[task] for task in TASKS},
            "Loss",
            "Training Loss per Task",
            f"taskwise_train_loss_curve_{self.current_epoch}.png",
        )
        self._plot_lines(
            {f"{task} Loss": self.val_task_losses[task] for task in TASKS},
            "Loss",
            "Validation Loss per Task",
            f"taskwise_val_loss_curve_{self.current_epoch}.png",
        )
        self._plot_lines(
            {f"{task} AUC": self.train_task_aurocs[task] for task in TASKS},
            "ROC-AUC",
            "Training ROC-AUC per Task",
            f"taskwise_train_auroc_curve_{self.current_epoch}.png",
            legend_loc="lower right",
        )
        self._plot_lines(
            {f"{task} AUC": self.val_task_aurocs[task] for task in TASKS},
            "ROC-AUC",
            "Validation ROC-AUC per Task",
            f"taskwise_val_auroc_curve_{self.current_epoch}.png",
            legend_loc="lower right",
        )

    def _plot_lines(self, series, ylabel, title, filename, legend_loc="upper right"):
        plt.figure(figsize=(10, 8))
        for label, values in series.items():
            plt.plot(range(1, len(values) + 1), values, label=label)
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.title(title, fontsize=20)
        plt.legend(fontsize=18, loc=legend_loc)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model, os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))

        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
        }
        torch.save(state, os.path.join(self.output_dir, "result_metrics.pt"))
        self._write_table("valid_markdowntable.txt", self.val_table)
        self._write_table("test_markdowntable.txt", self.test_table)
        self._write_table("train_markdowntable.txt", self.train_table)

    def _write_table(self, filename, table):
        with open(os.path.join(self.output_dir, filename), "w") as fp:
            fp.write(table.get_string())

    def _format_test_row(self, metrics, epoch):
        return ["epoch " + str(epoch)] + [
            f'{metrics["binding_auroc"]:.4f}',
            f'{metrics["binding_auprc"]:.4f}',
            f'{metrics["binding_f1"]:.4f}',
            f'{metrics["binding_sensitivity"]:.4f}',
            f'{metrics["binding_specificity"]:.4f}',
            f'{metrics["binding_accuracy"]:.4f}',
            f'{metrics["binding_precision"]:.4f}',
            f'{metrics["binding_optimal_threshold"]:.4f}',
            f'{metrics["site_auroc"]:.4f}',
            f'{metrics["site_auprc"]:.4f}',
            f'{metrics["site_f1"]:.4f}',
            f'{metrics["site_sensitivity"]:.4f}',
            f'{metrics["site_specificity"]:.4f}',
            f'{metrics["site_accuracy"]:.4f}',
            f'{metrics["site_precision"]:.4f}',
            f'{metrics["function_auroc"]:.4f}',
            f'{metrics["function_auprc"]:.4f}',
            f'{metrics["function_f1"]:.4f}',
            f'{metrics["function_sensitivity"]:.4f}',
            f'{metrics["function_specificity"]:.4f}',
            f'{metrics["function_accuracy"]:.4f}',
            f'{metrics["function_precision"]:.4f}',
            f'{metrics["modulator_auroc"]:.4f}',
            f'{metrics["modulator_auprc"]:.4f}',
            f'{metrics["modulator_f1"]:.4f}',
            f'{metrics["modulator_sensitivity"]:.4f}',
            f'{metrics["modulator_specificity"]:.4f}',
            f'{metrics["modulator_accuracy"]:.4f}',
            f'{metrics["modulator_precision"]:.4f}',
            f'{metrics["loss"]:.4f}',
        ]

    def _log_metric(self, name, value, **kwargs):
        if self.experiment:
            self.experiment.log_metric(name, value, **kwargs)

    def _log_test_metrics(self, metrics):
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            for metric, value in metrics.items():
                self.experiment.log_metric(f"test_{metric}", value)

    def _print_test_metrics(self, metrics):
        print(f"Test metrics at current epoch ({self.best_epoch}):")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


def safe_auc_pr(labels, preds):
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.0, 0.0
    return roc_auc_score(labels, preds), average_precision_score(labels, preds)


def optimal_threshold(labels, preds):
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, preds)
    return thresholds[np.argmax(tpr - fpr)]
