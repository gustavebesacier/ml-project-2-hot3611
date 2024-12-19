import os
import numpy as np
import segmentation_models_pytorch as smp
import torch
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import matplotlib.pyplot as plt
import utils.helpers
from utils.variables.env_variables import set_device
from utils.variables.constants import *
import utils.variables.path_variables

PLOT_DATASET = False

bce_loss = nn.BCEWithLogitsLoss()
jaccard_loss = smp.losses.JaccardLoss(mode='binary')

def loss_fn(outputs, targets):
    """
    Combines BCEWithLogitsLoss and Jaccard Loss for training.

    @param outputs: Model predictions.
    @param targets: Ground truth masks.
    @return: Combined loss value.
    """
    return bce_loss(outputs, targets) + jaccard_loss(outputs, targets)


def accuracy(outputs, masks):
    """
    Computes accuracy based on model outputs and ground truth masks.

    @param outputs: Model predictions.
    @param masks: Ground truth masks.
    @return: Accuracy as a float value.
    """
    outputs = torch.sigmoid(outputs) > 0.5
    correct = (outputs == masks).float().sum()  # Count correct predictions
    total = torch.numel(masks)  # Total number of elements in masks
    return (correct / total).item()


def f1_metric(output, masks):
    """
    Computes F1 score using segmentation-models-pytorch utility functions.

    @param output: Model predictions.
    @param masks: Ground truth masks.
    @return: F1 score as a float value.
    """
    tp, fp, fn, tn = smp.metrics.get_stats(output, masks.int(), mode="binary")
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

METRICS = {
    'ACC': accuracy,
    'F1': f1_metric
}

LOG_HEADERS = ['epoch', 'loss', 'acc', 'f1']

class MyUNet(smp.Unet):
    def __init__(self, encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels,classes=classes)



def plot_loss_training(losses, losses2=None, title=None, label_l1=None, label_l2=None, show=True):
    """
    Plots training and validation losses per epoch.

    @param losses: List of training losses.
    @param losses2: List of validation losses (optional).
    @param title: Plot title (optional).
    @param label_l1: Label for training loss (optional).
    @param label_l2: Label for validation loss (optional).
    @param show: Whether to display the plot. Default is True.
    """
    os.makedirs(utils.variables.path_variables.PARAMETERS_UNET['graph_path'], exist_ok=True)
    title_fig = os.path.join(utils.variables.path_variables.PARAMETERS_UNET['graph_path'], f"{title}.pdf")

    label_l1 = label_l1 if label_l1 else "Loss"
    plt.plot(range(len(losses)), losses, label=label_l1, color=LOSS_COLOR_1)

    if losses2:
        label_l2 = label_l2 if label_l2 else "Validation Loss"
        plt.plot(range(len(losses2)), losses2, label=label_l2, color=LOSS_COLOR_2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title(title if title else "Training Loss per Epoch")
    plt.legend(fontsize=10)
    plt.grid()
    plt.savefig(title_fig, dpi=1000)
    if show:
        plt.show(block=False)


def plot_all_metrics(loss_train, loss_val, acc_train, acc_val, f1_train, f1_val, epoch=None, show=True):
    """
    Plots training and validation metrics including loss, accuracy, and F1 score.

    @param loss_train: List of training losses.
    @param loss_val: List of validation losses.
    @param acc_train: List of training accuracies.
    @param acc_val: List of validation accuracies.
    @param f1_train: List of training F1 scores.
    @param f1_val: List of validation F1 scores.
    @param epoch: Current epoch number (optional).
    @param show: Whether to display the plot. Default is True.
    """
    os.makedirs(utils.variables.path_variables.PARAMETERS_UNET['graph_path'], exist_ok=True)
    title_fig = os.path.join(utils.variables.path_variables.PARAMETERS_UNET['graph_path'], f"Metrics_Epoch_{epoch}.pdf")

    loss_train = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in loss_train]
    loss_val = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in loss_val]
    acc_train = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in acc_train]
    acc_val = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in acc_val]
    f1_train = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in f1_train]
    f1_val = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in f1_val]


    fig, ax1 = plt.subplots()

    ax1.plot(range(len(loss_train)), loss_train, LOSS_COLOR_1, label="Train Loss")
    ax1.plot(range(len(loss_val)), loss_val, LOSS_COLOR_2, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')

    # Plot accuracy and F1 score on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(acc_train)), acc_train, ACC_COLOR, label="Train Accuracy")
    ax2.plot(range(len(acc_val)), acc_val, ACC_COLOR, linestyle="--", label="Validation Accuracy")
    ax2.plot(range(len(f1_train)), f1_train, F1_COLOR, label="Train F1 Score")
    ax2.plot(range(len(f1_val)), f1_val, F1_COLOR, linestyle="--", label="Validation F1 Score")
    ax2.set_ylabel("Accuracy and F1")

    # Combine legends from both y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    fig.legend(lines, labels, loc='lower center', ncol=6, fontsize=8)

    plt.title("Model Training Metrics")
    plt.tight_layout()
    plt.savefig(title_fig, dpi=1000)
    if show:
        plt.show(block=False)


class WarmupThenCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate scheduler combining warmup and cosine annealing.

    @param optimizer: Optimizer to apply the scheduler.
    @param warmup_scheduler: Warmup scheduler instance.
    @param cosine_scheduler: Cosine annealing scheduler instance.
    @param num_warmup_steps: Number of warmup steps.
    """

    def __init__(self, optimizer, warmup_scheduler, cosine_scheduler, num_warmup_steps):
        self.warmup_scheduler = warmup_scheduler
        self.cosine_scheduler = cosine_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.step_count = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.step_count < self.num_warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        return self.cosine_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.step_count < self.num_warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step(epoch)
        self.step_count += 1


def get_lr_scheduler(len_dataset: int, optimizer, n_epochs:int) -> WarmupThenCosineScheduler:
    restart = utils.variables.path_variables.PARAMETERS_UNET['lr_sched_restart']
    warmup = utils.variables.path_variables.PARAMETERS_UNET['lr_sched_warmup']

    num_training_steps = n_epochs * len_dataset
    T_0 = int(np.floor(n_epochs * restart))
    T_mult = n_epochs
    num_warmup_steps = int(np.floor(warmup * num_training_steps))

    # 1. Warmup scheduler
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # 2. Cosine scheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=utils.variables.path_variables.PARAMETERS_UNET["lr_sched_min_eta"]
    )

    # 3. Learning rate scheduler object
    lr_scheduler = WarmupThenCosineScheduler(
        optimizer,
        warmup_scheduler,
        cosine_scheduler,
        num_warmup_steps
    )

    return lr_scheduler


