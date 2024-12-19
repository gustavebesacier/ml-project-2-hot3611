import transfer.transfer_learning
from utils.variables.transformers_variables import PARAMETERS_TRANSFORMERS
import utils.helpers
from utils.variables.constants import DEFAULT_NUM_EPOCHS, LOG_SAVE_INTERVAL, DEFAULT_LOGS_PATH, DEFAULT_WEIGHTS_PATH, DEFAULT_DEVICE, DEFAULT_PLOT_DATASET, DEFAULT_LR_SCHEDULER_TYPE
from transformers import get_scheduler
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import json
import os


PLOT_DATASET = False

def accuracy_f1_transformer(prediction, truths, dim_=False):
    """
    Compute accuracy and F1 score for predictions and ground truths.

    @param prediction: Model predictions.
    @param truths: Ground truth values.
    @param dim_: True if the model output is a vector of the same dimension as the ground truth.
    @return: (accuracy, F1 score)
    """
    if not dim_:
        truths = truths.cpu().numpy().flatten()
        predicted = prediction.cpu().numpy().flatten()
        acc = (truths == predicted).mean()
        f1 = f1_score(truths, predicted, average=PARAMETERS_TRANSFORMERS['average_f1'])
    else:
        truths = truths.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        acc = (truths == prediction).mean()
        f1 = f1_score(truths, prediction, average=PARAMETERS_TRANSFORMERS['average_f1'])

    return acc, f1


def train_model(
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion,
    num_epochs=DEFAULT_NUM_EPOCHS,
    save=True,
    interplot=False,
    plot=True,
    name=False,
    lr_scheduler=None,
    dim_=False,
    device=DEFAULT_DEVICE,
):
    """
    Train a transformer model.

    @param train_loader: DataLoader for the training dataset.
    @param val_loader: DataLoader for the validation dataset.
    @param model: The model to train (e.g., Vision Transformers).
    @param optimizer: Optimizer for the model.
    @param criterion: Loss function.
    @param num_epochs: Number of training epochs.
    @param save: Whether to save model weights and metrics during training.
    @param interplot: Whether to plot metrics at intermediate epochs (for debugging).
    @param plot: Whether to plot and save final metrics after training.
    @param name: Name for saving model weights and logs.
    @param lr_scheduler: Learning rate scheduler.
    @param dim_: True if model output is a vector matching ground truth shape.
    @param device: Device for training (CPU/GPU).
    @return: Training and validation metrics (loss, accuracy, F1 score).
    """
    # Ensure model is on the correct device
    model.to(device)

    # Configure learning rate scheduler
    if lr_scheduler == DEFAULT_LR_SCHEDULER_TYPE:
        num_training_steps = num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
    elif lr_scheduler:
        lr_scheduler = lr_scheduler
    else:
        lr_scheduler = None

    print(f"Device: {device} | Learning rate scheduler: {lr_scheduler}")

    # Initialize metric history lists
    train_loss_history, train_acc_history, train_f1_history = [], [], []
    val_loss_history, val_acc_history, val_f1_history = [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0

        ##### TRAINING LOOP #####
        model.train()
        for images, truths in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            images, truths = images.to(device), truths.to(device)

            if DEFAULT_PLOT_DATASET:
                utils.helpers.plot_dataset(images, truths)

            optimizer.zero_grad()
            outputs = model(images)

            # Compute loss and metrics based on output dimensions
            if not dim_:
                outputs_ = outputs.permute(0, 2, 1)
                loss = criterion(outputs_, truths.long())
                with torch.no_grad():
                    prediction = torch.argmax(outputs, dim=-1)
                    acc_epoch, f1_epoch = accuracy_f1_transformer(prediction=prediction, truths=truths, dim_=dim_)
            else:
                loss = criterion(outputs, truths.float())
                prediction = (outputs > 0.5).float()
                acc_epoch, f1_epoch = accuracy_f1_transformer(prediction=prediction, truths=truths, dim_=dim_)

            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            train_loss += loss.item()
            train_acc += acc_epoch
            train_f1 += f1_epoch

        # Calculate average metrics for training
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_acc / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_accuracy)
        train_f1_history.append(avg_train_f1)

        print(
            f"Epoch {epoch + 1}\t | Train | Loss: {avg_train_loss:.4f} | Acc: {avg_train_accuracy:.4f} | F1: {avg_train_f1:.4f}"
        )

        # Save model weights
        if save:
            os.makedirs(DEFAULT_WEIGHTS_PATH, exist_ok=True)
            weight_path = os.path.join(DEFAULT_WEIGHTS_PATH, f"{name}.pth") if name else DEFAULT_WEIGHTS_PATH
            torch.save(model.state_dict(), weight_path)

        ##### VALIDATION LOOP #####
        model.eval()
        with torch.no_grad():
            for images, truths in val_loader:
                images, truths = images.to(device), truths.to(device)
                outputs = model(images)

                if not dim_:
                    outputs_ = outputs.permute(0, 2, 1)
                    loss = criterion(outputs_, truths.long())
                    prediction = torch.argmax(outputs, dim=-1)
                    acc_epoch_eval, f1_epoch_eval = accuracy_f1_transformer(prediction=prediction, truths=truths, dim_=dim_)
                else:
                    loss = criterion(outputs, truths.float())
                    prediction = (outputs > 0.5).float()
                    acc_epoch_eval, f1_epoch_eval = accuracy_f1_transformer(prediction=prediction, truths=truths, dim_=dim_)

                val_loss += loss.item()
                val_acc += acc_epoch_eval
                val_f1 += f1_epoch_eval

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_acc / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(avg_val_accuracy)
        val_f1_history.append(avg_val_f1)

        print(
            f"Epoch {epoch + 1}\t | Val  | Loss: {avg_val_loss:.4f} | Acc: {avg_val_accuracy:.4f} | F1: {avg_val_f1:.4f}"
        )

        # Intermediate plotting
        if interplot and (epoch + 1) % LOG_SAVE_INTERVAL == 0:
            transfer.transfer_learning.plot_loss_training(
                train_loss_history, val_loss_history, title=f"Epoch {epoch + 1} Metrics", label_l1="Train Loss", label_l2="Val Loss"
            )

        # Save logs
        if save:
            logs = {
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
                "train_acc": train_acc_history,
                "val_acc": val_acc_history,
                "train_f1": train_f1_history,
                "val_f1": val_f1_history,
            }
            os.makedirs(DEFAULT_LOGS_PATH, exist_ok=True)
            log_path = os.path.join(DEFAULT_LOGS_PATH, f"{name}_epoch_{epoch}.json")
            with open(log_path, "w") as log_file:
                json.dump(logs, log_file)

    # Final plotting
    if plot:
        transfer.transfer_learning.plot_loss_training(
            train_loss_history, val_loss_history, title="Final Metrics", label_l1="Train Loss", label_l2="Val Loss"
        )

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_history, val_f1_history