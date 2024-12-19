import json
from tqdm import tqdm
import utils.helpers
from transfer.transfer_learning import *
from utils.variables.env_variables import set_device
from utils.variables.variables import *
import utils.variables.path_variables


def train_model(train_loader, val_loader, model, num_epochs=NUM_EPOCHS, save=True, interplot=False, plot=True,
                name: str = False, lr_scheduler=False, gamma=DEFAULT_GAMMA, step=DEFAULT_STEP):
    """
    Trains a model using the specified training and validation data loaders.

    @param train_loader: Data loader for the training dataset.
    @param val_loader: Data loader for the validation dataset.
    @param model: The model to be trained.
    @param num_epochs: Number of training epochs. Default is NUM_EPOCHS.
    @param save: Whether to save the model weights and logs. Default is True.
    @param interplot: Whether to plot intermediate metrics during training. Default is False.
    @param plot: Whether to plot the final metrics after training. Default is True.
    @param name: Name to use when saving model weights and logs. Default is False.
    @param lr_scheduler: Specifies the learning rate scheduler to use. Default is False (no scheduler).
    @param gamma: Multiplicative factor for the learning rate decay. Default is DEFAULT_GAMMA.
    @param step: Step size for the learning rate scheduler. Default is DEFAULT_STEP.

    @return: None
    """
    device = set_device(force_cpu=False)

    model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},  # Lower LR for pre-trained encoder
        {'params': model.decoder.parameters(), 'lr': 1e-3},
        {'params': model.segmentation_head.parameters(), 'lr': 1e-3}
    ])

    if lr_scheduler == 'default':
        lr_scheduler = get_lr_scheduler(len_dataset=len(train_loader), optimizer=optimizer, n_epochs=num_epochs)
    elif lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    else:
        lr_scheduler = None

    print(f"Device: {device} | Learning rate scheduler: {lr_scheduler}")

    train_loss_history = []
    train_acc_history = []
    train_f1_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []

    for epoch in range(num_epochs):

        train_loss, val_loss = 0., 0.
        train_acc, val_acc = 0., 0.
        train_f1, val_f1 = 0., 0.

        model.train()

        metrics_epoch_train = dict(zip(METRICS.keys(), torch.zeros(len(METRICS))))

        for images, truths in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}"):

            images, truths = images.to(device), truths.to(device)

            if PLOT_DATASET:
                utils.helpers.plot_dataset(images, truths)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, truths)
            loss.backward()
            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            with torch.no_grad():
                outputs_ = torch.sigmoid(outputs) > 0.5

            train_loss += loss.item()
            metrics_epoch_train['ACC'] += accuracy(outputs, truths)  # .cpu()
            metrics_epoch_train['F1'] += f1_metric(outputs_, truths).cpu()
            train_acc += accuracy(outputs, truths)
            train_f1 += f1_metric(outputs_, truths)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_acc / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)

        metrics_epoch_train['ACC'] /= len(train_loader)
        metrics_epoch_train['F1'] /= len(train_loader)

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_accuracy)
        train_f1_history.append(avg_train_f1)

        tqdm.write(
            f"Epoch {epoch + 1}\t | Train | Loss: {avg_train_loss:.4f} | Acc: {avg_train_accuracy:.4f} | F1: {avg_train_f1:.4f}")

        if save:
            if not os.path.exists(utils.variables.path_variables.PARAMETERS_UNET['weights_path']):
                os.makedirs(utils.variables.path_variables.PARAMETERS_UNET['weights_path'])
            if name:
                emplacement = os.path.join(utils.variables.path_variables.PARAMETERS_UNET['weights_path'], f"{name}.pth")
            else:
                emplacement = utils.variables.path_variables.PARAMETERS_UNET['weight_emplacement']
            torch.save(model.state_dict(), emplacement)

        model.eval()

        with torch.no_grad():

            for images, truths in val_loader:
                images, truths = images.to(device), truths.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, truths)
                val_loss += loss.item()
                val_acc += accuracy(outputs, truths)

                outputs = torch.sigmoid(outputs) > 0.5
                val_f1 += f1_metric(outputs, truths)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_acc / len(val_loader)
        avg_f1 = val_f1 / len(val_loader)

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(avg_val_accuracy)
        val_f1_history.append(avg_f1)

        tqdm.write(
            f"Epoch {epoch + 1}\t | Test  | Loss: {avg_val_loss:.4f} | Acc: {avg_val_accuracy:.4f} | F1: {avg_f1:.4f}")

        if interplot:
            if (epoch + 1) % 5 == 0:
                titre = f"Training loss and accuracy (epoch {epoch})"
                plot_loss_training(losses=train_loss_history, losses2=train_acc_history, title=titre,
                                   label_l1="Train loss", label_l2="Train accuracy")
                titre2: str = f"Training and validation losses (epoch {epoch})"
                plot_loss_training(losses=train_loss_history, losses2=val_loss_history, title=titre2,
                                   label_l1="Train loss", label_l2="Test loss")
                plot_all_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history,
                                 train_f1_history, val_f1_history, epoch=str(epoch))

        if save:  # Save the logs for each epoch
            out = {
                'train_loss': train_loss_history,
                'test_loss': val_loss_history,
                'train_f1': train_f1_history,
                'test_f1': val_f1_history,
                'train_acc': train_acc_history,
                'test_acc': val_acc_history
            }

            for key, value in out.items():
                out[key] = [elem.item() if isinstance(elem, torch.Tensor) else elem for elem in value]

            l_path = utils.variables.path_variables.PARAMETERS_UNET['logs_path']
            if not os.path.exists(l_path):
                os.makedirs(l_path)
            file_path = os.path.join(l_path, f'log_epoch_{epoch}.json') if not name else os.path.join(l_path,
                                                                                                      f'{name}_log_epoch_{epoch}.json')
            with open(file_path, 'w') as file:
                json.dump(out, file)

    if plot:  # Save final metrics
        titre: str = f"Training loss and accuracy ({num_epochs} epochs)"
        plot_loss_training(losses=train_loss_history, losses2=train_acc_history, title=titre, label_l1="Train loss",
                           label_l2="Train accuracy")
        titre2: str = f"Training and validation losses ({num_epochs} epochs)"
        plot_loss_training(losses=train_loss_history, losses2=val_loss_history, title=titre2, label_l1="Train loss",
                           label_l2="Test loss")
        plot_all_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_history,
                         val_f1_history)