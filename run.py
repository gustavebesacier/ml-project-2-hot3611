import vision_transformer.ViT_models
import vision_transformer.ViT_train
from CNN.FullCNN import *
from data_loader import prepare_data_transfer, prepare_transformer_data, prepare_data_cnn
from utils.variables.transformers_variables import *
from utils.variables.env_variables import *
from utils.variables.path_variables import *
import torch
import torch.nn as nn
import os
from transfer.transfer_learning import MyUNet
from utils.trainer import train_model
from submission import generate_submission


# UNET_PTH = PARAMETERS_UNET["weight_emplacement"]

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = set_device(force_cpu=False)

    if TRAIN_MODELS:
        if TRANSFER:
            best_model_epoch_nb = 75
            encoder = 'resnet101'
            train_loader, val_loader = prepare_data_transfer(n_augmentation=SAMPLES_PER_IMAGE, MixUp=False, short=False)
            model = MyUNet(encoder_name=encoder)
            train_model(train_loader, val_loader, model, num_epochs=best_model_epoch_nb, name='ResNet101', lr_scheduler=None)

        if TRANSFORMERS:
            print("Training the transformers models.")

            # 0. Appropriate data
            train_loader, val_loader = prepare_transformer_data(
                n_augmentation=PARAMETERS_TRANSFORMERS['augmentation'],
                MixUp=PARAMETERS_TRANSFORMERS['do_mixup']
            )

            # 1. Linformers
            model_linformers = vision_transformer.ViT_models.get_LinformerViT()
            optimizer = torch.optim.AdamW(model_linformers.parameters(), lr=PARAMETERS_TRANSFORMERS['learning_rate'])
            scheduler = None
            criterion = nn.BCELoss()

            model_results = vision_transformer.ViT_train.train_model(
                train_loader,
                val_loader,
                model_linformers,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=PARAMETERS_TRANSFORMERS['epochs'],
                lr_scheduler=scheduler,
                dim_=True,
                name=f"linformer_lr_aug2_mixup_{PARAMETERS_TRANSFORMERS['do_mixup']}",
                device=set_device(force_cpu=False)
            )
            print(model_results)

            # 2. Pretrained ViT
            model_ViT_pretrained = vision_transformer.ViT_models.PretrainedViT()
            optimizer = torch.optim.AdamW(model_ViT_pretrained.parameters(),
                                          lr=PARAMETERS_TRANSFORMERS['learning_rate'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

            results = vision_transformer.ViT_train.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model_ViT_pretrained,
                num_epochs=PARAMETERS_TRANSFORMERS['epochs'],
                optimizer=optimizer,
                criterion=criterion,
                lr_scheduler=scheduler,
                name=f"pretrained_VIT_output_625_lr_mixup_{PARAMETERS_TRANSFORMERS['do_mixup']}",
                dim_=True,
                device=set_device(force_cpu=True)
            )

            print(results)

        if FULL_CNN:
            train_loader, val_loader = prepare_data_cnn(short=False)
            model = FullCNN().to(device)
            train_model(train_loader, val_loader, model, name="full_cnn_plot")

    if SUBMISSION:
        if TRANSFER:
            model = MyUNet("resnet101").to(device)
            model_name = os.path.join(PARAMETERS_UNET["weights_path"],
                                      'resnet101_no_lr_no_mixup.pth')  # UNET_PTH
            print("Model name Transfer")
            generate_submission(model=model, model_name=model_name, device=device,
                                submission_file_name=SUBMISSION_FILENAME)
            print(f"Submission file saved to {SUBMISSION_FILENAME}.")
        else:
            print("Need to use transfer learning for AICrowd submission")
            return

if __name__ == "__main__":
    main()