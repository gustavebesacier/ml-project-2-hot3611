from utils.variables.env_variables import *
from utils.variables.variables import *
from utils.variables.path_variables import *
import os
# Transformers settings
PARAMETERS_TRANSFORMERS = {
    "batch_size": BATCH_SIZE,
    "train_size": 0.8, # train/test split
    "patch_size": PATCH_SIZE,
    "number_heads": 12,
    "pos_encoding_n": 1000,
    "in_features": 768,
    # "out_features": 16, # 256,
    "number_transformers_blocks": 5,
    "dropout_prob": 0.1,
    "lr_sched": True,
    "lr_sched_restart": 0.15,
    "lr_sched_increase": 2,
    "lr_sched_warmup": 0.1,
    "lr_sched_min_eta": 5e-10,
    "bias_attention": False,
    "activationMLP": "leaky",
    "device": set_device(force_cpu=False),
    "weights_path": os.path.join(ROOT_FOLDER, "model_weights", "CurbCatcher"),
    "weight_emplacement": os.path.join(ROOT_FOLDER, "model_weights", "CurbCatcher", "CurbCatcher.pth"),
    "weight_emplacement_ViT": os.path.join(ROOT_FOLDER, "model_weights", "CurbCatcher", "ViT.pth"),
    "graph_path": os.path.join(ROOT_FOLDER, "graphs", "CurbCatcher"),
    "logs_path": os.path.join(ROOT_FOLDER, "logs", "CurbCatcher"),
    "logs_path_ViT": os.path.join(ROOT_FOLDER, "logs", "ViT"),
    "train_loop": True,
    "inference": True,
    "bias_mlp": True,
    "ViT_shape": 224,           # ViT
    "dim": 128,                 # ViT
    "depth": 128,               # ViT
    "lin_nheads":16,            # ViT
    "lin_dropout": 0.2,         # ViT
    "num_classes": 25*25,           # ViT
    "gamma": 0.4,               # ViT

    ## Training parameters
    "do_mixup": False,           # ViT
    "augmentation": 1,          # ViT
    "learning_rate": 5e-3,      # ViT
    "epochs": 2,              # ViT
    "average_f1": 'micro',      # ViT

}