import torch
import torch.nn as nn
from linformer import Linformer
from transformers import ViTForImageClassification
from vit_pytorch.efficient import ViT
import os
import sys
import inspect
from utils.variables.constants import *
import utils
from utils.variables.transformers_variables import PARAMETERS_TRANSFORMERS
import vision_transformer.ViT_train

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import data_loader


class LinformerViT(nn.Module):
    """
    A Vision Transformer model with Linformer for efficient attention mechanism.
    """
    def __init__(self, model):
        super(LinformerViT, self).__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass with sigmoid activation.

        @param x: Input tensor.
        @return: Transformed tensor after passing through the model and sigmoid.
        """
        return self.sigmoid(self.model(x))


class PretrainedViT(nn.Module):
    """
    Pre-trained Vision Transformer model (ViT) with a custom classifier layer.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained ViT model
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vit.classifier = nn.Linear(in_features=768, out_features=DEFAULT_NUM_CLASSES, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward pass with sigmoid activation.

        @param inputs: Input tensor.
        @return: Transformed logits after passing through the model and sigmoid.
        """
        f_pass = self.vit(inputs)
        return self.sigmoid(f_pass.logits)


def get_LinformerViT(device=utils.variables.env_variables.set_device(force_cpu=False)):
    """
    Creates and returns a Linformer-based Vision Transformer model.

    @param device: The device to use (CPU or GPU).
    @return: A LinformerViT model instance.
    """
    efficient_transformer = Linformer(
        dim=DEFAULT_DIM,
        seq_len=DEFAULT_SEQ_LEN,
        depth=DEFAULT_DEPTH,
        heads=DEFAULT_HEADS,
        dropout=DEFAULT_DROPOUT
    )

    base_model = ViT(
        dim=DEFAULT_DIM,
        image_size=DEFAULT_IMAGE_SIZE,
        patch_size=DEFAULT_PATCH_SIZE,
        num_classes=DEFAULT_NUM_CLASSES,
        transformer=efficient_transformer,
        channels=3
    ).to(device)

    model = LinformerViT(base_model)
    return model