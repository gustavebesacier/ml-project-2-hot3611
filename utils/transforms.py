import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import numpy as np

from utils.variables.variables import VERTICAL_FLIP_PROBABILITY, HORIZONTAL_FLIP_PROBABILITY, BLUR_PROBABILITY, \
    CHANNEL_PERM_PROBABILITY, CONTRAST, BRIGHTNESS, SATURATION, HUE


def transform_images(image, truth, hFlip, vFlip, rot, color, hFlip_prob=HORIZONTAL_FLIP_PROBABILITY, vFlip_prob=VERTICAL_FLIP_PROBABILITY, blur=True, channel_permutation=True):
    """
    Applies a series of random transformations to an image and its corresponding ground-truth label for data augmentation.

    @param image: The input image to be transformed.
    @param truth: The corresponding ground-truth label to be transformed.
    @param hFlip: Whether to apply horizontal flipping.
    @param vFlip:  Whether to apply vertical flipping.
    @param rot: Whether to apply random rotations (0°, 90°, 180°, 270°).
    @param color: Whether to apply random color adjustments (brightness, contrast, saturation, and hue).
    @param hFlip_prob: The probability of applying horizontal flipping. Default is 0.5.
    @param vFlip_prob: The probability of applying vertical flipping. Default is 0.5.
    @param blur: Whether to apply Gaussian blur to the image. Default is True.
    @param channel_permutation: Whether to randomly permute the color channels of the image. Default is True.

    @return: A tuple containing the transformed image and truth as:
        - image: The augmented image.
        - truth: The augmented ground-truth label.

    Transformations Applied:
        - Horizontal Flip: Randomly flips the image and truth horizontally with a given probability.
        - Vertical Flip: Randomly flips the image and truth vertically with a given probability.
        - Rotation: Randomly rotates the image and truth by one of the angles {0°, 90°, 180°, 270°}.
        - Color Jitter: Adjusts brightness, contrast, saturation, and hue of the image.
        - Gaussian Blur: Applies a Gaussian blur with a random kernel size.
        - Channel Permutation: Randomly rearranges the color channels of the image.
    """

    if hFlip and torch.rand(1) < hFlip_prob:
        image = TF.hflip(image)
        truth = TF.hflip(truth)
    if vFlip and torch.rand(1) < vFlip_prob:
        image = TF.vflip(image)
        truth = TF.vflip(truth)
    if rot:
        angle = torch.randint(0, 4, (1,)).item() * 90  # Randomly pick 0, 90, 180, or 270
        image = TF.rotate(image, angle)
        truth = TF.rotate(truth, angle)
    if color:
        color_jitter = T.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE)
        image = color_jitter(image)
    if blur and torch.rand(1) < BLUR_PROBABILITY:
        kernel = np.random.randint(5)*2 + 1 # ensure odd number
        blur_ = T.GaussianBlur(kernel)
        image = blur_(image)

    if channel_permutation and torch.rand(1) < CHANNEL_PERM_PROBABILITY:
        permut = T.v2.RandomChannelPermutation()
        image = permut(image)

    return image, truth

MODEL_PARAMETERS = {
    'unet': {
        'target_size': 416
    },
    'vit': {
        'target_size': 224
    }
}