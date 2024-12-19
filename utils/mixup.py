from PIL import Image
import numpy as np
import utils.transforms
from utils.variables.path_variables import *
from utils.variables.constants import *
import os

def mix_it_up(proportion=DEFAULT_PROPORTION, alpha=DEFAULT_ALPHA, plot=False, short=False):
    """
    Applies the MixUp data augmentation technique to a subset of images and their labels.

    @param proportion: Proportion of the dataset to use for MixUp. Default is DEFAULT_PROPORTION.
    @param alpha: Alpha parameter for the Beta distribution used in MixUp. Default is DEFAULT_ALPHA.
    @param plot: Whether to plot the original and mixed images. Default is False.
    @param short: Whether to use the short version of the dataset. Default is False.

    @return: None. Mixed images and labels are saved to disk.
    """
    # Set directories based on whether the short dataset is used
    image_dir = IMAGES_PATH_SHORT if short else IMAGES_PATH
    mask_dir = GROUNDTRUTH_PATH_SHORT if short else GROUNDTRUTH_PATH

    # Get image and ground-truth filenames, excluding unwanted files
    image_filenames = [f for f in os.listdir(image_dir) if f != '.DS_Store' and not f.startswith('mixup')]
    truth_filenames = [f for f in os.listdir(mask_dir) if f != '.DS_Store' and not f.startswith('mixup')]

    # Randomly select N pairs of images and ground truths
    N = int(np.floor(proportion * len(image_filenames)))
    idx_a = np.random.randint(len(image_filenames), size=N)
    idx_b = np.random.randint(len(image_filenames), size=N)

    # Perform MixUp for each selected pair
    for i, (a, b) in enumerate(zip(idx_a, idx_b)):
        # Load images and labels
        image_a_path = os.path.join(image_dir, image_filenames[a])
        image_b_path = os.path.join(image_dir, image_filenames[b])
        truth_a_path = os.path.join(mask_dir, truth_filenames[a])
        truth_b_path = os.path.join(mask_dir, truth_filenames[b])

        image_a, image_b = Image.open(image_a_path).convert("RGB"), Image.open(image_b_path).convert("RGB")
        truth_a, truth_b = Image.open(truth_a_path).convert("L"), Image.open(truth_b_path).convert("L")

        np_image_a, np_image_b = np.array(image_a).astype(np.float32), np.array(image_b).astype(np.float32)
        np_truth_a, np_truth_b = np.array(truth_a).astype(np.float32), np.array(truth_b).astype(np.float32)

        # MixUp the images and labels using a Beta distribution
        l = np.random.beta(alpha, alpha)
        np_mixed_image = l * np_image_a + (1 - l) * np_image_b
        np_truth_image = l * np_truth_a + (1 - l) * np_truth_b

        mixed_image = Image.fromarray(np.uint8(np_mixed_image))
        mixed_truth = Image.fromarray(np.uint8(np_truth_image))

        # Plot the images if required
        if plot:
            utils.helpers.plot_3_images(image_a, image_b, mixed_image)

        # Save the mixed images and labels
        mixed_img_path = os.path.join(image_dir, f"mixup_satImage_{MIXUP_START_INDEX + i}.{IMAGE_EXTENSION}")
        mixed_truth_path = os.path.join(mask_dir, f"mixup_satImage_{MIXUP_START_INDEX + i}.{IMAGE_EXTENSION}")
        mixed_image.save(mixed_img_path)
        mixed_truth.save(mixed_truth_path)

    # Print confirmation
    print(f"Data has been MixedUp (image indexes from {MIXUP_START_INDEX} to {MIXUP_START_INDEX + N - 1}).")
