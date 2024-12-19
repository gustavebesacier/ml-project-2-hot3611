from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
from sklearn.model_selection import train_test_split as train_test_split_
import numpy as np
import platform
import multiprocessing
import utils.transforms
import utils.mixup
from utils.variables.transformers_variables import *
import utils.transforms as trf
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MPC = 'spawn' if platform.system() == 'Windows' else 'fork' if platform.processor() == 'arm' else None # to enable DataLoader on Apple Silicon chips
WORKERS = multiprocessing.cpu_count() - 2


image_transforms_transformers = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.flatten(0)),
])

mask_transforms_transformers = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.flatten(0)),
])

def prepare_data_transfer(train_test_split: float=TRAIN_TEST_SPLIT_RATIO, model: str= 'unet', n_augmentation=1, batch_size = BATCH_SIZE, MixUp=True, short=False) -> (DataLoader, DataLoader):
    """ Prepare data for UNET model fine-tuning.

    @param train_test_split: relative proportion between training and validation sets.
    @param model: model name.
    @param n_augmentation: augmentation factor for dataset size (default=1 means original size)
    @param batch_size: size of the batch in the dataloader.
    @param MixUp: if True, it applies the MixUp (creates new images by combining existing images).
    @param short: if True, it uses a very short dataset. Need previous settings -> see ReadMe

    @return: (training dataloader, validation dataloader).
    """
    if short:
        image_dir = IMAGES_PATH_SHORT
        mask_dir = GROUNDTRUTH_PATH_SHORT
    else:
        image_dir = IMAGES_PATH
        mask_dir = GROUNDTRUTH_PATH

    if MixUp:
        utils.mixup.mix_it_up(short=short)

    image_filenames = [f for f in os.listdir(image_dir) if f != '.DS_Store']
    truth_filenames = [f for f in os.listdir(mask_dir) if f != '.DS_Store']

    if not MixUp:
        image_filenames = [f for f in image_filenames if not f.startswith('mixup')]
        truth_filenames = [f for f in truth_filenames if not f.startswith('mixup')]


    train_imgs, val_imgs, train_masks, val_masks = train_test_split_(
        image_filenames, truth_filenames, 
        test_size=(1 - train_test_split), 
        random_state=42,  # for reproducibility
        shuffle=True
    )

    trgt_size = trf.MODEL_PARAMETERS[model.lower()]['target_size']

    train_dataset = ImagePreprocessedDataset(
        image_dir=image_dir,
        truth_dir=mask_dir,
        image_filenames=train_imgs,
        truth_filenames=train_masks,
        n_augmentation=n_augmentation,
        target_size=(trgt_size, trgt_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=WORKERS,
        persistent_workers=True,
        multiprocessing_context=MPC
    )
    
    val_dataset = ImagePreprocessedDataset(
        image_dir=image_dir,
        truth_dir=mask_dir,
        image_filenames=val_imgs,
        truth_filenames=val_masks,
        transform=transforms,
        target_size=(trgt_size, trgt_size)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
        multiprocessing_context=MPC
    )

    return train_loader, val_loader


def prepare_transformer_data(train_test_split: float = TRAIN_TEST_SPLIT_RATIO, n_augmentation=1, batch_size=BATCH_SIZE, MixUp=False, short=False) -> (DataLoader, DataLoader):
    """ Prepare data for transformers models training.

    @param train_test_split: relative proportion between training and validation sets.
    @param n_augmentation: augmentation factor for dataset size (default=1 means original size).
    @param batch_size: size of the batch in the dataloader.
    @param MixUp: if True, it applies the MixUp (creates new images by combining existing images).
    @param short: if True, it uses a very short dataset. Need previous settings -> see ReadMe.

    @return: (training dataloader, validation dataloader).
    """
    if short:
        # Use short dataset
        image_dir = IMAGES_PATH_SHORT
        mask_dir = GROUNDTRUTH_PATH_SHORT
    else:
        # Use full dataset
        image_dir = IMAGES_PATH
        mask_dir = GROUNDTRUTH_PATH

    if MixUp:
        # Apply MixUp for data augmentation
        utils.mixup.mix_it_up(short=short)

    image_filenames = [f for f in os.listdir(image_dir) if f != '.DS_Store']
    truth_filenames = [f for f in os.listdir(mask_dir) if f != '.DS_Store']

    if not MixUp:
        # if not MixUp, avoid using previously created images with MixUp
        image_filenames = [f for f in image_filenames if not f.startswith('mixup')]
        truth_filenames = [f for f in truth_filenames if not f.startswith('mixup')]

    # Train/test data split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split_(
        image_filenames, truth_filenames,
        test_size=(1 - train_test_split),
        random_state=42,  # for reproducibility
        shuffle=True
    )

    transforms = True
    trgt_size = PARAMETERS_TRANSFORMERS["ViT_shape"]

    train_dataset = ImagePreprocessedDataset(
        image_dir=image_dir,
        truth_dir=mask_dir,
        image_filenames=train_imgs,
        truth_filenames=train_masks,
        transform=transforms,
        n_augmentation=n_augmentation,
        target_size= (trgt_size, trgt_size),
        Transformers=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=WORKERS,
        persistent_workers=True,
        multiprocessing_context=MPC
    )

    val_dataset = ImagePreprocessedDataset(
        image_dir=image_dir,
        truth_dir=mask_dir,
        image_filenames=val_imgs,
        truth_filenames=val_masks,
        transform=transforms,
        target_size=(trgt_size, trgt_size),
        Transformers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
        multiprocessing_context=MPC
    )

    return train_loader, val_loader

def prepare_data_cnn(train_test_split: float=TRAIN_TEST_SPLIT_RATIO, batch_size=BATCH_SIZE, short: bool = False, target_size=(IMAGE_WIDTH,IMAGE_HEIGHT)):
    """
    Prepare data for a CNN that classifies each pixel as road or background.

    @param train_test_split: relative proportion between training and validation sets.
    @param batch_size: size of the batch in the dataloader.
    @param short: if True, use a smaller subset of the dataset.
    @param target_size: size (H,W) to resize images and masks.

    @return: (training dataloader, validation dataloader).
    """

    # Select directories based on whether short dataset is requested
    if short:
        image_dir = IMAGES_PATH_SHORT
        mask_dir = GROUNDTRUTH_PATH_SHORT
    else:
        image_dir = IMAGES_PATH
        mask_dir = GROUNDTRUTH_PATH

    # List image and mask files
    image_filenames = [f for f in os.listdir(image_dir) if f != '.DS_Store']
    truth_filenames = [f for f in os.listdir(mask_dir) if f != '.DS_Store']

    # Train/val split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split_(
        image_filenames, truth_filenames,
        test_size=(1 - train_test_split),
        random_state=42,
        shuffle=True
    )
    train_dataset = ImagePreprocessedDataset(
        image_dir=image_dir,
        truth_dir=mask_dir,
        image_filenames=train_imgs,
        truth_filenames=train_masks,
        transform=True,  # apply data augmentation if desired
        target_size=target_size,
        Transformers=False
    )

    val_dataset = ImagePreprocessedDataset(
        image_dir=image_dir,
        truth_dir=mask_dir,
        image_filenames=val_imgs,
        truth_filenames=val_masks,
        transform=False, # usually no augmentation for validation
        n_augmentation=1,
        target_size=target_size,
        Transformers=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=WORKERS,
        persistent_workers=True,
        multiprocessing_context=MPC
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
        persistent_workers=True,
        multiprocessing_context=MPC
    )

    # Print dataset sizes
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))

    # Print the shape of one batch for sanity check
    # We only take one batch from train_loader to print shapes
    for images, masks in train_loader:
        print("Batch images shape:", images.shape)  # [B, C, H, W]
        print("Batch masks shape:", masks.shape)    # [B, 1, H, W] or [B, H, W], depending on dataset
        break

    return train_loader, val_loader


def transform_to_patches(data_images, data_truth, patch_size: int, train_size: int, only_truth=False):
    """Function obtained in 'segment_aerial_images.ipynb' file, not our work."""
    img_patches = None
    if not only_truth:
        img_patches = [img_crop(data_images[i], patch_size, patch_size) for i in range(train_size)]
        img_patches = np.asarray(
            [
                img_patches[i][j]
                for i in range(len(img_patches))
                for j in range(len(img_patches[i]))
            ]
        )

    gt_patches = [img_crop(data_truth[i], patch_size, patch_size) for i in range(train_size)]

    gt_patches = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )

    return img_patches, gt_patches

class ImagePreprocessedDataset(Dataset):
    """ Build a dataset of images, with the required transformations. """
    def __init__(self, image_dir: str, truth_dir: str, image_filenames: str, truth_filenames: str, transform=DATA_AUGMENTATION, n_augmentation = SAMPLES_PER_IMAGE, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), Transformers:bool=False):
        self.image_dir = image_dir
        self.truth_dir = truth_dir
        self.image_filenames = image_filenames
        self.truth_filenames = truth_filenames
        self.transform = transform
        self.n_augmentation = n_augmentation

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(target_size)

        self.Transformers = Transformers

    def __len__(self):
        return len(self.image_filenames) * self.n_augmentation

    def __getitem__(self, idx):

        original_idx = idx // self.n_augmentation

        # Load image and ground truth
        img_path = os.path.join(self.image_dir, self.image_filenames[original_idx])
        gt_path = os.path.join(self.truth_dir, self.truth_filenames[original_idx])
        image = Image.open(img_path).convert("RGB")
        truth = Image.open(gt_path).convert("L")

        image = self.resize(image)

        if not self.Transformers:
            truth = self.resize(truth) # If transformers, do not resize the image so possible to have 16*16 patches

        image = self.to_tensor(image)
        truth = self.to_tensor(truth)

        truth = (truth > 0).float()

        # Apply transforms only to n_augmentations - 1 images
        if self.transform:
            if (VAR_AUGMENTATION and idx % self.n_augmentation != 0) or not VAR_AUGMENTATION:
                image, truth = utils.transforms.transform_images(image, truth, hFlip=True, vFlip=True, rot=True, color=True, channel_permutation=False)



        if self.Transformers:
            _, truth = transform_to_patches(data_images=None, data_truth=truth, train_size=len(truth), patch_size=16,
                                         only_truth=True)

            # Reduce the size from (625, 16, 16) by taking assigning 0 or 1 to each square, based on treshold
            proportion_ones = np.mean(truth, axis=(1, 2))
            truth = (proportion_ones > 0.2).astype(int)


        return image, truth


def img_crop(im, w, h):
    """Function obtained in 'segment_aerial_images.ipynb' file, not our work."""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


if __name__ == "__main__":
    print("DataLoader main")