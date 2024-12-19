import numpy as np
import matplotlib.image as mpimg
import re
import torch
import tqdm
from PIL import Image as PILImage
import torchvision.transforms.functional as F
from utils.variables.path_variables import *

foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


# assign a label to a patch

def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"test_(\d+)\.png", image_filename).group(1))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def generate_submission(model, model_name: str, device, submission_file_name="submission.csv"):
    """Generates the full submission"""

    data_path = TEST_IMAGES_DIR
    submission_name = SUBMISSION_FILENAME
    predictions_path = PREDICTIONS_DIR

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    submission_path = SUBMISSION_DIR

    if not os.path.exists(submission_path):
        os.makedirs(submission_path)
    submission_pathname = os.path.join(submission_path, submission_file_name)  # path to submission

    angles = [0, 90, 180, 270]

    model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))
    model.eval()

    for idx, image_name in enumerate(tqdm.tqdm(os.listdir(TEST_IMAGES_DIR))):
        if image_name.startswith('.') or not os.path.isdir(os.path.join(TEST_IMAGES_DIR, image_name)):
            continue

        image_path = os.path.join(TEST_IMAGES_DIR, image_name, image_name + ".png")
        im = np.asarray(PILImage.open(image_path))

        predictions = []
        augmented = []

        pilimg_temp = PILImage.fromarray((im).astype(np.uint8))  # make sure image is np array otherwise breaks

        for angle in angles:
            im_rot = F.rotate(pilimg_temp, angle)
            augmented.append(np.array(im_rot) / 255.0)  # normalize: we map to [0,1]

        images = torch.tensor(np.array(augmented)).permute(0, 3, 1, 2).float()

        images = images.to(device)

        # forward pass
        with torch.no_grad():
            masks = model(images)
            prediction = torch.sigmoid(masks)
        # prediction is torch.Size([4, 1, 608, 608]): 4 images, 1 L channel, 608x608

        pred_tsr = prediction.cpu()  # torch.from_numpy(prediction.cpu().numpy())
        tsr = torch.zeros(
            pred_tsr[0].shape,
            dtype=pred_tsr[0].dtype,
            device=pred_tsr[0].device
        )

        for i in range(len(pred_tsr)):
            temp_unrot = F.rotate(pred_tsr[i], -angles[i])
            tsr = tsr + (temp_unrot / float(len(angles)))

        predictions.append(tsr.numpy())

        pred_ = predictions[0][0]

        pred_ = np.where(pred_ >= 0.5, 1, 0).astype(np.uint8)
        predict = pred_ * 255

        PILImage.fromarray(predict).convert("L").save(
            os.path.join(predictions_path, image_name + ".png")
        )

    image_filenames = []
    for i in range(1, 51):
        image_filename = os.path.join(predictions_path, f"test_{i}.png")
        image_filenames.append(image_filename)

    masks_to_submission(submission_pathname, *image_filenames)