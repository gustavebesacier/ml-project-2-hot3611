import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training")
ROOT_FOLDER = os.path.abspath(os.path.join(path, "../../..")) # /CS-433-Project2

data_train_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training")
data_train_images = os.path.join(data_train_root, "images")
data_train_truth = os.path.join(data_train_root, "groundtruth")

ROOT_TRAINING = os.path.join(ROOT_FOLDER, "dataset/training")
IMAGES_PATH = os.path.join(ROOT_TRAINING, "images") # "dataset/training/images/"
GROUNDTRUTH_PATH = os.path.join(ROOT_TRAINING, "groundtruth") # "dataset/training/groundtruth/"

ROOT_TRAINING_SHORT = os.path.join(ROOT_FOLDER, "dataset/training_short")
IMAGES_PATH_SHORT = os.path.join(ROOT_TRAINING_SHORT, "images") # "dataset/training/images/"
GROUNDTRUTH_PATH_SHORT = os.path.join(ROOT_TRAINING_SHORT, "groundtruth") # "dataset/training/groundtruth/"

ROOT_AUGMENTED = os.path.join(ROOT_FOLDER, "dataset/augmented")
AUGMENTED_IMAGES_PATH = os.path.join(ROOT_AUGMENTED, "images") # "dataset/augmented/images/"
AUGMENTED_GROUNDTRUTH_PATH = os.path.join(ROOT_AUGMENTED, "groundtruth") # "dataset/augmented/groundtruth/"

ROOT_TEST = os.path.join(ROOT_FOLDER, "dataset/test_set_images")
TEST_IMAGES_DIR = ROOT_TEST # os.path.join(ROOT_TEST, "images") # "dataset/test_set_images/"
SUBMISSION_FILENAME = "submission.csv"
SUBMISSION_DIR = os.path.join(ROOT_FOLDER, "submissions")

PREDICTIONS_DIR = os.path.join(ROOT_FOLDER, "predictions")

ROOT_TEST_SHORT = os.path.join(ROOT_FOLDER, "dataset/test_set_images_short")
TEST_IMAGES_DIR_SHORT = ROOT_TEST_SHORT # os.path.join(ROOT_TEST, "images") # "dataset/test_set_images/"

TEST_PATH = ROOT_TEST # "dataset/test_set_images"

PATH_LOG = os.path.join(ROOT_FOLDER, "CurbCatcher")
PATH_LOG_FILE = os.path.join(PATH_LOG, "log.txt")
PATH_LOG_METRICS = os.path.join(PATH_LOG, "metrics.txt")


PARAMETERS_UNET = {
    'weights_path': os.path.join(ROOT_FOLDER, "model_weights", "Unet"),
    'weight_emplacement': os.path.join(ROOT_FOLDER, "model_weights", "Unet", "Unet.pth"),
    'logs_path': os.path.join(ROOT_FOLDER, "logs", "Unet"),
    'graph_path': os.path.join(ROOT_FOLDER, "graphs", "Unet"),
    "lr_sched_restart": 0.15,
    "lr_sched_increase": 2,
    "lr_sched_warmup": 0.1,
    'lr_sched_min_eta':5e-10
}