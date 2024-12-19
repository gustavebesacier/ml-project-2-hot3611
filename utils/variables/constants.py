# CONSTANTS FOR LR_SCHED AND METRICS
from utils.variables.env_variables import set_device
from utils.variables.path_variables import PARAMETERS_UNET
from utils.variables.transformers_variables import PARAMETERS_TRANSFORMERS

LOSS_COLOR_1 = "b"  # Color for training loss plot
LOSS_COLOR_2 = "r"  # Color for validation loss plot
ACC_COLOR = "r"     # Color for accuracy plot
F1_COLOR = "k"      # Color for F1 score plot
DEFAULT_T_MULT = 1  # Default T_mult for cosine annealing scheduler
DEFAULT_WARMUP_RATIO = 0.1  # Warmup ratio of total training steps

# Constants for MixUp
DEFAULT_PROPORTION = 0.5  # Default proportion of images to mix
DEFAULT_ALPHA = 0.8      # Default alpha for the Beta distribution
MIXUP_START_INDEX = 101  # Starting index for mixed images
IMAGE_EXTENSION = "png"  # File extension for saved mixed images

# Constants for submission
ANGLES = [0, 90, 180, 270]  # Rotation angles for data augmentation
THRESHOLD = 0.5  # Threshold for binary segmentation
TEST_IMAGE_COUNT = 50  # Number of test images

# Constants for Vision Transformer
DEFAULT_IMAGE_SIZE = 400  # Size of the input image
DEFAULT_PATCH_SIZE = PARAMETERS_TRANSFORMERS['patch_size']
DEFAULT_NUM_CLASSES = PARAMETERS_TRANSFORMERS['num_classes']
DEFAULT_DIM = PARAMETERS_TRANSFORMERS['dim']
DEFAULT_DEPTH = PARAMETERS_TRANSFORMERS['depth']
DEFAULT_HEADS = PARAMETERS_TRANSFORMERS['lin_nheads']
DEFAULT_DROPOUT = PARAMETERS_TRANSFORMERS['lin_dropout']
DEFAULT_SEQ_LEN = 14 * 14 + 1  # Sequence length for ViT
DEFAULT_LR = 1e-4  # Learning rate for optimizer
DEFAULT_NUM_EPOCHS = 100  # Number of training epochs
DEFAULT_STEP_SIZE = 1  # Step size for learning rate scheduler
DEFAULT_GAMMA = PARAMETERS_TRANSFORMERS['gamma']  # Gamma for learning rate scheduler

# Constants for Model Training
LOG_SAVE_INTERVAL = 5                  # Interval for saving intermediate metrics
DEFAULT_LR_SCHEDULER_TYPE = 'linear'   # Default learning rate scheduler type
DEFAULT_PLOT_DATASET = False           # Whether to plot dataset samples
DEFAULT_DEVICE = set_device(force_cpu=False)  # Default device (CPU/GPU)
DEFAULT_LOGS_PATH = PARAMETERS_UNET['logs_path']
DEFAULT_WEIGHTS_PATH = PARAMETERS_UNET['weights_path']


