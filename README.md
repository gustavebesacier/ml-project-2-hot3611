# CS-433-Project2
This project is about trying to extract roads from satellite images, as part of the EPFL course [CS-433](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/). We implement various Transformers and Convolutional Neural Networks.

## Setup
The code requires some external libraries, mainly for algorithmic implementations. All the detail can be found in [requirement.txt](requirements.txt) and installed using the command
```$ pip install -r requirements.txt  ```
## Data
Get data from [AICrowd page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).
Organize the files as:
```bash
.
├── CNN
│   ├── FullCNN.py 
├── LICENSE
├── README.md
├── data_loader.py
├── dataset
│   ├── test_set_images
│   │   ├── test_1
│   │   │   └── test_1.png
│   │   ├── …
│   ├── test_set_images_short
│   │   ├── test_1
│   │   │   └── test_1.png
│   │   ├── …
│   ├── training
│   │   ├── groundtruth
│   │   │   ├── satImage_001.png
│   │   │   ├── …
│   │   └── images
│   │       ├── satImage_001.png
│   │       ├── …
│   └── training_short
│       ├── groundtruth
│       │   ├── satImage_001.png
│       │   ├── …
│       └── images
│           ├── satImage_001.png
│           ├── …
├── graphs
│   ├── …
├── logs
│   ├── …
├── model_weights
│   ├── …
├── predictions
│   ├── test_1.png
│   ├── …
├── run.py
├── submission.py
├── submissions
│   ├── submission.csv
├── transfer
│   └── transfer_learning.py
├── utils
│   ├── helpers.py
│   ├── mixup.py
│   ├── trainer.py
│   ├── transforms.py
│   └── variables
│       ├── constants.py
│       ├── env_variables.py
│       ├── path_variables.py
│       ├── transformers_variables.py
│       └── variables.py
└── vision_transformer
    ├── ViT_models.py
    ├── ViT_train.py


```

## Usage
The code can be used for different tasks; and only modules [run.py](run.py) and [variables.py](utils/variables.py) need to be modified.

### Model training
To train a model:
1. Go to [`variables`](utils/variables), then:
   - Set ```TRAIN_MODELS = True``` in [`variables.py`](utils/variables/variables.py)
   - Set the variable of the model of your choice to `True` (`FULL_CNN`, `UNET`, `TRANSFORMER` or `TRANSFER`), and the other to `False`
   - Specifiy the training parameters of your choice (for data augmentation, batch size, number of epochs).

2. Go to [run.py](run.py):
   - In the script, there is one data loader call for each model. Change the desired `mixup` and `short` parameter if needed.
   - Run this module and get the weights in the [`model_weights`](model_weights) folder, and training logs in [`logs`](logs)

Note: it is possible to use `short=True` in all `prepare_data` function, but it is required to first download the `training_short.zip`file from the [Google Drive](https://drive.google.com/drive/folders/1Ylp8epANF9HM3bYzXbJVtyEfRiI3s1zQ?usp=share_link) and add it in the [dataset](dataset) folder.

### Model submission
You can download some of the pretrained weights from the [Google Drive](https://drive.google.com/drive/folders/1Ylp8epANF9HM3bYzXbJVtyEfRiI3s1zQ?usp=share_link) and save them on your disk. Then, set `SUBMISSION=True` in [variables.py](utils/variables.py), then specify the path in [run.py](run.py).

To create a submission file, as for the training, set to `True` the same variables corresponding to each model, then specify the name of the model weights (only the `name.pth`). The submission file is created as `submission.csv` in [`submissions`](submissions).

## Acknowledgements
We warmly thank all the CS-433 education team and EPFL for the infrastructure.

## Contact
Higlhy Optimized Trio (HOT), Luna, Agustina, Gustave.

![a Little meme](./deeplearning.jpg)



