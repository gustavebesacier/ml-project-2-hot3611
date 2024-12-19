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
├── LICENSE
├── README.md
├── data_loader.py
├── mask_to_submission.py
├── submission_to_mask.py
├── test_set_images
│   ├── test_1
│   │   └── test_1.png
│   ...
├── training
│   ├── groundtruth
│   │   ├── satImage_001.png
│   │   ├── ...
│   └── images
│       ├── satImage_001.png
│       ├── ...
└── utils
    └── utils_load.py
```

## Usage
The code can be used for different tasks; and only modules [run.py](run.py) and [variables.py](utils/variables.py) need to be modified.

### Model training
To train a model:
1. Go to [variables.py](utils/variables.py), then:
   - Set ```TRAIN_MODELS = True```
   - Set the variable of the model of your choice to `True` (`FULL_CNN`, `UNET`, `TRANSFORMER` or `TRANSFER`), and the other to `False`
   - Specifiy the training parameters of your choice (for data augmentation, batch size, number of epochs)

2. Go to [run.py](run.py):
   - Set
   - CHECK APRèS LE TAF DE LUNA

Note: it is possible to use `short=True` in all `prepare_data` function, but it is required to first download the `training_short.zip`file from the [Google Drive](https://drive.google.com/drive/folders/1Ylp8epANF9HM3bYzXbJVtyEfRiI3s1zQ?usp=share_link) and add it in the [dataset](dataset) folder.
### Model submission
You can download some of the pretrained weights from the [Google Drive](https://drive.google.com/drive/folders/1Ylp8epANF9HM3bYzXbJVtyEfRiI3s1zQ?usp=share_link) and save them on your disk. Then, set `SUBMISSION=True` in [variables.py](utils/variables.py), then specify the path in [run.py](run.py), CHANGE THAT
This will use the pre-trained model to make predictions and generate a .csv file containing those predictions.

