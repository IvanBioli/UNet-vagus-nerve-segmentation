# nerve-segmentation




## Setup

All instructions are relative to the root directory of the repository.

### Install environment

1. We use conda for managing our environment, install the environment with the provided `.yml` file by running

``` 
conda env create -f vagus.yml
```

Make sure conda is installed on your computer otherwise visit https://docs.anaconda.com/anaconda/install/index.html

Alternatively, there is also a requirements.txt file provided. 

### Get Data

Steps are only required if training the model

1. In the `initialise_run()` in `src/config.py`, change directory to the root directory of the repository.
2. Download data from drive and put it in: `data/original_dataset`.
3. Run `src/data_convert.py` to convert the dataset to model ready format (only run this once).

## Training

1. Change the `model_id` parameter in `src/config.py` to a suitable name
2. Run `src/run.py`
3. To investigate different loss functions change the `_loss` in `src/train.py`. Refer to keras documentation for appropriate loss functions. Examples are provided in `src/loss.py`.

## Evaluation

1. To get some visualisations and the evaluation scores run `src/visualisation.py`

## Repository Description

* `data` - contains datasets
* `model_checkpoints` - contains pretrained models
* `model_losses` - contains metrics and losses of model training runs
* `notebooks` - research notebooks and code
* `papers`  - research papers
* `report` - report figures
* `results` - more pictures of various results
* `src` - code folder
  * `augmentation.py` - augmentation utility functions
  * `config.py` - constant and configuration parameters
  * `data_convert.py` - converts original dataset to model ready format
  * `data_utils.py` - various data utility functions
  * `eval.py` - evaluation utility functions
  * `fine_tune.py` - fine tuning function
  * `loss.py` - loss functions and metrics that are passed into model during training
  * `model.py` - function to build model
  * `post_processing.py` - utility functions for after training
  * `run.py` - main run script for model training / fine tuning
  * `stats.py` - utility functions for prediction statistics
  * `train.py` - main script for training
  * `visualisation.py` - main script for visualising model results
