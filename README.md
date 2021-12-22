# U-Net for segmenting fascicles in vagus nerve histologies
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

1. In the `initialise_run()` in `src/config.py`, change directory to the root directory of the repository.
2. Download data from Google Drive (link [here](https://drive.google.com/file/d/1n-2QMZYHBPzEn63J9pFw_tijWtcLZu2X/view?usp=sharing)), unzip the folder and put it in: `data/original_dataset`.
3. Run `src/data_convert.py` to convert the dataset to model ready format (only run this once).

## Evaluation and results visualization
1. Please, be sure that the `cur_model_id` parameter in `src/config.py` is set to the model for which you want to visualize
the results. To use our best pre-trained model, use `cur_model_id = 'FL_and_BCE_Adam_default'`
2. To visualize the model predictions and the evaluation scores run `src/visualisation.py`

## Training

Steps required only if training the model again (it should take around 4 hours):
1. Change the `cur_model_id` parameter in `src/config.py` to a suitable name in order to avoid overriding our pre-trained models. 
2. Run `src/run.py`
3. To investigate different loss functions change the `_loss` in `src/train.py`. Refer to keras documentation for appropriate loss functions. Examples are provided in `src/loss.py`.
Beware that our predictions are integers and not one-hot labels.

## Repository Description

* `data` - contains datasets
* `model_checkpoints` - contains pretrained models
* `model_losses` - contains metrics and losses of model training runs
* `report` - report figures and latex
* `results` - more pictures of various results
* `src` - code folder
  * `augmentation.py` - augmentation utility functions
  * `config.py` - constant and configuration parameters
  * `data_convert.py` - converts original dataset to model ready format
  * `data_utils.py` - various data utility functions
  * `eval.py` - evaluation utility functions
  * `fine_tune.py` - fine tuning function (to be used for future extensions of this work)
  * `loss.py` - loss functions and metrics that are passed into model during training
  * `model.py` - function to build model
  * `post_processing.py` - utility functions for post-processing
  * `run.py` - main run script for model training
  * `stats.py` - utility functions for prediction statistics
  * `train.py` - main script for training
  * `visualisation.py` - main script for visualising model results
