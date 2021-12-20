# nerve-segmentation




## Setup

All instructions are relative to the root directory of the repository.

### Install environment

1. We use conda for managing our environment, install the environment with the provided `.yml` file by running

``` 
conda env create -f vagus.yml
```

Make sure conda is installed on your computer otherwise visit https://docs.anaconda.com/anaconda/install/index.html

### Get Data

Steps are only required if training the model

1. In the `initialise_run()` in `src/config.py`, change directory to the root directory of the repository.
2. Download data from drive and put it in: `data/original_dataset`.
3. Run `src/data_convert.py` to convert the dataset to model ready format (only run this once).

## Training

1. Run `src/run.py`

## Evaluation