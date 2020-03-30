# Towards taming oral cancer speech in the wild

### Description
This git repository contains the training and analysis code used in our paper. 
The training script for the neural network is not included.


### Installation
The extracted features and the raw dataset can be downloaded from:
https://zenodo.org/record/3732322#.XoHWYfGxVFM


After succesful feature extraction, you need to create a virtual environment with the packages required. This is contained in the requirements file. Using pip

`python3 -m venv venv_name
source/venv/bin/activate
pip install -r requirements.txt`

This should ensure that every package is installed. 

### Demo

Finally, in order to reproduce a tab separated file containing the results in the paper you need to:

`./run.sh`

If you want to also reproduce the figures, then you need to call

`python3 analysis.py`


