# Towards taming oral cancer speech in the wild

This git repository contains the analysis code used in our paper, except the neural network model.

The feature extraction can be done by running the Kaldi feature extraction script on the data that you have to download from: (TODO: include)
`./run_feature.sh`

After succesful feature extraction, you need to create a virtual environment with the packages required. This is contained in the requirements file. Using pip

`python3 -m venv venv_name
source/venv/bin/activate
pip install -r requirements.txt`

This should ensure that every package is installed. Finally, in order to reproduce a tab separated file containing the results in the paper you need to:

`./run.sh`

If you want to also reproduce the figures, then you need to call

`python3 analysis.py`


