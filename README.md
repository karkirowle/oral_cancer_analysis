# Detecting and analysing spontaneous oral cancer speech in the wild

[![DOI](https://zenodo.org/badge/243517361.svg)](https://zenodo.org/badge/latestdoi/243517361)

### Description
This git repository contains the training and analysis code used in our paper. 
The training script for the neural network is not included.

### Paper 
The paper is accepted to Interspeech 2020. The arXiV preprint can be found [here](https://arxiv.org/abs/2007.14205)

### Installation
The extracted features and the raw dataset can be downloaded from [this](
https://zenodo.org/record/3732322#.XoHWYfGxVFM) website.


After successful feature extraction, you need to create a virtual environment with the packages required. 
This is contained in the requirements file. Using pip

`python3 -m venv venv_name
source/venv/bin/activate
pip install -r requirements.txt`

This should ensure that every package is installed. 

### Demo

Finally, in order to reproduce a tab separated file containing the results in the paper you need to:

`./run.sh`

If you want to also reproduce the figures, then you need to call

`python3 analysis.py`

### References

- EER evaluation code is from the [ASVspoof 2019 baseline](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip)
- American English phonetic posterior processing is from [this paper](https://github.com/guanlongzhao/fac-via-ppg) 
