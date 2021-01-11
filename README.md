# Parallel-self-learning-model

This is a project to implement a machine learning system that can self-learning by collecting data from google using crawler. This system can be split into three major stages, the crawler stage, feature extraction stage, and SVM modeling stage. Each stage takes a long time, so we using lots of parallel methods to speed up them. Finally, we get approximately 85 times speedup over the serial program.


## Hardware
* CPU: Intel(R) Xeon(R) Gold 6136 CPU @ 3.00GHz 
* Core: 2 * (12 cores 24 threads)
* GPU: RTX 2080 Ti 12GB

OS: 
* CentOS 8

## Reproducing implementation
To reproduct our implementation without do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Train models](#train-models)
4. [Make Submission](#make-submission)


## Installation
Using Anaconda is strongly recommended.

### Build environment
```
conda create -n pp_smls python=3.6
conda activate pp_smls
git clone https://github.com/vbnmzxc9513/Parallel-self-learning-model.git
pip install -r requirement.txt
```

## Dataset Preparation  
https://www.kaggle.com/kmader/food41

## Train model

## Modification guide

### Crawler.py

### Feature Extraction

### SVM train
