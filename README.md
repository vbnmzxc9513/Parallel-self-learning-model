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
3. [Train model](#train-model)


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
We got dataset from Kaggle Food Images (Food-101): https://www.kaggle.com/kmader/food41
You can download images from the above url or using Kaggle api download images by following command:
```
kaggle datasets download -d kmader/food41
```

## Train model

To train models, run following commands.
```
python main.py
````

## Modification guide

### crawler.py
Use for crawler google imgs with multithread.   
To use selenium to crawler imgs , you need to assign the path of chrome driver by yourself.
```
line:56  driver_path = 'yourpath/driver/chromedriver'
```
note : driver_path must be an absolute path.   

### convert_img_hog.py
Ues for converting image to HOG in multithread   
To prepare the training data for converting image data to eigenvector, simply execute all the cells. After preparing training data, change the input path input:  
```
line:19 self.train_path = 'yourpath/train'
```
It will automatically convert the data to HOG file

### cuda_svm.py

## File Explanation
