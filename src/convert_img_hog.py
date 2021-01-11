import random
import multiprocessing
import threading
import os
from multiprocessing import Pool
from multiprocessing import Process, Queue
from queue import Queue
from os.path import isfile, isdir, join
from skimage import feature as ft
import cv2
import numpy as np
import time
from skimage import data, exposure
import pandas as pd


mutex = threading.Lock()

class Config():
    def __init__(self):
        self.train_path = join(os.getcwd(), 'train')
        self.test_path = join(os.getcwd(), 'test')
        self.save_name = 'Train_HOG'
        
        self.class_dict = {}
        self.class_len=0

        for class_name in os.listdir(self.train_path):
            self.class_dict[str(self.class_len)]=class_name
            self.class_len+=1

def convert_hog(img):
    features = ft.hog(img,  # input image
                      orientations=9,  # number of bins
                      pixels_per_cell=(20, 20), # pixel per cell
                      cells_per_block=(2,2), # cells per blcok
                      block_norm = 'L1',
                      transform_sqrt = True, # power law compression (also known as gamma correction)
                      feature_vector=True) # return HOG map
    hog_image_rescaled = exposure.rescale_intensity(features, in_range=(-1, 1))

    return features

def rescale_linear(array):
    """Rescale an arrary linearly."""
    
    new_max = 1
    new_min = -1
    
    minimum = numpy.min(array)
    maximum = numpy.max(array)
    
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b



def load_balance(worker, class_dict, path, save_name, config):
   
    temp_list = []
    vect = 0
    print(class_dict)
    for class_name in (class_dict):
        for img in os.listdir(join(path, class_name)):
            
            img_path = join(path, class_name, img)
                
            input_img = cv2.imread(img_path)
                
            input_img = cv2.resize(input_img, (512, 512))
                
            feature = convert_hog(input_img)

            if vect == 0:
                vect = len(feature)
            feature_len = len(feature)

            class_num = list(config.class_dict.keys())[list(config.class_dict.values()).index(class_name)]
            
            list(feature).append(class_num)
            str_to_write = str(class_num)
            for i in range(len(feature)):
                str_to_write = str_to_write +" "+ str(i+1)+":"+str(feature[i])
                

            if feature_len == vect:
                temp_list.append(str_to_write)
            

    temp_list[len(temp_list)-1] = temp_list[len(temp_list)-1]+'\n'
     
    mutex.acquire()
    with open(save_name, "a") as outfile:
        outfile.write("\n".join(temp_list))
    
    mutex.release()
    
    print("worker : "+str(worker)+" finished!" + " data length = " + str(len(temp_list)))




temp_list = []

def main():
    np = multiprocessing.cpu_count()
    np = 7
    print('You have '+str(np)+' CPUs')
    config = Config()
    print(config.class_dict)
    class_num = len(config.class_dict)
    print("Total class:" + str(class_num))
    each_class_proc = int(class_num/np)
    print('Each processor avg handle : ' + str(each_class_proc) + ' class')
    
    
    ##########You should config below########
    path = config.train_path
    print(path)
   
    config.save_name = 'Train_food7'
   

    try:
        os.remove(config.save_name)
    except OSError as e:
        print(e)
    else:
        print("File is deleted successfully")
    
    start = time.time()
    
    ##########You should config obove########
    
    p_list = []
    for i in range(np):
        if i == np-1 :
            each_class_proc+=(class_num%np) 
        p_list.append(Process(target=load_balance, args=(i, os.listdir(path)[i:i+each_class_proc], path, config.save_name,config)))

    for xx in p_list:
        xx.start()

    for xx in p_list:
        xx.join()
    end = time.time()
    print("Time : " + str(end - start))


    


