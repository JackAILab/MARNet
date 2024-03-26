# %%
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from scipy import ndimage
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datas import*  #sss
import os
import h5py

def scipy_rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
    volume[volume < 0] = 0 
    volume[volume > 1] = 1
    return volume

# =============================== train_data_h5f_path =============================== #
train_save_data_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_Data.he5' # h5f data path
train_save_label_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_Data_label.he5' # h5f label path
train_save_patient_name_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_name_label.he5' # h5f name path

test_save_data_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_Data.he5'
test_save_label_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_Data_label.he5'
test_save_patient_name_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_name_label.he5' 

class BasicDataset(Dataset):
    def __init__(self, image_path, label, data_type, transform=None):
        self.image_path = image_path
        self.label = label
        self.transform = transform
        self.data_type = data_type
        if self.data_type == 'train':
            data_h5f = h5py.File(train_save_data_path, 'r')
            label_h5f = h5py.File(train_save_label_path, 'r')
        elif self.data_type == 'test':
            data_h5f = h5py.File(test_save_data_path, 'r')
            label_h5f = h5py.File(test_save_label_path, 'r')
        self.length = len(data_h5f)
        self.keys = list(data_h5f.keys())
        random.shuffle(self.keys)
        data_h5f.close()
        label_h5f.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data_type == 'train':
            data_h5f = h5py.File(train_save_data_path, 'r')
            label_h5f = h5py.File(train_save_label_path, 'r')
            name_h5f = h5py.File(train_save_patient_name_path,'r')
        elif self.data_type == 'test':
            data_h5f = h5py.File(test_save_data_path, 'r')
            label_h5f = h5py.File(test_save_label_path, 'r')
            name_h5f = h5py.File(test_save_patient_name_path,'r')
        
        key = self.keys[idx]
        
        data_arr = np.array(data_h5f[key]) 
        label_arr = np.array(label_h5f[key])
        name_arr = np.array(name_h5f[key])
        
        data_h5f.close()
        label_h5f.close()     
        name_h5f.close()
        
        data_arr = np.resize(data_arr, (1, 20, 256, 256))
        
        return data_arr, label_arr ,name_arr
