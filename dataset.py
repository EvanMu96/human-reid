import torch
import torch.nn as nn
import torchvision
import random
import math
import datetime
#import adabound
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim import lr_scheduler
from torch.nn import functional as F
import numpy as np

class JID_train(Dataset):
    """
    Jockey Recogition Training set.
    """
    def __init__(self, filelist, dataset_location='.', transform=None):
        self.filelist = open(filelist).readlines()
        self.filelist = [line.strip('\n').split('\t') for line in self.filelist]
        self.train_labels = torch.LongTensor([int(l[1]) for l in self.filelist])
        self.dataset_location = dataset_location
        self.transform = transform
        
    def __getitem__(self, index):
        data = Image.open(os.path.join(self.dataset_location, self.filelist[index][0]))

        label = np.array(int(self.filelist[index][1]), dtype=np.long)
        if self.transform:
            data = self.transform(data)
            
        return data, label
    
    
    def __len__(self):
        return len(self.filelist)
    
    def __repr__(self):
        return "JID_train"

class JID_opst_test(Dataset):
    """
    Open-set evaluation dataset of Jockey Recognition
    """
    def __init__(self, filelist, dataset_location,transform=None):
        self.pairlist = open(filelist).readlines()[1:]
        self.pairlist = [line.strip('\n').split('\t') for line in self.pairlist]
        self.dataset_location = dataset_location
        self.transform = transform

    def __getitem__(self, index):
        name1 = self.pairlist[index][0]
        name2 = self.pairlist[index][1]
        sameflag = int(self.pairlist[index][2])
        img1 = Image.open(os.path.join(self.dataset_location, name1))
        img2 = Image.open(os.path.join(self.dataset_location, name2))

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)

        return (img1, img2, sameflag)

    def __len__(self):
        return len(self.pairlist)
    
    def __repr__(self):
        return "JID_verification"

class JID_svm(Dataset):
    def __init__(self, filelist, dataset_location='.', transform=None):
        self.filelist = open(filelist).readlines()
        self.filelist = [line.strip('\n').split('\t') for line in self.filelist]
        self.train_labels = torch.LongTensor([int(l[1]) for l in self.filelist])
        self.dataset_location = dataset_location
        self.transform= transform

    def __getitem__(self, index):
            data = Image.open(os.path.join(self.dataset_location, self.filelist[index][0]))
            label = np.array(int(self.filelist[index][1]), dtype=np.long)
            
            if self.transform:
                data = self.transform(data) # note that if you want numpy array you may first convert Image object to Tensor the get numpy array
                
            
            return data, label

    def __len__(self):
        return len(self.filelist)
    
    def __repr__(self):
        return "JID_Identification"