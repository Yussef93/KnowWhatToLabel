from PIL import Image,ImageChops
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,SubsetRandomSampler
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import cv2
import numbers
from PIL import ImageFilter
from Code import augmentations


class CellsDataset(Dataset):
    def __init__(self, root_dir, dataset_selection, split=False, train_valid='train', target=None,k_shot=None,
                 split_type='train_valid',transform=None):
        self.root_dir = root_dir
        self.selection = dataset_selection
        self.target = target
        self.k_shot = k_shot
        self.transform = transform
        #self.eval = eval
        self.split = split
        self.split_type = split_type
        self.train_valid = train_valid
        self.ground_truth_train = []
        self.ground_truth_train_pl = []
        self.ground_truth_valid = []
        self.ground_truth_test = []
        self.images_train = []
        self.images_valid = []
        self.images_test = []

        for set in dataset_selection:
            ground_truth_pl_prefix = set + '/Groundtruth_PL/'
            ground_truth_prefix = set+'/Groundtruth/'
            image_prefix = set+'/Image/'
            #if set == 'TNBC':
                #self.root_dir = '../Datasets/FewShot/Microscopy/Target/Selection_6/FinetuneSamples/3-shot/preprocessed/'
            if self.split == True:

                if self.split_type == 'train_valid':
                    file_names_g = sorted([self.root_dir + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])
                    file_names_i = sorted(
                        [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                         f[0] != '.'])[:len(file_names_g)]

                    f = open('../Datasets/Train_Valid_40/valid_ids_{}.pickle'.format(set), 'rb')
                    valid_samples = pickle.load(f)
                    f.close()
                    for i in range(len(file_names_g)):
                        if i not in valid_samples:
                            self.ground_truth_train.append(file_names_g[i])
                            self.images_train.append(file_names_i[i])
                        else:
                            self.ground_truth_valid.append(file_names_g[i])
                            self.images_valid.append(file_names_i[i])

                elif self.split_type == 'train_valid_test':
                    self.ground_truth_train =  sorted([self.root_dir + 'Train/' + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + 'Train/' + ground_truth_prefix) if f[0] != '.'])
                    self.images_train = sorted([self.root_dir + 'Train/' +image_prefix + f for f in os.listdir(self.root_dir +'Train/' + image_prefix)
                                                if f[0] != '.'])[:len(self.ground_truth_train)]

                    self.ground_truth_valid = sorted([self.root_dir + 'Valid/' + ground_truth_prefix + f for f in
                                                      os.listdir(self.root_dir + 'Valid/' + ground_truth_prefix) if
                                                      f[0] != '.'])
                    self.images_valid = sorted([self.root_dir + 'Valid/' + image_prefix + f for f in
                                                os.listdir(self.root_dir + 'Valid/' + image_prefix) if
                                                f[0] != '.'])[:len(self.ground_truth_valid)]

                    self.ground_truth_test = sorted([self.root_dir + 'Test/' + ground_truth_prefix + f for f in
                                                      os.listdir(self.root_dir + 'Test/' + ground_truth_prefix) if
                                                      f[0] != '.'])
                    self.images_test = sorted([self.root_dir + 'Test/' + image_prefix + f for f in
                                                os.listdir(self.root_dir + 'Test/' + image_prefix) if
                                                f[0] != '.'])[:len(self.ground_truth_test)]


            else:
                self.ground_truth_train +=  sorted([self.root_dir + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])


                self.ground_truth_train_pl += sorted([self.root_dir + ground_truth_pl_prefix + f for f in
                                                   os.listdir(self.root_dir + ground_truth_pl_prefix) if f[0] != '.'])
                self.images_train +=  sorted(
                        [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                         f[0] != '.'])[:len(self.ground_truth_train)]


    def __len__(self):
        if self.split == True:
            if self.train_valid == 'train':
                return(len(self.ground_truth_train))
            elif self.train_valid == 'valid':
                return(len(self.ground_truth_valid))
            elif self.train_valid == 'test':
                return(len(self.ground_truth_test))
        else:
            return(len(self.ground_truth_train))
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.images_train[idx])
        ground_truth = Image.open(self.ground_truth_train[idx])
        ground_truth_pl = Image.open(self.ground_truth_train_pl[idx])

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],std=[0.5])])
        if self.transform:
            image = self.transform(image)
            ground_truth = transforms.ToTensor()(ground_truth)
            return image,ground_truth
        else:
            image_contrast = augmentations.AutoContrast(image,1.4)
            image_contrast = transform(image_contrast)
            val = 1.3
            if self.target=='B5' or self.target=='B39':
                val = 1.7
            image_brightness = augmentations.Brightness(image,val)
            image_brightness = transform(image_brightness)

            image_Sharpness = augmentations.Sharpness(image,val)
            image_Sharpness = transform(image_Sharpness)

            image = transform(image)

            img_augment = {'original': image, 'contrast':image_contrast,'sharpness':image_Sharpness,'brightness':image_brightness}


            ground_truth = transforms.ToTensor()(ground_truth)
            ground_truth_pl = transforms.ToTensor()(ground_truth_pl)

            return image,ground_truth,ground_truth_pl,img_augment,idx


