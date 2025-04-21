import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image, ImageFile

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        if if_train:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        scores_names = [f'score{i}' for i in range(2,12)]
        y = np.array([row[k] for k in scores_names])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}')
        image = default_loader(image_path)
        x = self.transform(image)
        return x, p.astype('float32')

class AADB(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(224, scale=(1.05, 1.25)),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        #score_names=['score']
        y=np.array(row['score'])
        #y=np.array([row[k] for k in score_names])

        image_id = row['ImageFile']


        image_path = os.path.join(self.images_path, image_id)
        image = default_loader(image_path)

        #image = image.resize((224, 224))
        x = self.transform(image)
        return x,y.astype('float32')

class flicker(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(224, scale=(1.05, 1.25)),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        #score_names=['score']
        y=np.array((row['score'])-1)/4
        #y=np.array([row[k] for k in score_names])

        image_id = row['ImageFile']


        image_path = os.path.join(self.images_path, image_id)
        image = default_loader(image_path)

        #image = image.resize((224, 224))
        x = self.transform(image)
        return x,y.astype('float32')
class PARADataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        if if_train:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row['aestheticScore_mean']/5])
        image_id = row['imageName']
        session_id = row['sessionId']
        image_path = os.path.join(self.images_path, session_id,image_id)
        image = default_loader(image_path)
        x = self.transform(image)
        return x, y.astype('float32')
