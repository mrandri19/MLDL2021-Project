import torch
import glob
import os
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
import random
import json

def augmentation(img, label):
  return img, label

def augmentation_pixel(img):
  return img

# Credit goes to https://stackoverflow.com/a/36960495
def onehot_initialization_v2(a):
    ncols = a.max()+1
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, classes_info_path, scale, loss='dice', mode='train'):
        # Init torch's Dataset and set mode
        super().__init__()
        self.mode = mode

        classes_info = json.load(open(classes_info_path))
        camvid_from_idda_index = {
          arr[0]: (11 if arr[1] == 255 else arr[1])
          for arr in classes_info['label2camvid']
        }
        self.func = np.vectorize(camvid_from_idda_index.__getitem__)

        # Get all jpg images paths
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.jpg')))
        self.image_list.sort()
        
        # Get all png images label paths
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        
        # Normalize (standardize) images
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        seed = random.random()
        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # Load image and randomly resize and crop
        img = Image.open(self.image_list[index])
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        img = np.array(img)

        # Load label and randomly resize and crop like the image
        label = Image.open(self.label_list[index])
        if self.mode == 'train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        label = np.array(label)

        # Augmentations
        if self.mode == 'train':
            if random.random() < 0.5:
              img, label = augmentation(img, label)
        if self.mode == 'train':
            if random.random() < 0.5:
              img = augmentation_pixel(img)

        # Convert image to tensor in CHW format
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        # Convert label to tensor in CHW format
        if self.loss == 'dice':
            label = label[:, :, 0]
            label_with_camvid_indices = self.func(label)
            label = onehot_initialization_v2(label_with_camvid_indices)
            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            label = torch.from_numpy(label)
            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            label = torch.from_numpy(label).long()
            return img, label

    def __len__(self):
        return len(self.image_list)

