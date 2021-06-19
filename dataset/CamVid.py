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

def augmentation(image, label):
    # `.copy()` will create a new image with positive stride, thus avoiding
    # later issues with pytorch. See https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/10
    image = np.flip(image, axis=1).copy()
    label = np.flip(label, axis=1).copy()
  
    return image, label


def augmentation_pixel(image):
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    #r = random.randrange(2)
    #if  r == 0:
    #1) gaussian blur: reduces the noise (low-pass filter that preserves low spatial frequency and reduces image noise)
    #it reduces the level of detail
    #std dev su ogni asse: min=2/max=3 -> viene scelto in modo random in questo intervallo  (da provare a modificare?)
    #kernel_size: maggiore il valore maggiore lo smoothing
    #    transformer = transforms.Compose([transforms.ToTensor(),transforms.GaussianBlur(kernel_size=15, sigma=(2.0, 3.0))])

    #if r == 1:
    #2) altering colors
    #brightness: scelto valore random in [max(0, 1 - brightness), 1 + brightness]
    #contrast: scelto valore random in [max(0, 1 - contrast), 1 + contrast]
    #saturation: scelto valore random in [max(0, 1 - saturation), 1 + saturation]
    #hue: scelto valore random in [-hue, hue]
    #    transformer = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=1, contrast=2, saturation=2, hue=0.2)])

    #if r == 2:
    #3)  convert image to grayscale
    #    transformer = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(num_output_channels=3)])

    #return (
    #  transformer(image).permute(1,2,0).numpy() * 255
    #).astype(np.uint8)
    return image


class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()
        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        # self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_list]
        self.label_info = get_label_info(csv_path)
        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = scale
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        # random crop image
        # =====================================
        # w,h = img.size
        # th, tw = self.scale
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        # img = F.crop(img, i, j, th, tw)
        # =====================================

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train' or self.mode == 'adversarial_train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        # =====================================

        img = np.array(img)
        if self.mode != 'adversarial_train':
          label = Image.open(self.label_list[index])
        else:
          imarray = np.zeros(shape=(2,2,4)) * 255
          label = Image.fromarray(imarray.astype('uint8')).convert('RGBA')

        # crop the corresponding label
        # =====================================
        # label = F.crop(label, i, j, th, tw)
        # =====================================

        # randomly resize label and random crop
        # =====================================
        if self.mode == 'train' or self.mode == 'adversarial_train':
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)

        label = np.array(label)

        # augment image and label
        if self.mode == 'train' or self.mode == 'adversarial_train':
            if random.random() < 0.5:
              img, label = augmentation(img, label)

        # augment pixel image
        if self.mode == 'train' or self.mode == 'adversarial_train':
            # set a probability of 0.5
            if random.random() < 0.5:
              img = augmentation_pixel(img)

        # image -> [C, H, W]

        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'dice':
            # label -> [num_classes, H, W]
            if self.mode != 'adversarial_train':
              label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    data = CamVid(['/data/sqy/CamVid/train', '/data/sqy/CamVid/val'],
                  ['/data/sqy/CamVid/train_labels', '/data/sqy/CamVid/val_labels'], '/data/sqy/CamVid/class_dict.csv',
                  (720, 960), loss='crossentropy', mode='val')
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info('/data/sqy/CamVid/class_dict.csv')
    for i, (img, label) in enumerate(data):
        print(label.size())
        print(torch.max(label))

