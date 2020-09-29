from torch.utils.data import Dataset, DataLoader
import torch
import os
import glob
from utils import *
import numpy as np
from PIL import Image
import cv2

np.random.seed(2020)


class TrainValSplitter(object):
    def __init__(self, imgs_path_root, train_val_split=0.85):
        super(TrainValSplitter, self).__init__()
        self.imgs_path_root = imgs_path_root
        self.train_val_split = train_val_split

    def setup(self):
        """准备好图片和对应的label"""
        normal_imgs = glob.glob(os.path.join(self.imgs_path_root, "normal", "*.jpg"))
        phone_imgs = glob.glob(os.path.join(self.imgs_path_root, "phone", "*.jpg"))
        smoke_imgs = glob.glob(os.path.join(self.imgs_path_root, "smoke", "*.jpg"))

        normal_imgs_indexs = np.arange(len(normal_imgs))
        phone_imgs_indexs = np.arange(len(phone_imgs))
        smoke_imgs_indexs = np.arange(len(smoke_imgs))
        np.random.shuffle(normal_imgs_indexs)  # in-place shuffle
        np.random.shuffle(phone_imgs_indexs)  # in-place shuffle
        np.random.shuffle(smoke_imgs_indexs)  # in-place shuffle

        train_normal_imgs = [normal_imgs[idx] \
                             for idx in normal_imgs_indexs[:int(self.train_val_split * len(normal_imgs))]
                             ]
        val_normal_imgs = [normal_imgs[idx] \
                           for idx in normal_imgs_indexs[int(self.train_val_split * len(normal_imgs)):]
                           ]

        train_phone_imgs = [phone_imgs[idx] \
                            for idx in phone_imgs_indexs[:int(self.train_val_split * len(phone_imgs))]
                            ]
        val_phone_imgs = [phone_imgs[idx] \
                          for idx in phone_imgs_indexs[int(self.train_val_split * len(phone_imgs)):]
                          ]

        train_smoke_imgs = [smoke_imgs[idx] \
                            for idx in smoke_imgs_indexs[:int(self.train_val_split * len(smoke_imgs))]
                            ]
        val_smoke_imgs = [smoke_imgs[idx] \
                          for idx in smoke_imgs_indexs[int(self.train_val_split * len(smoke_imgs)):]
                          ]

        train_samples = []
        for item in train_normal_imgs:
            train_samples.append((item, LABEL2INDEX["normal"]))
        for item in train_phone_imgs:
            train_samples.append((item, LABEL2INDEX["phone"]))
        for item in train_smoke_imgs:
            train_samples.append((item, LABEL2INDEX["smoke"]))
        train_indexs = np.arange(len(train_samples))
        np.random.shuffle(train_indexs)
        train_samples_shuffled = [train_samples[idx] for idx in train_indexs]

        val_samples = []
        for item in val_normal_imgs:
            val_samples.append((item, LABEL2INDEX["normal"]))
        for item in val_phone_imgs:
            val_samples.append((item, LABEL2INDEX["phone"]))
        for item in val_smoke_imgs:
            val_samples.append((item, LABEL2INDEX["smoke"]))
        val_indexs = np.arange(len(val_samples))
        np.random.shuffle(val_indexs)
        val_samples_shuffled = [val_samples[idx] for idx in val_indexs]
        return train_samples_shuffled, val_samples_shuffled


class MyDataset(Dataset):
    def __init__(self, samples, transforms=None):
        super(MyDataset, self).__init__()
        self.samples = samples
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        verbose = False

        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        # if img.size(0) == 4:
        #     print("RGBA IMAGES FOUND!!!!")
        if verbose:
            verbose_img = img.numpy()
            verbose_img[0] = verbose_img[0] * 0.229 + 0.485
            verbose_img[1] = verbose_img[1] * 0.224 + 0.456
            verbose_img[2] = verbose_img[2] * 0.225 + 0.406
            verbose_img = (verbose_img * 255).astype(np.uint8)
            cv2.imshow(INDEX2LABEL[label], np.transpose(verbose_img, axes=[1, 2, 0])[:, :, ::-1])
            cv2.waitKey(0)
            print(path)

        return img, label


class TestDataset(Dataset):
    def __init__(self, test_imgs_root, transforms=None):
        super(TestDataset, self).__init__()
        self.imgs = glob.glob(os.path.join(test_imgs_root, "*.jpg"))
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.imgs[index]  # img and corresponding path
