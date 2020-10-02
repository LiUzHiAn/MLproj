from torch.utils.data import Dataset, DataLoader
import torch
import os
import glob
from utils import *
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


np.random.seed(2020)


class TrainValSplitter(object):
    def __init__(self, imgs_path_root: str, train_val_split=0.85):
        super(TrainValSplitter, self).__init__()
        self.imgs_path_root = imgs_path_root
        self.train_val_split = train_val_split


    def setup(self):
        """prepare image path and correspoding label for the sample"""
        train_samples = []
        val_samples = []


        root_path = Path(self.imgs_path_root)
        sub_classes = [sub_path for sub_path in root_path.iterdir() if sub_path.is_dir()]
        for sub_class in sub_classes:
            label = NAME2LABLE[str(sub_class.name)]
            sub_class_imgs = list(sub_class.glob("*.*"))
            sub_class_samples = [(img, label) for img in sub_class_imgs]

            print(str(sub_class.name), len(sub_class_samples))

            np.random.shuffle(sub_class_samples)
            train_samples = train_samples + sub_class_samples[:int(self.train_val_split * len(sub_class_samples))]
            val_samples = val_samples + sub_class_samples[int(self.train_val_split * len(sub_class_samples)):]


        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)
        return train_samples, val_samples


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
            cv2.imshow(LABLE2NAME[label], np.transpose(verbose_img, axes=[1, 2, 0])[:, :, ::-1])
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
