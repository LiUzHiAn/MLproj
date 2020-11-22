import torch
import cv2
from vis.grad_cam import Grad_CAM, visualize_gradcam_with_img, denormalize
from dataset.mydataset import TestDataset
from torch.utils.data import DataLoader
from cfg import test_transforms, DEVICE
from model.resnet import MyResNet
from model.vgg_net import MyVGGNet
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    dataset_test = TestDataset(test_imgs_root="../data/test", transforms=test_transforms)
    dataloader_test = DataLoader(dataset_test, batch_size=1)
    model = MyResNet("resnet50", pretrained=True, num_classes=3).to(DEVICE)
    # model = MyVGGNet("vgg19", pretrained=True, num_classes=3).to(DEVICE)
    save_dict = torch.load("../ckpt/pretrained_resnet50-best-model.pt")
    model.load_state_dict(save_dict["model"])

    grad_cam = Grad_CAM(model, target_layer="model.layer4.2")
    # grad_cam = Grad_CAM(model, target_layer="model.features.35")

    for sample in dataloader_test:
        x = sample[0].to(DEVICE).requires_grad_()
        gcam = grad_cam.visualize(x, None)

        visualize_gradcam_with_img(gcam.detach().cpu().numpy()[0][0],
                                   denormalize(x[0].cpu().detach()),
                                   save_fig_name=os.path.join("resnet50_grad_cam", sample[1][0].split('/')[-1]))
