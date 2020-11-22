import torch
import cv2
from vis.guided_propagation import normalize, Guided_backprop
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
    save_dict = torch.load("../ckpt/pretrained_resnet50-best-model.pt")

    # model = MyVGGNet("vgg19", pretrained=True, num_classes=3).to("cpu")
    # save_dict = torch.load("../pretrained_vgg19-best-model.pt", map_location="cpu")

    model.load_state_dict(save_dict["model"])
    guided_bp = Guided_backprop(model)

    # verbose = False

    # plt_save_dir = "./gradient_imgs"
    # if not os.path.exists(plt_save_dir):
    #     os.mkdir(plt_save_dir)

    for sample in dataloader_test:
        x = sample[0].to(DEVICE).requires_grad_()
        result = guided_bp.visualize(x, None)

        result = normalize(result)
        cv2.imwrite(os.path.join("resnet50_gbp", sample[1][0].split('/')[-1]),
                   (result * 255).astype(np.uint8)[:, :, ::-1])

        # ========== 把输入和输出一并可视化 ===============
        # input_arr = np.transpose(
        #     x[0].detach().cpu().mul_(torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        #         .add_(torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        #     , axes=[1, 2, 0])
        #
        # if verbose:
        #     cv2.imshow("input", (input_arr.numpy() * 255).astype(np.uint8)[:, :, ::-1])
        #     cv2.imshow("guided_propa", ((result * 255).astype(np.uint8)[:, :, ::-1]))
        #     cv2.waitKey(0)
        # else:
        #     f, axarr = plt.subplots(1, 2)
        #     axarr[0].imshow(input_arr)
        #     axarr[0].set_title("Input Image")
        #     axarr[0].axis('off')
        #
        #     axarr[1].imshow(result)
        #     axarr[1].set_title("Guided Gradient")
        #     axarr[1].axis('off')
        #     plt.savefig(os.path.join(plt_save_dir, (sample[1][0].split("/"))[-1]))
        #     plt.close()
