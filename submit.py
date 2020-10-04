import torch.nn.functional as F
from model.resnet import *
from dataset.mydataset import TestDataset
from torch.utils.data import Dataset, DataLoader
from cfg import *
import torch
from pprint import pprint
from utils import *
import json


def get_predcitions(model, dataloader, device):
    model.eval()

    results = []

    images = []
    probs = []

    with torch.no_grad():
        for sample in dataloader:
            x = sample[0].to(device)
            x_path = sample[1][0]

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(x[0].cpu())
            probs.append(y_prob[0].cpu())

            results.append({"image_name": x_path.split('/')[-1], "category": LABLE2SUBMITNAME[top_pred[0].item()],
                            "score": round(y_prob[0][top_pred[0]].cpu().item(),5)})

    images = torch.cat(images, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, probs, results


def _main():
    dataset_test = TestDataset(test_imgs_root="./data/test", transforms=test_transforms)
    dataloader_test = DataLoader(dataset_test, batch_size=1)

    model = MyResNet("resnet101", pretrained=True, num_classes=3).to(DEVICE)
    save_dict = torch.load("./pretrained-resnet101-best-model_99.12.pt")
    model.load_state_dict(save_dict["model"])

    images, probs, results = get_predcitions(model, dataloader_test, DEVICE)

    with open("result.json", "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    _main()
