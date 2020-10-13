import torch.nn.functional as F
from model.resnet import *
from dataset.mydataset import TestDataset
from torch.utils.data import Dataset, DataLoader
from cfg import *
import torch
from pprint import pprint
from utils import *
import json
import pandas as pd
from dataset.mydataset import MyDataset
from submit import get_predcitions


def inference_on_validation_set():
    df_val = pd.read_csv('../resources/val.csv')
    val_samples = [tuple(x) for x in df_val.to_numpy()]
    dataset_val = MyDataset(val_samples, transforms=test_transforms)
    dataloader_val = DataLoader(dataset_val, batch_size=1)

    model = MyResNet("resnet101", pretrained=True, num_classes=3).to(DEVICE)
    save_dict = torch.load("./ckpt/resnet101-best-model_99.47.pt")
    model.load_state_dict(save_dict["model"])

    model.eval()

    results = []
    gts = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader_val):
            x = x.to(DEVICE)

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            results.append({"image_name": val_samples[idx][0],
                            "category": LABLE2SUBMITNAME[top_pred[0].item()],
                            "score": round(y_prob[0][top_pred[0]].cpu().item(), 5)})
            gts.append({"image_name": val_samples[idx][0],
                        "category": LABLE2SUBMITNAME[y.item()]})

    with open("result_val.json", "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    inference_on_validation_set()