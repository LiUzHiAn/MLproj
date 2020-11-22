# from model.cbam_pretrained_resnet import AttentionResNet
from dataset.mydataset import *
# from model.cbam_resnet import ResidualNet
from model.vgg_net import MyVGGNet
from model.resnet import MyResNet
from utils import *

from cfg import *
import pandas as pd


def _main():
    # splitter = TrainValSplitter(imgs_path_root="./data/train")
    # train_samples, val_samples = splitter.setup()

    # df_train = pd.read_csv('./resources/train.csv')
    # train_samples = [tuple(x) for x in df_train.to_numpy()]

    df_val = pd.read_csv('./resources/val.csv')
    val_samples = [tuple(x) for x in df_val.to_numpy()]

    dataset_val = MyDataset(val_samples, transforms=test_transforms)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # model = ResidualNet('ImageNet', depth=101, num_classes=3, att_type="CBAM", torch_pretrained=True).to(DEVICE)
    # model = MyVGGNet("vgg19", pretrained=True, num_classes=3).to(DEVICE)

    model = MyResNet("resnet50", pretrained=True, num_classes=3).to(DEVICE)
    save_dict = torch.load("./ckpt/pretrained_resnet50_noDataAug-best-model.pt")
    model.load_state_dict(save_dict["model"])

    valid_acc_1, valid_acc_2 = evaluate(model, dataloader_val, DEVICE)

    print(f'\t Valid Acc @1: {valid_acc_1 * 100:6.2f}% | '
          f'Valid Acc @2: {valid_acc_2 * 100:6.2f}%')


def evaluate(model, dataloader, DEVICE):
    epoch_acc_1 = 0
    epoch_acc_2 = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)

            acc_1, acc_2 = calculate_topk_accuracy(y_pred, y.unsqueeze(dim=-1))

            epoch_acc_1 += acc_1.item()
            epoch_acc_2 += acc_2.item()

    epoch_acc_1 /= len(dataloader)
    epoch_acc_2 /= len(dataloader)

    return epoch_acc_1, epoch_acc_2


if __name__ == '__main__':
    _main()
