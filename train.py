from dataset.mydataset import *
from model.cbam_resnet import ResidualNet
import torch.optim as optim
import time
from utils import *
from tensorboardX import SummaryWriter
from cfg import *
import pandas as pd
from shutil import copyfile


def _main():
    # splitter = TrainValSplitter(imgs_path_root="./data/train")
    # train_samples, val_samples = splitter.setup()

    df_train = pd.read_csv('./resources/train.csv')
    train_samples = [tuple(x) for x in df_train.to_numpy()]

    df_val = pd.read_csv('./resources/val.csv')
    val_samples = [tuple(x) for x in df_val.to_numpy()]

    dataset_train = MyDataset(train_samples, transforms=train_transforms)
    dataset_val = MyDataset(val_samples, transforms=test_transforms)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = ResidualNet('ImageNet', depth=101, num_classes=3, att_type="CBAM").to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    params_to_update = model.parameters()
    # print param to learn (i.e., the unfrozen layers )
    print("Params to learn:")
    if PRETRAIN:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                pass
                # print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=START_LR, momentum=0.9)

    # optimizer = optim.Adam(model.parameters(), lr=START_LR)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_valid_loss = float('inf')

    writter = SummaryWriter(log_dir="./logs/%s" % EXP_NAME)
    # copy current exp setting
    copyfile("./cfg.py", os.path.join("./logs/%s" % EXP_NAME, "cfg.py"))
    for epoch in range(NUM_EPOCHS):

        start_time = time.monotonic()

        writter.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch)

        train_loss, train_acc_1, train_acc_2 = train(model, dataloader_train, optimizer, criterion, DEVICE)
        writter.add_scalar("loss/train_loss", train_loss, global_step=epoch)
        writter.add_scalar("loss/train_acc_1", train_acc_1, global_step=epoch)
        writter.add_scalar("loss/train_acc_2", train_acc_2, global_step=epoch)

        lr_scheduler.step()

        valid_loss, valid_acc_1, valid_acc_2 = evaluate(model, dataloader_val, criterion, DEVICE)
        writter.add_scalar("loss/valid_loss", valid_loss, global_step=epoch)
        writter.add_scalar("loss/valid_acc_1", valid_acc_1, global_step=epoch)
        writter.add_scalar("loss/valid_acc_2", valid_acc_2, global_step=epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(save_dict, '%s-best-model.pt' % EXP_NAME)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | '
              f'Train Acc @2: {train_acc_2 * 100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | '
              f'Valid Acc @2: {valid_acc_2 * 100:6.2f}%')


def train(model, dataloader, optimizer, criterion, DEVICE):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0

    model.train()

    for idx, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc_1, acc_2 = calculate_topk_accuracy(y_pred, y.unsqueeze(dim=-1))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_2 += acc_2.item()

    epoch_loss /= len(dataloader)
    epoch_acc_1 /= len(dataloader)
    epoch_acc_2 /= len(dataloader)

    return epoch_loss, epoch_acc_1, epoch_acc_2


def evaluate(model, dataloader, criterion, DEVICE):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_2 = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_2 = calculate_topk_accuracy(y_pred, y.unsqueeze(dim=-1))

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_2 += acc_2.item()

    epoch_loss /= len(dataloader)
    epoch_acc_1 /= len(dataloader)
    epoch_acc_2 /= len(dataloader)

    return epoch_loss, epoch_acc_1, epoch_acc_2


if __name__ == '__main__':
    _main()
