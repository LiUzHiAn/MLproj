from pathlib import Path
from dataset.mydataset import *
import pandas as pd

if __name__ == "__main__":
    splitter = TrainValSplitter('./data/train', 0.9)

    train_samples, val_samples = splitter.setup()

    n_samples = len(train_samples) + len(val_samples)
    print(n_samples)

    name = ['img_path', 'label']
    df_train = pd.DataFrame(columns=name, data=train_samples)
    df_train.to_csv('./resources/train.csv', encoding='utf-8', index=False)

    df_val = pd.DataFrame(columns=name, data=val_samples)
    df_val.to_csv('./resources/val.csv', encoding='utf-8', index=False)

    all_samples = train_samples[:] + val_samples[:]
    np.random.shuffle(all_samples)
    df_all = pd.DataFrame(columns=name, data=all_samples)
    df_all.to_csv('./resources/all.csv', encoding='utf-8', index=False)
