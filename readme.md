## Prepare the datasets
```bash
ln -s ${YOUR_TRAIN_DIR} ./data/train
ln -s ${YOUR_TEST_DIR}  ./data/test
```
## Train
python train.py

## Evaluate
python submit.py



### Generate the samples list
python split_train_val.py
