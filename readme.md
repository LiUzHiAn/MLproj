## Prepare the datasets
```bash
$ ln -s ${YOUR_TRAIN_DIR} ./data/train
$ ln -s ${YOUR_TEST_DIR}  ./data/test
```

## Generate the samples list
python split_train_val.py

## Train
```bash
$ python train.py
```
训练的每个epoch都会在验证集上对模型进行评估

## Evaluate
```bash
$ python eval.py
```
## Inference
对测试集进行推理
```bash
$ python submit.py
```
对验证集进行推理
```bash
$ python val_inference.py
```

## Visualization
```bash
$ cd vis
$ python vis_nn_focus.py
$ python vis_cam.py
```
Turn on the `verbose` switch at Line20 in `vis_nn_focus.py` for real-time visualization.

## Pre-trained models
Please contact me via csliuzhian at mail.scut.edu.cn