## Prepare the datasets
```bash
$ ln -s ${YOUR_TRAIN_DIR} ./data/train
$ ln -s ${YOUR_TEST_DIR}  ./data/test
```
## Train
python train.py

## Evaluate
python submit.py



### Generate the samples list
python split_train_val.py

### Visualize NN gradient
```bash
$ cd vis
$ python vis_nn_focus.py
```
Turn on the `verbose` switch at Line20 in `vis_nn_focus.py` for real-time visualization.