## Training

### Prepare H5f Data

1. First we need to open the train.py file, find the commented code from line 42 to line 44, and remove the comment.
   When these three lines of code are executed, train.py will be used to prepare the H5f file for training. When these three lines of code are not executed, trian.py will train the model.

```python
# ================ H5f file for rewriting data ================ # # line41
preparePDData(data_path = train_data_path, patch_size=256, stride=256 ,data_type='train') # line42
print("prepare_train_PDData successful!") # line43
exit() # line44
```

2. execute train.py

```
python train.py
```

**Tips: the paths in the codes need to be modified.**

### Train

1. After preparing the H5f file, we need to block the three lines of code mentioned earlier and reexecute train.py.

```python
# ================ H5f file for rewriting data ================ # # line41
# preparePDData(data_path = train_data_path, patch_size=256, stride=256 ,data_type='train') # line42
# print("prepare_train_PDData successful!") # line43
# exit() # line44
```

2. execute train.py

```
python train.py
```

As the training progresses, the model will be saved to the path of `ModelLog/`

**Tips: the paths in the codes need to be modified.**