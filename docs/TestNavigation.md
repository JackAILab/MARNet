## Testing

### Prepare H5f Data

1. First we need to open the test.py file, find the commented code from line 133 to line 135, and remove the comment.
   When these three lines of code are executed, test.py will be used to prepare the H5f file for training. When these three lines of code are not executed, test.py will test the model that we choose.

```python
# ================ H5f file for rewriting data ================ # # line132
preparePDData(data_path = test_data_path, patch_size=256, stride=256 ,data_type='test') # line133
print("prepare_test_PDData successful!") # line134
exit() # line135
```

2. execute test.py

```
python test.py
```

### Train

1. After preparing the H5f file, we need to block the three lines of code mentioned earlier.

```python
# ================ H5f file for rewriting data ================ # # line132
# preparePDData(data_path = test_data_path, patch_size=256, stride=256 ,data_type='test') # line133
# print("prepare_test_PDData successful!") # line134
# exit() # line135
```

2. choose the model

In the `ModelLog/` directory, we have saved a series of models. Now we can select some of these models for testing. start is the number of the first model to be tested, and end is the number of the last model to be tested.

```
for modelEpoch in range(start,end,gap):
    # load model
    model = torch.load(f'./ModelLog/model_e_{modelEpoch}.pth')
```

3. execute test.py

```
python test.py
```