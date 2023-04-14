

from utils import *
from dataset_h5f import BasicDataset, scipy_rotate
from h5fMake import preparePDData
from cal_acc import MyAverageMeter
from curses.ascii import isdigit
from metrics import Multi_accuracy
from metrics import *
from torch.optim.lr_scheduler import MultiStepLR
from model import generate_model
import pandas as pd 
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn 
import math
import argparse
from email import parser
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# You can choose to specify a graphics card here

torch.backends.cudnn.enabled = False 

# load data
df = pd.read_csv('/Data/TrainData.csv')
df_shuffle = df.sample(frac=1, random_state=100) 
df_train = df_shuffle

# prepare data
train_data_path = list(df_train['data_path'])
train_labels = list(df_train['label_list'])

train_transform = None 
test_transform = None


# ================ H5f file for rewriting data ================ #
# preparePDData(data_path = train_data_path, patch_size=256, stride=256 ,data_type='train')
# print("prepare_train_PDData successful!")
# exit()


# prepare datasets
train_set = BasicDataset(train_data_path, train_labels,'train', train_transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=1)
print("Data Load success!")



# generate model
model = generate_model(model_depth=101, n_input_channels=1, n_classes=2).cuda()
# We can also choose to import existing models
# modelPath = ''
# model = torch.load(modelPath)
continue_to_train_epoch = 0



print("model generate success!")

parser = argparse.ArgumentParser(description="JackNet_Train")
parser.add_argument("--milestone", type=int,
                    default=[30, 50, 80], help="When to decay learning rate")
opt = parser.parse_args()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-4)  
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=opt.milestone, gamma=0.2)


epochs = 2000

# steps_per_epoch = 10
steps_per_epoch = 30
save_model_epoch = 30
save_path = './ModelLog/'

num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader)) # train iterator times
global_step = 0
global_epoch = 0 + continue_to_train_epoch # the epoch begin to train
save_epoch = 0 # epoch to save a model

running_loss = MyAverageMeter()
running_acc = MyAverageMeter() 

model.train()

Note = open('./ResDir/TrainAcc.txt', 'a')
Note.truncate(0)

person_posibility_path = './ResDir/TrainLog.txt'
f = open(person_posibility_path, 'w')


def code2name(code):
    _code = str(code)
    code = ''
    for i in range(0, len(_code)):
        if isdigit(_code[i]):
            code += _code[i]
    name = ''
    i = 0
    if code[i] == '1':
        name += 'HC'
    elif code[i] == '2':
        name += 'PD'
    i += 1
    for j in range(i, i + 3):
        name += code[j]
    i += 3
    name += '_'
    name += code[i]
    name += '_'
    i += 1
    for j in range(i, len(code)):
        name += code[j]
    return name


print("begin to train!")
# begin to train
if __name__ == '__main__':
    for i in range(num_iter):
        path_i = 0
        for data in train_loader:
            global_step += 1

            image, label, name_code = data
            name_list = []
            for i in range(0, len(name_code)):
                name_list.append(code2name(name_code[i]))

            image = image.cuda()
            label = label.cuda()

            logit = model(image)
            loss = criterion(logit, label)
            pred = torch.argmax(logit, dim=-1)
            posibility_list = logit.cpu().detach().numpy()
            acc = accuracy(label.cpu().numpy(), pred.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), image.size(0))
            running_acc.update(acc, image.size(0))
            
            # save model
            if global_epoch % save_model_epoch == 0:
                save_epoch += 1
                path = save_path + f'/model_e_{global_epoch}.pth'
                torch.save(model,path)

            # output information about the training process
            if global_step % steps_per_epoch == 0:
                for i in range(0, len(name_code)):
                    msg = f'{name_list[i]},{str(label[i].cpu().numpy())},{posibility_list[i]}'
                    f.write(msg+'\n')
                    
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_acc = running_acc.get_average()

                msg = "epoch: %d, train_loss: %.4f, train_acc: %.4f" % (global_epoch, epoch_loss, epoch_acc)
                print(msg)
                Note.write(msg+'\n')
