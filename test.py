
from utils import *
from dataset_h5f import BasicDataset
from h5fMake import preparePDData
from cal_acc import MyAverageMeter
from curses.ascii import isdigit
from metrics import Multi_accuracy
from metrics import *
import pandas as pd 
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn 
import math
import argparse
from email import parser
import torch
import os
from sklearn import metrics
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# You can choose to specify a graphics card here

torch.backends.cudnn.enabled = False


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


# testing function
def eval(model, eval_loader, criterion):
    model.eval()

    running_loss = MyAverageMeter()
    running_acc = MyAverageMeter()

    iterator = 0
    ouput_info_step = 30
    print_epoch = 0
    f = open('./ResDir/TestAcc.txt','w')
    f_each = open('./ResDir/TestLog.txt','w')
    from cal_acc import GetTarget
    target = GetTarget()
    for data in eval_loader:
        image, label, name = data
        image = image.cuda()
        label = label.cuda()
        name_list = []
        for i in range(0, len(name)):
            name_list.append(code2name(name[i]))      

        with torch.no_grad():
            logit = model(image)
            loss = criterion(logit, label)
            pred = torch.argmax(logit, dim=-1)
            posibility_list = logit.cpu().detach().numpy()
            # ----- Output prediction information ----- #
            for i in range(0, len(name)): 
                msg = f'{name_list[i]},{str(label[i].cpu().numpy())},{str(pred[i].cpu().numpy())}, {posibility_list[i]}'
                f_each.write(msg+'\n')
            acc = accuracy(label.cpu().numpy(), pred.cpu().numpy())
        running_loss.update(loss.item(), image.size(0))
        running_acc.update(acc, image.size(0))
        
        # get targets
        true_ = label.cpu().numpy().tolist()
        pred_ = pred.cpu().numpy().tolist()
        for i in range(0,len(true_)):
            target.insert([true_[i],pred_[i]])
        
        
        cur_loss = running_loss.get_average()
        cur_acc = running_acc.get_average()
        if iterator % ouput_info_step == 0:
            print_epoch += 1
            msg = f'epoch: {print_epoch}, loss: {cur_loss}, acc: {cur_acc}'
            print(msg)
            f.write(msg+'\n')
        iterator += 1
    epoch_loss = running_loss.get_average()
    epoch_acc = running_acc.get_average()
    running_loss.reset()
    running_acc.reset()
    
    epoch_recall = target.getRecall()
    epoch_precision = target.getPrec()
    epoch_sen = target.getSPE()
    epoch_F1 = target.getF1()
    
    model.train()
    return epoch_loss, epoch_acc,epoch_recall,epoch_precision,epoch_sen,epoch_F1



# load data
df2 = pd.read_csv(
    './Data/TestData.csv')
df2_shuffle = df2.sample(frac=1, random_state=100)
df_test = df2_shuffle

test_data_path = list(df_test['data_path'])
test_labels = list(df_test['label_list'])


print()

train_transform = None  
test_transform = None

# ================ H5f file for rewriting data ================ #
# preparePDData(data_path = test_data_path, patch_size=256, stride=256 ,data_type='test')
# print("prepare_test_PDData successful!")
# exit()

test_set = BasicDataset(test_data_path, test_labels, 'test', test_transform)
test_loader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=1)
print("Data Load success!")


f = open('ResDir/AllEpochAcc.txt','a')
start = 90
end = 1501

# Import all. pth files in a loop and test all existing models
for modelEpoch in range(630,631,60):
    # load model
    model = torch.load(f'./ModelLog/model_e_{modelEpoch}.pth')
    # Model parameter settings
    parser = argparse.ArgumentParser(description="JackNet_Train")
    parser.add_argument("--milestone", type=int,
                        default=[30, 50, 80], help="When to decay learning rate")
    opt = parser.parse_args()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.milestone, gamma=0.2)

    print("begin to test...")
    test_loss, test_acc,test_recall,test_precision , sen, F1= eval(model, test_loader, criterion)
    msg = f'model Epoch: {modelEpoch}, test_loss: {test_loss}, test_acc: {test_acc},test_recall:{test_recall},test_precision:{test_precision}, test_sen: {sen}, test_f1: {F1}'
    f.write(msg+'\n')
    print(msg)