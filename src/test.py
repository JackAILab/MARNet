
"""
/home/huangjh/Project/PDAnalysis
"""
'''
nvidia-smi
'''
# 指定卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from utils import *
from dataset_h5f import BasicDataset, scipy_rotate
from h5fMake import preparePDData
from cal_acc import MyAverageMeter
from curses.ascii import isdigit
from metrics import Multi_accuracy
from metrics import *
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd # 对panda做处理
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn # 做深度学习！
import math
import argparse
from optparse import Option
from email import parser
import torch
import os
from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.backends.cudnn.enabled = False  # BUG 1109s


exp_number = 'lstm-sa'

# 存储模型和日志的地方
log_save_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/test/'
model_save_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/run/{exp_number}/'
os.system("mkdir -p " + log_save_path)

def code2name(code):
    _code = str(code)
    # 把数字提取一下
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


def eval(model, eval_loader, criterion):
    model.eval()
    
    acc_list = []
    loss_list = []
    
    f_each = open(log_save_path + f"{exp_number}_res.txt",'w')
    
    cnt = 0
    for data in eval_loader:
        print(f'     {cnt}/{len(eval_loader)}', end='\r')
        cnt += 1
        
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
            # ----- 输出病人的预测信息 ----- #
            for i in range(0, len(name)):  # name_code其实就是batchsize
                msg = f'{name_list[i]},{str(label[i].cpu().numpy())},{str(pred[i].cpu().numpy())}, {posibility_list[i]}'
                f_each.write(msg+'\n')
            acc = accuracy(label.cpu().numpy(), pred.cpu().numpy())

        loss_list.append(loss.item())
        acc_list.append(acc)
    
    epoch_loss = sum(loss_list) / len(loss_list)
    epoch_acc = sum(acc_list) / len(acc_list)

    return epoch_loss, epoch_acc



# load data

df2 = pd.read_csv(
    '/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/data/new_02_test.csv')
df2_shuffle = df2.sample(frac=1, random_state=100)
df_test = df2_shuffle

test_data_path = list(df_test['data_path'])
test_labels = list(df_test['label_list'])



train_transform = None  
test_transform = None


# ================ Added 11.19 Jack 用于重写数据的h5f文件
# preparePDData(data_path = test_data_path, patch_size=256, stride=256 ,data_type='test')
# print("prepare_test_PDData successful!")
# exit()
# ================ Added 11.19 Jack 用于重写数据的h5f文件

test_set = BasicDataset(test_data_path, test_labels, 'test', test_transform)
test_loader = DataLoader(test_set, batch_size = 10, shuffle=False, num_workers=1)
print("Data Load success!")


f = open(log_save_path + "test_result.txt",'a')

from model_lstm_sa import generate_model

keys = [0, 3, 6, 9, 12]
for i in range(0, len(keys)):
    model = generate_model(model_depth=101, n_input_channels=1, n_classes=2).cuda()
    model_save_root_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/run/lstm-sa/{keys[i]}.pt'
    model.load_state_dict(torch.load(model_save_root_path))
    # model = torch.load(model_save_root_path)

    parser = argparse.ArgumentParser(description="JackNet_Train")
    parser.add_argument("--milestone", type=int,
                        default=[30, 50, 80], help="When to decay learning rate")
    opt = parser.parse_args()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)  # Adam这个优化器比较重要
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.milestone, gamma=0.2)  # 变化学习率，很有效的一个参数

    print("begin to test...")
    test_loss, test_acc = eval(model, test_loader, criterion)
    # msg = f'exp: {exp_number}, test_loss: {test_loss}, test_acc: {test_acc}'
    msg = f'name: {exp_number}, epoch: {keys[i]}, test_loss: {test_loss}, test_acc: {test_acc}'
    f.write(msg+'\n')
    print(msg)