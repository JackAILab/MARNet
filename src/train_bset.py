
from utils import *
from utils import DiffLoss,SimilarityKL
from dataset_h5f import BasicDataset, scipy_rotate
from h5fMake import preparePDData
from cal_acc import MyAverageMeter
from curses.ascii import isdigit
from metrics import Multi_accuracy
from metrics import *
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd 
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn  
import math
import argparse
from optparse import Option
from email import parser
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.backends.cudnn.enabled = False  


df = pd.read_csv("/home/huangjiehui/Project/PDAnalysis/ReviseStage/MakeData1121/new_23_train.csv")
df_shuffle = df.sample(frac=1, random_state=100)  
df_train = df_shuffle

train_data_path = list(df_train["data_path"])
train_labels = list(df_train["label_list"])

train_transform = None
test_transform = None


# preparePDData(data_path = train_data_path, patch_size=256, stride=256 ,data_type='train')
# print("prepare_train_PDData successful!")
# exit()


train_set = BasicDataset(train_data_path, train_labels, "train", train_transform)
train_loader = DataLoader(train_set, batch_size=3, shuffle=True, num_workers=1)
print("Data Load success!")


from model_best_lstm import generate_model
model = generate_model(model_depth=101, n_input_channels=1, n_classes=2).cuda()
# model = torch.load('/home/huangjiehui/Project/PDAnalysis/ReviseStage/yufc_OrgTwoClasses/run/exp4-pretrained/20.pt')
model.load_state_dict(torch.load('/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/run/lstm/15.pt'))
continue_to_train_epoch = 16

print("model generate success!")


parser = argparse.ArgumentParser(description="JackNet_Train")
parser.add_argument(
    "--milestone", type=int, default=[30, 50, 80], help="When to decay learning rate"
)
opt = parser.parse_args()
criterion = nn.CrossEntropyLoss()

criterion_diff = DiffLoss()       
criterion_simi = SimilarityKL()
w_simi=0.2 

optimizer = optim.Adam(params=model.parameters(), lr=5*(1e-5))  
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=opt.milestone, gamma=0.2
)


model.train()  


epochs = 50
save_model_epoch = 2 

exp_number = 'best-lstm'
mode = 'lstm'

log_save_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/Log/{exp_number}/'
model_save_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/run/{exp_number}/'
in_model_save_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/run/{exp_number}/in/'
in_log_save_path = f'/home/huangjiehui/Project/PDAnalysis/ReviseStage/XiaoRong/Log/{exp_number}/in/'
os.system("mkdir -p " + log_save_path)
os.system("mkdir -p " + model_save_path)
os.system("mkdir -p " + in_model_save_path)
os.system("mkdir -p " + in_log_save_path)




train_log_path = log_save_path + 'train.log'
f2 = open(train_log_path, 'w')


def code2name(code):
    _code = str(code)

    code = ""
    for i in range(0, len(_code)):
        if isdigit(_code[i]):
            code += _code[i]
    name = ""
    i = 0
    if code[i] == "1":
        name += "HC"
    elif code[i] == "2":
        name += "PD"
    i += 1
    for j in range(i, i + 3):
        name += code[j]
    i += 3
    name += "_"
    name += code[i]
    name += "_"
    i += 1
    for j in range(i, len(code)):
        name += code[j]
    return name


print("begin to train!")
if __name__ == "__main__":
    for i in range(continue_to_train_epoch, epochs): 

        person_posibility_path = in_log_save_path + f'soft_c_{i}.txt' 
        f = open(person_posibility_path, "w")
        msg = "patient_name,true,pred,loss"
        f.write(msg + "\n")

        hard_path = in_log_save_path + f'hard_c_{i}.txt'
        f1 = open(hard_path, 'w')
        msg = "patient_name,true,pred"
        f1.write(msg + '\n')
        
        
        current_epoch_acc = []
        current_epoch_loss = []
        cnt = 0
        length = len(train_loader)
        for data in train_loader:
            print(f"     training: {cnt}/{length}", end='\r')
            cnt += 1
            image, label, name_code = data
            name_list = []
            for j in range(0, len(name_code)):
                name_list.append(code2name(name_code[j]))

            image = image.cuda()
            label = label.cuda()

            logit = model(image)
            
            loss_mse = criterion(logit, label)
            
            if mode == 'ca' or mode == 'sa' or mode == 'ca-sa' or mode == 'none':   
                loss_simi = 0
                loss_diff = 0
            elif mode == 'lstm' or mode == 'lstm-ca' or mode == 'lstm-ca-sa':
                loss_simi = criterion_diff(model.x_origin, model.x_lstm) 
                loss_diff = 0
            elif mode == 'all':
                loss_simi = criterion_diff(model.x_origin, model.x_lstm) 
                loss_diff = criterion_simi(model.x_lstm, model.x_ca, model.x_sa)        
            else:
                assert(False)
            
            
            loss = loss_mse + 0.05*loss_simi + 0.5*loss_diff # loss_mse 0.6748 loss_simi=tensor(9.9700, device='cuda:0', grad_fn=<MeanBackward0>)  loss_diff=0.4253

            
            pred = torch.argmax(logit, dim=-1)
            posibility_list = logit.cpu().detach().numpy()
            acc = accuracy(label.cpu().numpy(), pred.cpu().numpy())

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            current_epoch_acc.append(acc)
            current_epoch_loss.append(loss)



            for j in range(0, len(name_code)): 
                msg = (
                    f"{name_list[j]},{str(label[j].cpu().numpy())},{posibility_list[j]},{loss.item()}"
                )
                f.write(msg + "\n")

            for j in range(0, len(name_code)):
                msg = f"{name_list[j]}, {str(label[j].cpu().numpy())}, {pred.cpu().numpy()[j]}"
                f1.write(msg + '\n')
                

            
            in_epoch_loss = sum(current_epoch_loss) / len(current_epoch_loss)
            in_epoch_acc = sum(current_epoch_acc) / len(current_epoch_acc)
            msg_in = "epoch: %d, %d/%d, train_loss: %.4f, train_acc: %.4f" % (
                i,
                cnt,
                length,
                in_epoch_loss,
                in_epoch_acc,
            )
            print(msg_in)
            
                

        epoch_loss = sum(current_epoch_loss) / len(current_epoch_loss)
        epoch_acc = sum(current_epoch_acc) / len(current_epoch_acc)
        
        msg = "epoch: %d, train_loss: %.4f, train_acc: %.4f" % (
            i,
            epoch_loss,
            epoch_acc,
        )
        print(msg)
        f2.write(msg + "\n")
        if i % save_model_epoch == 0:
            path = model_save_path + f"./{i}.pt"
            torch.save(model.state_dict(), path)
            print(f"save {i}.pt success")