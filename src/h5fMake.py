from datas import*  #sss

import torch
import h5py
from curses.ascii import isdigit
from locale import atoi


# =============================== train_data_h5f_path =============================== #
train_save_data_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_Data.he5' # h5f数据路径
train_save_label_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_Data_label.he5' # h5f标签路径
train_save_patient_name_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_name_label.he5' # h5f存病人名字

test_save_data_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_Data.he5' # h5f数据路径
test_save_label_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_Data_label.he5' # h5f标签路径
test_save_patient_name_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_name_label.he5' # h5f存病人名字

def get_name_code(string):
    code=''
    if 'HC' in string:
        code+='1'
    elif 'PD' in string:
        code+='2'
    for i in range(0,len(string)):
        if isdigit(string[i]):
            code+=string[i]
    return code

def preparePDData(data_path, patch_size, stride, data_type): # 传入 256 可以扩大 4 倍的图像数据
    
    h5f_label = 0 # 控制h5f文件的写入
    
    if data_type == 'train':
        data_h5f = h5py.File(train_save_data_path, 'w') # 存数据
        label_h5f = h5py.File(train_save_label_path,'w') # 存标签
        name_h5f = h5py.File(train_save_patient_name_path,'w') # 存名字
    elif data_type == 'test':
        data_h5f = h5py.File(test_save_data_path, 'w') # 存数据
        label_h5f = h5py.File(test_save_label_path,'w') # 存标签
        name_h5f = h5py.File(test_save_patient_name_path,'w') # 存名字
    
    for i in range(len(data_path)): # len(data_path)):  # 这里是循环操作563张图
        if (data_path[i]=="/data1/hjh/ProjectData/PDData/3result_ALL/PD001_1_11_10.npy"):  #BUG 文件不对  _sss
            continue
        image = np.load(data_path[i], allow_pickle=True)
        lbl = int(data_path[i][30:31])
        # 二分类
        if lbl != 0:
            lbl = 1
        # 二分类
        name = data_path[i][42:-4]
        
        image = np.transpose(image, (2, 1, 0)) # (512, 512, 331)
        
        # 对图片数据进行一个多样化的拓展（翻转，斜变等）
        # # image = image[:,:,0:5] # 一次性取了10张数据进来
        image = image.astype(np.float32)
        # #sss -- 数据增强
        # # 这里如果切10张就是10 -- 不切就是20
        for j in range(20):
            im=image[:,:,j]
            frame_3D_last = Im2Patch(im,win=patch_size, stride=stride) # 这里翻转不太清楚要不要删去
            frame_3D_last = frame_3D_last[0,:,:,:]
            frame_3D_last = np.transpose(frame_3D_last, (2, 1, 0))  #(4, 256, 256)
            if (j==0) : # 初始化 frame_3D_new
                frame_3D_new = np.stack((np.zeros((256,256)),np.zeros((256,256)))).reshape(2,256,256) 
                frame_3D_new = np.vstack((frame_3D_new,frame_3D_last))
            else:
                frame_3D_new = np.vstack((frame_3D_new,frame_3D_last))
        image = frame_3D_new.astype(np.float32)
        image = image[2:,:,:] #[82,256,256]
        
        image = np.expand_dims(image, axis=0) # Jack处理图像时即扩展了Z轴
        
        # print()
        # ========================== Step2: 增强后的数据进行拆解
        for img_count in range(0,8):
            image_slices = image[:,img_count*10:(img_count+1)*10,:,:]
            # ========================== Step3: h5f 文件创建
            # ========================== Step4: h5f 文件s写入
            data_h5f.create_dataset(str(h5f_label), data=torch.from_numpy(image_slices))
            label_h5f.create_dataset(str(h5f_label), data=torch.tensor(lbl, dtype=torch.long))
            # 给名字编码
            name_code = get_name_code(name)
            name_code = atoi(name_code)
            name_h5f.create_dataset(str(h5f_label),data = torch.tensor(name_code, dtype=torch.long))
            h5f_label += 1

        # data_h5f.create_dataset(str(h5f_label), data=torch.from_numpy(image))
        # label_h5f.create_dataset(str(h5f_label), data=torch.tensor(lbl, dtype=torch.long))
        # name_code = get_name_code(name)
        # name_code = atoi(name_code)
        # name_h5f.create_dataset(str(h5f_label),data = torch.tensor(name_code, dtype=torch.long))
        # h5f_label += 1
        print(f'write successful : {i}/{len(data_path)}', end='\r')