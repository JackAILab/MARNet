from datas import*  #sss

import torch
import h5py
from curses.ascii import isdigit
from locale import atoi


# =============================== train_data_h5f_path =============================== #
train_save_data_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_Data.he5' # h5f data path
train_save_label_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_Data_label.he5' # h5f label path
train_save_patient_name_path = '/data3/hjh/PDRevise/02-new-h5f-DE/train_name_label.he5' # h5f name path

test_save_data_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_Data.he5' 
test_save_label_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_Data_label.he5'
test_save_patient_name_path = '/data3/hjh/PDRevise/02-new-h5f-DE/test_name_label.he5'

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

def preparePDData(data_path, patch_size, stride, data_type): 
    
    h5f_label = 0 
    
    if data_type == 'train':
        data_h5f = h5py.File(train_save_data_path, 'w') 
        label_h5f = h5py.File(train_save_label_path,'w') 
        name_h5f = h5py.File(train_save_patient_name_path,'w') 
    elif data_type == 'test':
        data_h5f = h5py.File(test_save_data_path, 'w') 
        label_h5f = h5py.File(test_save_label_path,'w') 
        name_h5f = h5py.File(test_save_patient_name_path,'w') 
    
    for i in range(len(data_path)): 

        image = np.load(data_path[i], allow_pickle=True)
        lbl = int(data_path[i][30:31])
        if lbl != 0:
            lbl = 1
        name = data_path[i][42:-4]
        image = np.transpose(image, (2, 1, 0)) 
        image = image.astype(np.float32)
        for j in range(20):
            im=image[:,:,j]
            frame_3D_last = Im2Patch(im,win=patch_size, stride=stride)
            frame_3D_last = frame_3D_last[0,:,:,:]
            frame_3D_last = np.transpose(frame_3D_last, (2, 1, 0))
            if (j==0) : 
                frame_3D_new = np.stack((np.zeros((256,256)),np.zeros((256,256)))).reshape(2,256,256) 
                frame_3D_new = np.vstack((frame_3D_new,frame_3D_last))
            else:
                frame_3D_new = np.vstack((frame_3D_new,frame_3D_last))
        image = frame_3D_new.astype(np.float32)
        image = image[2:,:,:] #[82,256,256]
        
        image = np.expand_dims(image, axis=0) 
        
        for img_count in range(0,8):
            image_slices = image[:,img_count*10:(img_count+1)*10,:,:]

            data_h5f.create_dataset(str(h5f_label), data=torch.from_numpy(image_slices))
            label_h5f.create_dataset(str(h5f_label), data=torch.tensor(lbl, dtype=torch.long))

            name_code = get_name_code(name)
            name_code = atoi(name_code)
            name_h5f.create_dataset(str(h5f_label),data = torch.tensor(name_code, dtype=torch.long))
            h5f_label += 1

        print(f'write successful : {i}/{len(data_path)}', end='\r')