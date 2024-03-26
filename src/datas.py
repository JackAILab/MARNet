import cv2
import numpy as np

from PIL import Image
from PIL import ImageEnhance

def img_translation(self):
    M = np.float32([[1, 0, 0], [0, 1, 100]])
    image = cv2.warpAffine(self, M, (self.shape[1], self.shape[0]))
    M = np.float32([[1, 0, 0], [0, 1, -100]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image 
def img_scale(self):
    image = cv2.resize(self, (200, 100))
    return image 
    

def darker(self,percetage=0.9):
    self_copy = self.copy()
    w = self.shape[1]
    h = self.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            self_copy[xj,xi,0] = int(self[xj,xi,0]*percetage)
            self_copy[xj,xi,1] = int(self[xj,xi,1]*percetage)
            self_copy[xj,xi,2] = int(self[xj,xi,2]*percetage)
    return self_copy


def brighter(self, percetage=1.5):
    self_copy = self.copy()
    w = self.shape[1]
    h = self.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            self_copy[xj,xi,0] = np.clip(int(self[xj,xi,0]*percetage),0,a_max=255)
            self_copy[xj,xi,1] = np.clip(int(self[xj,xi,1]*percetage),0,a_max=255)
            self_copy[xj,xi,2] = np.clip(int(self[xj,xi,2]*percetage),0,a_max=255)
    return self_copy

def addGaussianNoise(image,percetage=0.3):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

def Im2Patch(img, win=256, stride=256):
    img = img.reshape(1,512,512)
    k = 0
    endc = img.shape[0] # endc 1
    endw = img.shape[1] # endw 512
    endh = img.shape[2] # endh 512
    patch = img[:, 0:endw - win + 1:stride, 0:endh - win  + 1:stride] # [:,0:512:100,0:512:100]
    TotalPatNum = patch.shape[1] * patch.shape[2] # 2*2
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32) # Y(1, 65536, 4)

    for i in range(win): #　win 100
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum]) # Y(3,10000,15) reshape  (1,256,256,4)
