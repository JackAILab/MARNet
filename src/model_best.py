from doctest import IGNORE_EXCEPTION_DETAIL
from functools import partial
import math
from turtle import forward
from typing_extensions import Self

from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from metrics import f1_score

def get_inplanes():
    return [64, 128, 256, 512] 






def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


#SEBlock
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x



class ChannelAttention(nn.Module):
    def __init__(self,inplanes,ratio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(inplanes,inplanes//16,1,bias=False) 
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv3d(inplanes//16,inplanes,1,bias=False)

        self.sigmoid=nn.Sigmoid()

    def forward(self,x):

        avg_out = self.avg_pool(x)

        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out) 
        avg_out = self.fc2(avg_out)
        

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class PReNet_LSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv3d(128, 32, 3, 1, 1),  
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input

        h = Variable(torch.zeros(batch_size, 32, row, col, col))
        c = Variable(torch.zeros(batch_size, 32, row, col, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            x = x.cuda() 

        x_list = []
 
        x1 = x
        x = torch.cat((input, x), 1) 
        x = self.conv0(x)
        x = torch.cat((x, h), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        x = h
        resx = x
        x = F.relu(self.res_conv1(x) + resx)
        x = self.conv(x) 
        return x 
    
    
"""这里添加了LSTM"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        # 注意力机制


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3, 
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool 


        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7), 
                               stride=(conv1_t_stride, 2, 2), 
                               padding=(conv1_t_size // 2, 3, 3), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes) 
        self.relu = nn.ReLU(inplace=True)


        self.ca=ChannelAttention(self.in_planes)
        self.sa = SpatialAttention()

        self.lstm=PReNet_LSTM(self.in_planes)


        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.ca1=ChannelAttention(self.in_planes)
        self.sa1 = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) 
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.pro = nn.Softmax(dim = 1) 
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print()
    

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 

        x = self.conv1(x) 

        x = self.bn1(x)
        x = self.relu(x)
        self.x_origin = x 

        x = self.relu(self.lstm(x)+x) 
        self.x_lstm = x 

        x = self.ca(x) * x
        self.x_ca = x 
        x = self.sa(x) * x
        self.x_sa = x 
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 

        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.pro(x)
        return x 


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == '__main__':
    model = generate_model(model_depth=10, n_input_channels=1, n_classes=2)
    x = torch.randn(4, 1, 64, 224, 224)
    res = model(x)
    print(res.shape)