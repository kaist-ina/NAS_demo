import torch.nn as nn
import random, sys
from model import common

#Choice 1: deeper
def get_config(quality):
    if quality == 'low': #Size: xxMB / Device: 1050ti
        return {4: {'layer' : 18, 'feature' : 9},
                3: {'layer' : 18, 'feature' : 8},
                2: {'layer' : 18, 'feature' : 4},
                1: {'layer' : 4, 'feature' : 2}}
    elif quality == 'medium': #Size
        return {4: {'layer' : 18, 'feature' : 21},
                3: {'layer' : 18, 'feature' : 18},
                2: {'layer' : 18, 'feature' : 9},
                1: {'layer' : 4, 'feature' : 7}}
    elif quality == 'high': #Size: 1.258MB / #Device: 1070
        return {4: {'layer' : 18, 'feature' : 32},
                3: {'layer' : 18, 'feature' : 29},
                2: {'layer' : 18, 'feature' : 18},
                1: {'layer' : 4, 'feature' : 16}}
    elif quality == 'ultra': #Size: xxMB / Device: 1080ti #Check
        return {4: {'layer' : 18, 'feature' : 48},
                3: {'layer' : 18, 'feature' : 42},
                2: {'layer' : 18, 'feature' : 26},
                1: {'layer' : 4, 'feature' : 26}}
    else:
        print('unsupported quality')
        sys.exit()

#Note: It can be easily implemented as using DNN specification as in a maniest file.
#      Currently, it uses a default setting as described in the paper
#M(ulti-resolution)Nas_Net
class Multi_Network(nn.Module):
    def __init__(self, quality, act=nn.ReLU(True)):
        super(Multi_Network, self).__init__()

        #Determine whether to apply biinear upscale in the front (for test infernce)
        self.apply_upscale = True

        #Set model architecture
        config = get_config(quality)
        self.networks= nn.ModuleList()
        self.networks.append(Single_Network(nLayer=config[4]['layer'], nFeat=config[4]['feature'], nChannel=3, scale=4, outputFilter=2, bias=True, act=act))
        self.networks.append(Single_Network(nLayer=config[3]['layer'], nFeat=config[3]['feature'], nChannel=3, scale=3, outputFilter=2, bias=True, act=act))
        self.networks.append(Single_Network(nLayer=config[2]['layer'], nFeat=config[2]['feature'], nChannel=3, scale=2, outputFilter=2, bias=True, act=act))
        self.networks.append(Single_Network(nLayer=config[1]['layer'], nFeat=config[1]['feature'], nChannel=3, scale=1, outputFilter=1, bias=True, act=act))

        self.scale_dict = {1:3, 2:2, 3:1, 4:0}
        self.target_scale = None

    #TODO: getOutputNode from edsr.py
    def getOutputNodes(self, target_scale):
        return self.networks[self.scale_dict[target_scale]].getOutputNodes()

    def setScale(self, scale):
        assert scale in self.scale_dict.keys()
        self.target_scale= scale

    def forward(self, x, idx=None):
        assert self.target_scale != None

        x = self.networks[self.scale_dict[self.target_scale]].forward(x, idx)

        return x

#S(ingle-resolution)NAS_Net
class Single_Network(nn.Module):
    def __init__(self, nLayer, nFeat, nChannel, scale, outputFilter, bias=True, act=nn.ReLU(True)):
        super(Single_Network, self).__init__()
        #model features
        self.nResblock = ((nLayer - 2) // 2)
        self.nFeat = nFeat
        self.nChannel = nChannel
        self.scale = scale #added - to be extended on multiple resolution
        #self.target_scale = None

        assert self.scale in [1,2,3,4]

        #Filter outputnodes for performance : current version use all intermediate Resblocks
        self.outputNode = []
        for i in range(self.nResblock // outputFilter + 1):
            self.outputNode.append(self.nResblock - outputFilter * i)
            assert self.nResblock - (outputFilter * i) >= 0
        if self.nResblock not in self.outputNode:
            self.outputNode.append(self.nResblock)
        self.outputNode = sorted(self.outputNode)
        self.outputList = common.random_gradual_03(self.outputNode)

        #Model head
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=self.nChannel, out_channels=self.nFeat, kernel_size=3, stride=1, padding=1, bias=bias)])

        #Model body (conv)
        self.body = nn.ModuleList()
        for _ in range(self.nResblock):
            modules_body = [common.ResBlock(self.nFeat, bias=bias, act=act)]
            self.body.append(nn.Sequential(*modules_body))

        body_end = []
        body_end.append(nn.Conv2d(in_channels=self.nFeat, out_channels=self.nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body_end = nn.Sequential(*body_end)

        #Model body (upscale)
        if self.scale > 1:
            self.upscale= nn.Sequential(*common.Upsampler(self.scale, self.nFeat, bias=bias))

        #tail
        self.tail = nn.Sequential(*[nn.Conv2d(in_channels=self.nFeat, out_channels=self.nChannel, kernel_size=3, stride=1, padding=1, bias=bias)])

    def getOutputNodes(self):
        return self.outputNode

    def forward(self, x, idx=None):
        #random idx choice for training
        if idx is None:
            idx = random.choice(self.outputList)
        else:
            assert idx <= self.nResblock and idx >= 0

        #feed-forward part
        #1.head
        x = self.head(x)
        res = x

        #2.body
        for i in range(idx):
            res = self.body[i](res)
        res = self.body_end(res)
        res += x

        #3.upscale
        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res

        #4.tail
        x = self.tail(x)

        return x
