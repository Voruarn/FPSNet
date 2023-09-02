import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .init_weights import init_weights
from .ResNet import resnet50, resnet101, resnet152
from .pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large
from .hrnetv2 import *

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

HRNet_out_ch = {
    'w32': [32, 64, 128, 256],   
    'w48': [48, 96, 192, 384],     # [layers, embed_dims, drop_path_rate]
}   


# backbone_name: pvt_small, resnet50, hrnetv2_32
class FPSNet(nn.Module):
    def __init__(self, n_channels=3, backbone_name='resnet50', mid_ch=64, is_deconv=True,
                is_batchnorm=True):
        super(FPSNet, self).__init__()      

        eout_channels=[256, 512, 1024, 2048]
        if backbone_name.find('pvt')!=-1:
            eout_channels=[64, 128, 320, 512]
        elif backbone_name.find('hrnet')!=-1:
            # hrnetv2_32, hrnetv2_48
            phi='w'+backbone_name.split('_')[-1]
            eout_channels=HRNet_out_ch[phi]

        out_ch=1
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        # Encoder
        self.backbone  = eval(backbone_name)(pretrained=False)
        self.eside1=ConvModule(eout_channels[0], mid_ch)
        self.eside2=ConvModule(eout_channels[1], mid_ch)
        self.eside3=ConvModule(eout_channels[2], mid_ch)
        self.eside4=ConvModule(eout_channels[3], mid_ch)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = Conv2Module(eout_channels[3], mid_ch, self.is_batchnorm)

        # AFFM
        self.AFF1=AFFM1(mid_ch)
        self.AFF2=AFFM(mid_ch)
        self.AFF3=AFFM(mid_ch)
        self.AFF4=AFFM4(mid_ch)

        # Decoder
        self.decoder4 = DAM(mid_ch, mid_ch, self.is_deconv)
        self.decoder3 = DAM(mid_ch, mid_ch, self.is_deconv)
        self.decoder2 = DAM(mid_ch, mid_ch, self.is_deconv)
        self.decoder1 = DAM(mid_ch, mid_ch, self.is_deconv)

        self.dside1 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,out_ch,3,padding=1)

        self.fuse=nn.Conv2d(out_ch*4, out_ch, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        outs = self.backbone(inputs)
        c1, c2, c3, c4 = outs

        maxpool4 = self.maxpool4(c4)
        center = self.center(maxpool4)  

        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)

        ca1=self.AFF1(c1,c2)
        ca2=self.AFF2(c1,c2,c3)
        ca3=self.AFF3(c2,c3,c4)
        ca4=self.AFF4(c3,c4)

        # decoder
        up4 = self.decoder4(center, ca4)  
        up3 = self.decoder3(up4, ca3) 
        up2 = self.decoder2(up3, ca2)  
        up1 = self.decoder1(up2, ca1) 

        # side pred
        d1=self.dside1(up1)
       
        d2=self.dside2(up2)
        d2 = _upsample_like(d2, d1)
        
        d3=self.dside3(up3)
        d3 = _upsample_like(d3, d1)
        
        d4=self.dside4(up4)
        d4 = _upsample_like(d4, d1)
        
        dfuse=torch.cat([d1,d2,d3,d4], dim=1)
        dfuse=self.fuse(dfuse)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)
        Sfuse= F.interpolate(dfuse, size=(H, W), mode='bilinear', align_corners=True)

        return F.sigmoid(Sfuse), F.sigmoid(S1), F.sigmoid(S2), F.sigmoid(S3), F.sigmoid(S4)
