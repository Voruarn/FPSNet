import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_weights import init_weights


class ChannelAttention(nn.Module):
    # Channel-attention module
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv2Module(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(Conv2Module, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size), nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class ConvUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(ConvUp, self).__init__()
        self.conv = Conv2Module(out_size * 2, out_size, False) # Conv+ReLU * 2
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('Conv2Module') != -1: continue
            init_weights(m, init_type='kaiming')


    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class SRM(nn.Module):
    # Self Refinement Module
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.in_ch=in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :self.in_ch, :, :], out2[:, self.in_ch:, :, :]

        return F.relu(w * out1 + b, inplace=True)
   

class DAM(nn.Module):
    # DAM: Dual-path Aggregation Module
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(DAM, self).__init__()
        self.conv = Conv2Module(out_size * 2, out_size, False) # Conv+ReLU * 2
        self.sr=SRM(out_size)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('Conv2Module') != -1: continue
            init_weights(m, init_type='kaiming')


    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.sr(self.conv(outputs0))


class AFFM(nn.Module):
    # AFFM: Adjacent Feature Fusion Module
    def __init__(self, cur_channel):
        super(AFFM, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        
        self.center_conv = ConvModule(cur_channel, cur_channel) 
        self.pre_conv = ConvModule(cur_channel, cur_channel)
        self.lat_conv = ConvModule(cur_channel, cur_channel)

        self.CBAM=CBAM(cur_channel)
        self.bn = nn.BatchNorm2d(cur_channel)
        self.relu = nn.ReLU(True)
        self.fuse = ConvModule(cur_channel, cur_channel)

    def forward(self, x_pre, x_cur, x_lat):
        cur_up=self.upsample2(x_cur)
        cur_ds=self.downsample2(x_cur)

        pre_ds=self.downsample2(x_pre)
        lat_up=self.upsample2(x_lat)
        
        center_c=self.center_conv(pre_ds+x_cur+lat_up)
        pre_c=self.pre_conv(x_pre+cur_up)
        lat_c = self.lat_conv(x_lat+cur_ds)

        x=self.relu(self.bn(center_c+self.downsample2(pre_c)+self.upsample2(lat_c)))
        x = x_cur +self.CBAM(x)
        x= self.fuse(x)
        return x


class AFFM1(nn.Module):
    def __init__(self, cur_channel):
        super(AFFM1, self).__init__()
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.center_conv = ConvModule(cur_channel, cur_channel)      
        self.lat_conv = ConvModule(cur_channel, cur_channel)

        self.CBAM=CBAM(cur_channel)
        self.bn = nn.BatchNorm2d(cur_channel)
        self.relu = nn.ReLU(True)
        self.fuse = ConvModule(cur_channel, cur_channel)

    def forward(self, x_cur, x_lat):
        cur_ds=self.downsample2(x_cur)
        lat_up=self.upsample2(x_lat)

        center_c=self.center_conv(x_cur+lat_up)
        lat_c = self.lat_conv(x_lat+cur_ds)

        x=self.relu(self.bn(center_c+self.upsample2(lat_c)))
        x = x_cur +self.CBAM(x)
        x= self.fuse(x)
        return x

class AFFM4(nn.Module):
    def __init__(self, cur_channel):
        super(AFFM4, self).__init__()
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.center_conv = ConvModule(cur_channel, cur_channel)
        self.pre_conv = ConvModule(cur_channel, cur_channel)

        self.CBAM=CBAM(cur_channel)
        self.bn = nn.BatchNorm2d(cur_channel)
        self.relu = nn.ReLU(True)
        self.fuse = ConvModule(cur_channel, cur_channel)

    def forward(self, x_pre, x_cur):
        cur_up=self.upsample2(x_cur)
        pre_ds=self.downsample2(x_pre)
       
        center_c=self.center_conv(pre_ds+x_cur)
        pre_c=self.pre_conv(x_pre+cur_up)
   
        x=self.relu(self.bn(center_c+self.downsample2(pre_c)))
        x = x_cur +self.CBAM(x)
        x= self.fuse(x)
        return x
    
