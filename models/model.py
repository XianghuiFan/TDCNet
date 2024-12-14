import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
import math
from .resnet import resnet18
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class MFFM(nn.Module):
    def __init__(self, dim, reduction=8):
        super(MFFM, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    
class UpSampleBN(nn.Module):
    def __init__(self, input_features, output_features, res = True):
        super(UpSampleBN, self).__init__()
        self.res = res

        self._net = nn.Sequential(nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(input_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(input_features),
                                  nn.LeakyReLU())

        self.up_net = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, kernel_size = 2, stride = 2, padding = 0, output_padding = 0),
                                    nn.BatchNorm2d(output_features, output_features),
                                    nn.ReLU(True))



    def forward(self, x, concat_with):
        if concat_with == None:
            if self.res:
                conv_x = self._net(x) + x
            else:
                conv_x = self._net(x)
        else:
            if self.res:
                conv_x = self._net(torch.cat([x, concat_with], dim=1)) + torch.cat([x, concat_with], dim=1)
            else:
                conv_x = self._net(torch.cat([x, concat_with], dim=1)) 

        return self.up_net(conv_x)


class MFFMfusion(nn.Module):
    def __init__(self, C):
        super(MFFMfusion, self).__init__()
        self.fusionmoudle = MFFM(C)

    def forward(self, in_data, x):

        b, c, h, w = x.size()
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((h // 2, w // 2))
        x  = x.repeat(1,2,1,1)
        x = self.avg_pool_2d(x)


        return self.fusionmoudle(x,in_data)



class DecoderBN(nn.Module):
    def __init__(self, num_features=128, lambda_val=1, res=True):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.lambda_val = lambda_val
        self.MFFM1 = MFFMfusion(48)
        self.MFFM2 = MFFMfusion(96)
        self.MFFM3 = MFFMfusion(192)
        self.up1 = UpSampleBN(192, features, res)
        self.up2 = UpSampleBN(features + 96, features, res)
        self.up3 = UpSampleBN(features + 48, features, res)
        self.up4 = UpSampleBN(features + 24, features//2, res)


    def forward(self, features):
        x_block4, x_block3, x_block2, x_block1= features[3], features[2], features[1], features[0]


        x_block2_1 = self.MFFM1(x_block2, x_block1)
        x_block3_1 = self.MFFM2(x_block3, x_block2_1)
        x_block4_1 = self.MFFM3(x_block4, x_block3_1)
        x_d0 = self.up1(x_block4_1, None)
        x_d1 = self.up2(x_d0, x_block3_1)
        x_d2 = self.up3(x_d1, x_block2_1)
        x_d3 = self.up4(x_d2, x_block1)


        return x_d3


class TDCNet(nn.Module):
    def __init__(self, lambda_val = 1, res = True):
        super(TDCNet, self).__init__()



        self.resnet = resnet18(pretrained=False)

        self.encoder = SwinTransformer(patch_size=2, in_chans= 4, embed_dim=24)
        self.decoder = DecoderBN(num_features=128, lambda_val=lambda_val, res=res)


        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, img, depth, **kwargs):
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)

        depth_cnn_feature_list = self.resnet(depth)
        encoder_x = self.encoder(torch.cat((img, depth), dim=1))

        encoder_x[0] += depth_cnn_feature_list[0]
        encoder_x[1] += depth_cnn_feature_list[1]
        encoder_x[2] += depth_cnn_feature_list[2]
        encoder_x[3] += depth_cnn_feature_list[3]

        decoder_x = self.decoder(encoder_x)

        out = self.final(decoder_x)


        return out


    @classmethod
    def build(cls, **kwargs):
 
        print('Building Encoder-Decoder model..', end='')
        m = cls(**kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    
    model = TDCNet.build(100)
    x = torch.rand(2, 3, 24, 32)
    y = torch.rand(2, 1, 24, 23)
    bins, pred = model(x,y)
    print(bins.shape, pred.shape)
