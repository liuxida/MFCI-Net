import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from EmotionNet import EmoClassifier
import numpy as np
import cv2
from torchvision import transforms
from torch.nn import init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(cv2.getBuildInformation())




mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
inv_normalize=transforms.Normalize(
    mean=[-m/s for m,s,in zip(mean,std)],
    std=[1/s for s in std]
)




class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # nn.ReLU(inplace=True),
            nn.SELU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class DAM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(DAM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            # nn.ReLU()
            nn.SELU()

        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            # nn.ReLU()
            nn.SELU()
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        self.CBAM = ECAAttention()

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x1 = self.CBAM(x)
        y1 = self.CBAM(y)
        x = sim_map * x1 + sim_map * y1 + x

        return x


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()
        self.relu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class up_conv_bn_relu(nn.Module):
    def __init__(self, up_size, in_channels, out_channels, kernal_size=1, padding=0, stride=1):
        super(up_conv_bn_relu, self).__init__()
        self.upSample = nn.Upsample(size=(up_size, up_size), mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        # self.act = nn.ReLU()
        self.act = nn.SELU()

    def forward(self, x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

from swin_transformer import SwinTransformer
swin_transformers = SwinTransformer(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4,8,16,32),features_only=True).to(device)


class MFCINet(nn.Module):
    def __init__(self):
        super(MFCINet, self).__init__()
        self.emotion_model = HED().to(device)
        for p in self.parameters():
            p.requires_grad = False
        self.model_x = torchvision.models.resnet50(pretrained=True).to(device)
        self.feature1_x = nn.Sequential(*list(self.model_x.children())[:5])
        self.cbam1 = ECAAttention()
        self.feature2_x = list(self.model_x.children())[5]
        # self.feature3_x = list(self.model_x.children())[6]
        # self.feature4_x = list(self.model_x.children())[7]

        # # 使用卷积层将输入 256x256x3 的图像转为 64x64x256
        # self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=4, padding=3)  # 下采样至 64x64
        # self.bn1 = nn.BatchNorm2d(128)
        # self.relu = nn.ReLU()

        # Swin Transformer 预训练模型
        self.swin_transformer = swin_transformers.to(device)

        self.up1_1 = up_conv_bn_relu(up_size=64, in_channels=256, out_channels=256)
        self.up2_1 = up_conv_bn_relu(up_size=32, in_channels=512, out_channels=512)
        self.up3_1 = up_conv_bn_relu(up_size=16, in_channels=1024, out_channels=1024)
        self.up4_1 = up_conv_bn_relu(up_size=8, in_channels=1024, out_channels=2048)

        # 适配输入通道为256
        # self.conv = nn.Conv2d(3, 768, kernel_size=1)  # 调整通道数适应 Transformer 输入

        # 用于调整输出的尺寸
        # self.upsample_32x32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.upsample_16x16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # self.upsample_8x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        # # 自定义卷积层以适配 Swin Transformer 的特征图大小
        # self.conv2 = nn.Conv2d(128, 512, kernel_size=1)  # 将 64x64x256 转为 32x32x512
        # self.conv3 = nn.Conv2d(512, 1024, kernel_size=1)  # 将 32x32x512 转为 16x16x1024
        # self.conv4 = nn.Conv2d(1024, 2048, kernel_size=1)  # 将 16x16x1024 转为 8x8x2048

        # self.model_s = torchvision.models.resnet50(pretrained=True)
        # self.feature1_s = nn.Sequential(*list(self.model_s.children())[:5])
        # self.feature2_s = list(self.model_s.children())[5]
        # self.feature3_s = list(self.model_s.children())[6]
        # self.feature4_s = list(self.model_s.children())[7]

        self.up1 = up_conv_bn_relu(up_size=64, in_channels=2048, out_channels=256)
        self.CBR1 = conv_bn_relu(512, 56)
        self.CBR2 = conv_bn_relu(512, 56)
        self.CBR3 = conv_bn_relu(1024, 56)
        self.CBR4 = conv_bn_relu(2048, 56)
        self.CBR5 = conv_bn_relu(56, 256)
        self.CBR6 = conv_bn_relu(512, 256)
        self.CBR7 = conv_bn_relu(512, 256)

        self.SADEM1 = DAM(56, 16)
        self.SADEM2 = DAM(56, 16)
        # self.SADEM1 = ImprovedFusionModule(56,56)
        # self.SADEM2 = ImprovedFusionModule(56,56)
        self.CBAM = ECAAttention()

        self.head = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(128, 1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        input_size = x.size()
        s = F.interpolate(x, size=[(input_size[2] // 2), (input_size[3] // 2)], mode="bilinear", align_corners=True)
        EAM = self.emotion_model(x).to(device)
        # 使用 interpolate 进行下采样
        EAM = F.interpolate(EAM, size=(64, 64), mode='bilinear', align_corners=True)
        EAM = EAM.to(device)
        x1 = self.feature1_x(x)  # 256, 128,128
        x2 = self.feature2_x(x1)  # 512,64,64
        # x3 = self.feature3_x(x2)# 1024,32,32
        # x4 = self.feature4_x(x3)

        # s = self.conv1(s)
        # s = self.bn1(s)
        # s = self.relu(s)
        # s = self.conv(s)
        # print(s.shape)
        batch_size,channels,height,width=s.shape
        s=F.interpolate(s, size=(224, 224), mode='bilinear', align_corners=False)

        # s = s.view(batch_size,channels,height*width).permute(0,2,1)
        s=self.swin_transformer.forward_features(s)


        # s=s.view(-1,7,7,1024)
        # s=s.permute(0,3,1,2)



        s1=s[0]
        s2=s[1]
        s3=s[2]
        s4=s[3]

        s1=s1.view(-1,28,28,256)
        s2=s2.view(-1,14,14,512)
        s3=s3.view(-1,7,7,1024)
        s4=s4.view(-1,7,7,1024)
        s1 = s1.permute(0, 3, 1, 2)
        s2 = s2.permute(0, 3, 1, 2)
        s3 = s3.permute(0, 3, 1, 2)
        s4 = s4.permute(0, 3, 1, 2)

        #
        # s1=s1.permute(0,3,2,1)
        # s2=s2.permute(0,3,2,1)
        # s3=s3.permute(0,3,2,1)
        # s4=s4.permute(0,3,2,1)

        s1=self.up1_1(s1)
        s2=self.up2_1(s2)
        s3=self.up3_1(s3)
        s4=self.up4_1(s4)

        # s4=self.upsample_8x8(s)
        # s1=self.swin_transformer.layers_0(s)
        #
        # s2=self.conv2(s1)
        # s2=self.swin_transformer.stage[1](s2)
        #
        # s3=self.conv3(s2)
        # s3=self.swin_transformer.stage[2](s3)
        #
        # s4=self.conv4(s3)
        # s4=self.swin_transformer.stage[3](s4)
        # s1 = self.feature1_s(s)  # 256, 64,64
        # s2 = self.feature2_s(s1)  # 512,32,32
        # s3 = self.feature3_s(s2)  # 1024,16,16
        # s4 = self.feature4_s(s3)

        s2 = self.CBR1(s2)
        x2 = self.CBR2(x2)
        C = self.SADEM1(x2, s2)
        s3 = self.CBR3(s3)
        C = self.SADEM2(C, s3)
        C = self.CBR5(C)

        x4_ = self.up1(s4)
        cat = torch.cat((C, x4_), dim=1)
        cat = self.CBR6(cat)

        h_EAM = cat * EAM
        h_EAM = self.CBAM(h_EAM)
        # h_EAM = self.gate(h_EAM)
        Fusion_F = torch.cat((h_EAM, cat), dim=1)
        Fusion_F = self.CBR7(Fusion_F)
        score_feature = self.avgpool(Fusion_F).view(Fusion_F.size(0), -1)
        score = self.head(score_feature)
        return score
