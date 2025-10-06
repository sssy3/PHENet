# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

current_file_path = os.path.abspath("/data/coding/HazyCDNet/modeling/Hazy_LightCDNet_baseline_FF21_plus_HAMF3_separate525_SR.py")
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from heightmodel import SwinTUNet
import cv2
import numpy as np
from torch_dct import dct_2d, idct_2d  
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class SRBranch(nn.Module):
    def __init__(self, in_channels_shallow=24, in_channels_deep=320):
        super().__init__()
        self.deep_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels_deep, in_channels_shallow, kernel_size=1)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels_shallow, in_channels_shallow, kernel_size=1)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_shallow*2, in_channels_shallow, kernel_size=3, padding=1),
            SEBlock(in_channels_shallow)
        )
        self.t_pred = nn.Sequential(
            nn.Conv2d(in_channels_shallow, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        self.A_pred = nn.AdaptiveAvgPool2d(1)
        self.to_rgb = nn.Conv2d(in_channels_shallow, 3, kernel_size=1)
    def forward(self, shallow_feat, deep_feat):
        deep_up = self.deep_up(deep_feat)    
        fused = self.fusion(torch.cat([shallow_feat, deep_up], dim=1))
        fused = self.up(fused)
        t = self.t_pred(fused)                 
        A = self.A_pred(fused).squeeze(-1).squeeze(-1)  
        A = A[:, :3]                         
        I_HR_hazy = self.to_rgb(fused) * t + A.view(-1,3,1,1) * (1 - t)
        return I_HR_hazy, t, A
class Height_feature_for_fusion(nn.Module):
    def __init__(self, feat_shapes):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._build_level_block(feat_shapes[i]) 
            for i in range(3)
        ])  
    def _build_level_block(self, shape):
        c, h, w = shape
        return nn.ModuleDict({
            'height_encoder': nn.Sequential(
                nn.Conv2d(1, c//4, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c//4, c, 3, padding=1), 
                nn.AdaptiveAvgPool2d((h, w))      
            )
        })    
    def forward(self, feats, ndsm):
        outputs = []
        for i, (feat, block) in enumerate(zip(feats, self.blocks)):
            h_enc = block['height_encoder'](ndsm)
            outputs.append(h_enc) 
        return outputs[0], outputs[1], outputs[2]

class FrequencyDecomposition(nn.Module):
    def __init__(self, bands=2):
        super().__init__()
        self.bands = bands
        self.filters = nn.Parameter(torch.randn(bands, 1, 7, 7)) 
        
    def forward(self, x):
        B, C, H, W = x.size()
        coeff = dct_2d(x)
        bands = []
        for i in range(self.bands):
            expanded_filter = self.filters[i].repeat(C, 1, 1, 1)
            band = F.conv2d(
                coeff,
                expanded_filter,
                stride=1,
                padding=3,  
                groups=C
            )
            bands.append(idct_2d(band))
        return bands

class CrossFrequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, feat, height_feat):
        B, C, H, W = feat.size()
        Q = self.query(feat).view(B, -1, H*W).permute(0,2,1)
        K = self.key(height_feat).view(B, -1, H*W)
        V = self.value(height_feat).view(B, -1, H*W)
        energy = torch.bmm(Q, K)
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(V, attention.permute(0,2,1)).view(B, C, H, W)
        return self.gamma * out + feat

class HAMF(nn.Module):
    def __init__(self, feat_shapes):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._build_level_block(feat_shapes[i]) 
            for i in range(3)
        ])
        
    def _build_level_block(self, shape):
        c, h, w = shape
        return nn.ModuleDict({
            'freq_decomp': FrequencyDecomposition(bands=2),
            'height_encoder': nn.Sequential(
                nn.Conv2d(1, c//4, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c//4, c, 3, padding=1),  
                nn.AdaptiveAvgPool2d((h, w))       
            ),
            'low_attn': CrossFrequencyAttention(c),
            'high_attn': CrossFrequencyAttention(c),
            'fusion': nn.Sequential(
                nn.Conv2d(c*2, c, 1),
                nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1),
                nn.Sigmoid()
            )
        })
    
    def forward(self, feats, ndsm):
        outputs = []
        for i, (feat, block) in enumerate(zip(feats, self.blocks)):
            h_enc = block['height_encoder'](ndsm)
            feat_low, feat_high = block['freq_decomp'](feat)
            h_low, h_high = block['freq_decomp'](h_enc)
            attn_low = block['low_attn'](feat_low, h_low)
            attn_high = block['high_attn'](feat_high, h_high)
            fused = block['fusion'](torch.cat([attn_low, attn_high], dim=1))
            outputs.append(feat + fused)       
        return outputs[0], outputs[1], outputs[2]

class DifferenceFeatureExtractor(nn.Module):
    def __init__(self, in_channels, pool_size=3, stride=1, padding=1, mlp_hidden_dim=64):
        super().__init__()
        self.img_avgpool = nn.AvgPool2d(pool_size, stride=stride, padding=padding)
        self.img_maxpool = nn.MaxPool2d(pool_size, stride=stride, padding=padding)
        self.height_avgpool = nn.AvgPool2d(pool_size, stride=stride, padding=padding)
        self.height_maxpool = nn.MaxPool2d(pool_size, stride=stride, padding=padding)
        self.img_mlp = nn.Sequential(
            nn.Conv2d(in_channels*3, mlp_hidden_dim, 1),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(mlp_hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.height_mlp = nn.Sequential(
            nn.Conv2d(in_channels*3, mlp_hidden_dim, 1),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(mlp_hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels*3, mlp_hidden_dim, 1),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(mlp_hidden_dim, in_channels*2, 1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU()
        )

    def forward(self, img1, img2, height_diff):
        img_diff = torch.abs(img1 - img2)
        img_avg = self.img_avgpool(img_diff)  
        img_max = self.img_maxpool(img_diff)  
        img_cat = torch.cat([img_diff, img_avg, img_max], dim=1)
        height_avg = self.height_avgpool(height_diff)
        height_max = self.height_maxpool(height_diff)
        height_cat = torch.cat([height_diff, height_avg, height_max], dim=1)
        img_feat = self.img_mlp(img_cat)
        height_feat = self.height_mlp(height_cat)
        w = height_feat+img_feat
        feat1 = img1 * (1+w)
        feat2 = img2 * (1+w)
        return self.fusion_conv(torch.cat([feat1, feat2, feat1-feat2], dim=1))

class FusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )
        
    def forward(self, high, low):
        high_up = F.interpolate(high, scale_factor=2, mode='bilinear')
        fused = torch.cat([high_up, low], dim=1)
        return self.conv(fused)

class PHENet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(PHENet, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d
        self.backbone = build_backbone('mobilenet_3f', output_stride, BatchNorm)
        self.up1 = nn.ConvTranspose2d(100, 128, kernel_size=2, stride=2)
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(128), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(64), nn.ReLU())
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        self.stage2 = DifferenceFeatureExtractor(24)
        self.stage3 = DifferenceFeatureExtractor(32)
        self.stage4 = DifferenceFeatureExtractor(320)
        self.conv1 = nn.Conv2d(24*2, 24, 1)
        self.conv2 = nn.Conv2d(32*2, 32, 1)
        self.conv3 = nn.Conv2d(320*2, 320, 1)
        self.fuse4 = FusionBlock(32+320)
        self.fuse3 = FusionBlock(24+176)
        self.hamf = HAMF([(24,64,64), (32,32,32), (320,16,16)])
        self.h_feat = Height_feature_for_fusion([(24,64,64), (32,32,32), (320,16,16)])
        self.sr_branch = SRBranch()

    def forward(self, x1, x2,h1,h2):
        FL_1,FM_1,FH_1 = self.backbone(x1)
        FL_2,FM_2,FH_2 = self.backbone(x2)
        h1 = (h1 - h1.min()) / (h1.max() - h1.min() + 1e-6)
        h2 = (h2 - h2.min()) / (h2.max() - h2.min() + 1e-6)
        if self.training:
            I_HR_hazy1, t1, A1 = self.sr_branch(FL_1, FH_1)
            I_HR_hazy2, t2, A2 = self.sr_branch(FL_2, FH_2)
            J1 = (I_HR_hazy1 - A1.view(-1,3,1,1)*(1-t1)) / (t1 + 1e-6)
            J2 = (I_HR_hazy2 - A2.view(-1,3,1,1)*(1-t2)) / (t2 + 1e-6)
        else:
            I_HR_hazy1 = I_HR_hazy2 = J1 = J2 = t1 = t2 = A1 = A2 =None  
        FL_1,FM_1,FH_1= self.hamf([FL_1,FM_1,FH_1], h1)
        FL_2,FM_2,FH_2= self.hamf([FL_2,FM_2,FH_2], h2)
        HL_1,HM_1,HH_1 = self.h_feat([FL_1,FM_1,FH_1], h1)
        HL_2,HM_2,HH_2 = self.h_feat([FL_2,FM_2,FH_2], h2)
        L_Hdiff = torch.abs(HL_1-HL_2)
        M_Hdiff = torch.abs(HM_1-HM_2)
        H_Hdiff = torch.abs(HH_1-HH_2)
        attcf2 = self.conv1(self.stage2(FL_1,FL_2,L_Hdiff))
        attcf3 = self.conv2(self.stage3(FM_1,FM_2,M_Hdiff))
        attcf4 = self.conv3(self.stage4(FH_1,FH_2,H_Hdiff))
        attcf34 = self.fuse4(attcf4, attcf3)
        attcf234 = self.fuse3(attcf34, attcf2) 
        FD1 = attcf234
        up1 = self.up1(FD1)
        block1 = self.block1(up1)
        up2 = self.up2(block1)
        block2 = self.block2(up2)
        return self.conv(block2), I_HR_hazy1, J1, t1, A1, I_HR_hazy2, J2, t2, A2
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.sr_branch,self.h_feat,self.hamf,self.stage2,self.stage3,self.stage4,self.conv1,self.conv2,self.conv3,self.fuse3,self.fuse4, self.up1, self.block1, self.up2,
                   self.block2, self.conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
if __name__ == "__main__":
    model = PHENet(backbone='mobilenet', output_stride=16)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    image1 = torch.rand(1,3,256,256)
    image2 = image1
    height1 = torch.rand(1,1,256,256)
    height2 = height1
    output = model(image1, image2, height1, height2)
    print(output)