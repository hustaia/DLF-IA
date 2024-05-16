import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
import logging as logger

from models.swin_transformer import swin_b, DecoderHead
from models.net_utils import FeatureFusionModule as FFM


def load_model(model, model_file):
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('downsample') >= 0 and k.find('layer') >= 0:
            name = k.replace('downsample.', '')
            name = name.replace('layers', 'downsamples')
            state_dict[name] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    return model


class MultiEncoderDecoder(nn.Module):
    def __init__(self, args=None):
        super(MultiEncoderDecoder, self).__init__()
        if args.backbone == 'swin_b':
            self.backbone_rgb = swin_b()
            self.backbone_t = swin_b()
            self.channels = [256, 512, 1024]
            self.num_heads = [6, 12, 24]
        else:
            raise Exception("Not implement")
        
        self.decode_head_rgb = DecoderHead(in_channels=self.channels, num_classes=1, embed_dim=512)
        self.decode_head_t = DecoderHead(in_channels=self.channels, num_classes=1, embed_dim=512)
        
        self.FFMs = nn.ModuleList()
        for i in range(3):
            fuse = FFM(dim=self.channels[i], reduction=1, num_heads=self.num_heads[i])
            self.FFMs.append(fuse)
        self.decode_head_w_rgb = DecoderHead(in_channels=self.channels, num_classes=1, embed_dim=512)
        self.decode_head_w_t = DecoderHead(in_channels=self.channels, num_classes=1, embed_dim=512)
        self.k = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))

        self.init_weights(args, pretrained_rgb=args.pretrained_model, pretrained_t=args.pretrained_model)
        
    
    def init_weights(self, args, pretrained_rgb=None, pretrained_t=None):
        if pretrained_rgb:
            logger.info('Loading pretrained model: {}'.format(pretrained_rgb))
            self.backbone_rgb = load_model(self.backbone_rgb, pretrained_rgb)
        if pretrained_t:
            logger.info('Loading pretrained model: {}'.format(pretrained_t))
            self.backbone_t = load_model(self.backbone_t, pretrained_t)
        logger.info('Initing weights ...')
        for fuse in self.FFMs:
            init_weight(fuse, nn.init.kaiming_normal_,
                nn.BatchNorm2d, args.bn_eps, args.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.decode_head_rgb, nn.init.kaiming_normal_,
                nn.BatchNorm2d, args.bn_eps, args.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.decode_head_t, nn.init.kaiming_normal_,
                nn.BatchNorm2d, args.bn_eps, args.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.decode_head_w_rgb, nn.init.kaiming_normal_,
                nn.BatchNorm2d, args.bn_eps, args.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.decode_head_w_t, nn.init.kaiming_normal_,
                nn.BatchNorm2d, args.bn_eps, args.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if args.dataset in ['RGBTCC']:
            k0, b0 = 10, 0.44
        elif args.dataset in ['DroneRGBT']:
            k0, b0 = 10, 0.364
        else:
            raise Exception("Not implement")
        nn.init.constant_(self.k, k0)
        nn.init.constant_(self.b, b0)
    
    
    def forward(self, RGBT, value):
        rgb = RGBT[0]
        t = RGBT[1]
        features_rgb = self.backbone_rgb(rgb)[1:]
        features_t = self.backbone_t(t)[1:]
        
        new_features_rgb = []
        new_features_t = []
        for i in range(3):
            feature_rgb, feature_t = features_rgb[i], features_t[i]
            attention_rgb, attention_t = self.FFMs[i](feature_rgb, feature_t)
            feature_rgb = attention_rgb.mul(feature_rgb)
            feature_t = attention_t.mul(feature_t)
            new_features_rgb.append(feature_rgb)
            new_features_t.append(feature_t)
        
        output_rgb = self.decode_head_rgb(features_rgb)
        output_t = self.decode_head_t(features_t)
        w_rgb = self.decode_head_w_rgb(new_features_rgb)
        w_t = self.decode_head_w_t(new_features_t)
        
        b, c, h, w = output_rgb.shape
        w_illumination = 1 / (1 + torch.exp(-self.k * (value.mean() - self.b)))
        w_illumination = w_illumination.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(b, c, h, w)
        
        output_fuse = torch.multiply(w_illumination, torch.multiply(output_rgb, w_rgb)) + torch.multiply(output_t, w_t)
        return output_rgb, output_t, output_fuse, w_rgb, w_t, w_illumination

