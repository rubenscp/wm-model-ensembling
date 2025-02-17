import torchvision
import torch.nn as nn
import torch
from thop import profile

from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead
)

# Importing python modules
from common.manage_log import *

def create_model(num_classes=91, size=300, nms=0.45):
    model_backbone = torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.DEFAULT
    )
    conv1 = model_backbone.conv1
    bn1 = model_backbone.bn1
    relu = model_backbone.relu
    max_pool = model_backbone.maxpool
    layer1 = model_backbone.layer1
    layer2 = model_backbone.layer2
    layer3 = model_backbone.layer3
    layer4 = model_backbone.layer4
    backbone = nn.Sequential(
        conv1, bn1, relu, max_pool, 
        layer1, layer2, layer3, layer4
    )
    out_channels = [512, 512, 512, 512, 512, 512]
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    )
    num_anchors = anchor_generator.num_anchors_per_location()
    head = SSDHead(out_channels, num_anchors, num_classes)
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(size, size),
        head=head,
        nms_thresh=nms
    )
    return model

def create_model_pytorchvision(num_classes_aux, size=300, nms=0.45, pretrained=True):
    # print(f'num_classes_aux: {num_classes_aux}')
    if size == 300:
        model = torchvision.models.detection.ssd300_vgg16(num_classes_aux, pretrained=True)
    else: 
        model = None
    return model    

def count_parameters(model):
    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    return number_of_parameters

def count_layers(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        return 1
    return 0

def compute_num_layers(model):
    num_layers = sum(count_layers(layer) for layer in model.modules())
    return num_layers

def compute_flops(model, input_size):
    # input = torch.randn(input_size).unsqueeze(0).cuda()
    # flops, _ = torch.autograd.profiler.profile(model, inputs=(input, ), verbose=True)

    model.eval()
    input = torch.randn(1, 3, 300, 300).cuda()  # Change size according to your use case
    flops, params = profile(model, inputs=(input, ))
    return flops, params

    
# if __name__ == '__main__':
#     model = create_model(2, 300)
#     print(model)
#     # Total parameters and trainable parameters.
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"{total_params:,} total parameters.")
#     total_trainable_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{total_trainable_params:,} training parameters.")

