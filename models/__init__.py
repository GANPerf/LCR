from .byol import BYOL
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet50_cub200,resnet50_stanfordcars, resnet18_cifar_variant2,resnet18_cifar_variant1,resnet50_aircrafts

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}(pretrained=True)")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    if model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    else:
        raise NotImplementedError
    return model