import os
import torch
import torch.nn as nn
import torchvision.models as models

__all__ = ['get_resnet18']

def get_resnet18(pretrained=True, input_dim = 3, f_path='./encoding/models/pretrain/resnet18-5c106cde.pth'):
    assert input_dim in (1, 3, 4)
    model = models.resnet18(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')
        
        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            weights = torch.load(f_path)
            for k, v in weights.items():
                weights[k] = v.data
            conv1_ori = weights['conv1.weight']
            conv1_new = torch.zeros((64, input_dim, 7, 7), dtype=torch.float32)
            if input_dim == 4:
                conv1_new[:, :3, :, :] = conv1_ori
                conv1_new[:,  3, :, :] = conv1_ori[:,  1, :, :]
            else:
                conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :]
            weights['conv1.weight'] = conv1_new
            model.load_state_dict(weights, strict=False)
        else:
            model.load_state_dict(torch.load(f_path), strict=False)
    else:
        raise ValueError('Please use pretrained resnet18.')
    
    return model