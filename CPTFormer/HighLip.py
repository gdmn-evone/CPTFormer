import os 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from CPTFormer.clip import clip

class CLIPModel(nn.Module):
    def name(self):
        return 'CPT'
    def __init__(self, name, args=None):
        super(CLIPModel, self).__init__()
        self.total_steps = 0
        self.args = args
        self.clip_model = clip.load(name, device="cpu", args=args)[0] 
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.image_encoder = self.clip_model.visual
        self.dtype = self.clip_model.dtype
        self.save_dir = os.path.join(self.args['checkpoints_dir'], self.args['name'])
        self.num_classes = args["num_classes"]
        self.head = nn.Linear(768, self.num_classes) 
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.freeze_stages()

    def freeze_stages(self):
        total_para_nums = 0
        train_para_nums = 0
        for name, param in self.named_parameters():
            total_para_nums += param.numel()
            if 'adapter' in name or 'hfe' in name or 'head' in name or 'level_embed' in name or 'interactions' in name or 'ln_post' in name or 'visual.proj' in name:
                param.requires_grad = True
                train_para_nums += param.numel()
            else:
                param.requires_grad = False
        print(f'Total parameters: {total_para_nums}, Trainable parameters: {train_para_nums}')

    def forward(self, image, phase):
        image_features = self.image_encoder(image.type(self.dtype), phase.type(self.dtype))   
        out = self.head(image_features)
        return out, image_features
    
    def save_networks(self, name, epoch, optimizer):
        save_filename = 'model_epoch_%s.pth' % name
        save_path = os.path.join(self.save_dir, save_filename)

        state_dict = {
            'epoch':epoch, 
            'model': self.state_dict(),
            'total_steps' : self.total_steps,
            'optimizer': optimizer.state_dict()
        }

        torch.save(state_dict, save_path)
        
    def adjust_learning_rate(self, optimizer ,min_lr=1e-6):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True
    
