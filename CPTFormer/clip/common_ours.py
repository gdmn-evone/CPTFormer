from functools import partial
import math
import torch
import torch.nn as nn
from collections import OrderedDict

# Thanks Vision transformer adapter for dense predictions(ICLR) and DeepfakeAdapter(IJCV)
class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=1024):
        super().__init__()

        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(1, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):  
        bs = x.shape[0]
        c1 = self.stem(x)   
        c2 = self.conv2(c1)    
        c3 = self.conv3(c2)     
        c4 = self.conv4(c3)     
        c2 = self.fc2(c2)    
        c3 = self.fc3(c3)    
        c4 = self.fc4(c4)       

        c2 = c2.view(bs, self.embed_dim, -1).transpose(1, 2)  
        c3 = c3.view(bs, self.embed_dim, -1).transpose(1, 2)  
        c4 = c4.view(bs, self.embed_dim, -1).transpose(1, 2)  

        return c2, c3, c4


class Injector(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, 64)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(64, dim))
        ]))
        self.ln_2 = norm_layer(dim)
        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat): 
        attn = self.self_attn(self.query_norm(query.permute(1, 0, 2)), self.feat_norm(feat), value=self.feat_norm(feat))[0]
        attn = attn+ self.mlp(self.ln_2(self.dropout(attn)))
        return query + self.gamma * attn.permute(1, 0, 2)


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, 64)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(64, dim))
        ]))
        self.ln_2 = norm_layer(dim)
        self.dropout = nn.Dropout(0.1)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        attn = self.self_attn(self.query_norm(query), self.feat_norm(feat.permute(1, 0, 2)), value=self.feat_norm(feat.permute(1, 0, 2)))[0]
        attn = attn+ self.mlp(self.ln_2(self.dropout(attn)))
        query = query + self.gamma * attn

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.injector = Injector(dim=dim, num_heads=num_heads, norm_layer=norm_layer)
        self.extractor = Extractor(dim=dim, num_heads=num_heads, norm_layer=norm_layer)

    def forward(self, x, h, blocks):    
        x = self.injector(query=x, feat=h)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        h = self.extractor(query=h, feat=x)

        return x, h


