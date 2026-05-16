import os
import torch
import numpy as np
import warnings
from validate import validate
warnings.filterwarnings("ignore")

vals = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def test_mygen9GANs(model, opt, world_size, rank, cmp = False):

    opt['dataroot'] = ''
    dataroot = opt['dataroot']
    opt['batch_size'] = 32

    if cmp:
        if opt['agnostic']:
            opt['mode'] = 'RandomCmp'
        else:
            opt['mode'] = 'StaticCmp'
    else: 
        opt['mode'] = 'NoCmp' 
     
    accs = {}; aps = {}
    model.eval()
    with torch.no_grad():
        for v_id, val in enumerate(vals):
            opt['dataroot'] = '{}/{}/{}'.format(dataroot, val, opt['mode'])
            opt['classes'] = os.listdir(opt['dataroot']) if multiclass[v_id] else ['']
            opt['no_resize'] = False    # testing without resizing by default
            opt['no_crop'] = True    # testing without resizing by default

            acc, ap, auc, _, _, _, _, _ = validate(model, opt, world_size, rank)
            if rank == 0:
                accs[val] = acc * 100
                aps[val] = ap * 100
    
    if rank == 0:
        avg_acc_list = list(accs.values())
        avg_ap_list = list(aps.values())
        avg_acc = sum(avg_acc_list) / len(avg_acc_list) if avg_acc_list else 0
        avg_ap = sum(avg_ap_list) / len(avg_ap_list) if avg_ap_list else 0
        return accs, aps, avg_acc, avg_ap
    else:
        return None, None, None, None


    
