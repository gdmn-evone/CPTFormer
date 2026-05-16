import os
import yaml
import warnings
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from eval_test_mygen9GANs import test_mygen9GANs
from eval_test8gan import test_8GANs
from sklearn.metrics import accuracy_score
from util import log, get_val_opt, seed_everything, print_options
from pytorch_metric_learning import losses, miners

# model
from CPTFormer.HighLip import CLIPModel
from CPTFormer.PRLoss import DynamicPairedRobustnessLoss as PRLoss

# DDP
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore')

def initial_ddp():
    dist.init_process_group(backend='nccl') 
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank
    
def initial():
    seed_everything(200371)
    with open('configs/CPT2.yaml', 'r') as file:  
        train_cfg = yaml.safe_load(file)  
    
    val_cfg = deepcopy(train_cfg)
    val_cfg = get_val_opt(val_cfg)
    test_cfg = deepcopy(val_cfg)
    train_cfg['dataroot'] = os.path.join(train_cfg['dataroot'], train_cfg['train_split'])
    
    return train_cfg, val_cfg, test_cfg
    
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=4 --standalone trainDDP_Phase.py
if __name__ == '__main__':
    
    #============Config and Model initialization============#
    rank, world_size, local_rank = initial_ddp()
    device = torch.device(f'cuda:{local_rank}')

    train_cfg, val_cfg, test_cfg = initial()
    if rank == 0:
        print_options(train_cfg)
        print(f"Running DDP on {world_size} GPUs.")
        print("****************val_cfg：", val_cfg)
        train_writer = SummaryWriter(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name'], "train"))
        val_writer = SummaryWriter(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name'], "val"))
        logger = log(path=os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), file="losses.logs")
    
    data_loader = create_dataloader(train_cfg, world_size, rank)
    if rank == 0:
        print('#Total training images = %d' % len(data_loader.dataset))

    model = CLIPModel(train_cfg["backbone"][5:], train_cfg).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.register_buffer('total_steps', torch.zeros(1, dtype=torch.long))
    
    params = ([p for p in model.parameters()])
    optimizer = torch.optim.AdamW(params,lr=train_cfg['lr'], betas=(0.9, 0.999))


    loss_fn = torch.nn.CrossEntropyLoss()
    loss_prl = PRLoss()
    mining_func = miners.MultiSimilarityMiner()
    ct = losses.CircleLoss()
    if rank == 0:
        early_stopping = EarlyStopping(patience=train_cfg['earlystop_epoch'], delta=0.0001, verbose=True)
    
    #===============Training================#
    last_acc = 0
    weight_dict = {}
    n_weight = 3
    for epoch in range(0, train_cfg['epoch']):
        model.train()
        data_loader.sampler.set_epoch(epoch)
        train_loss, train_true, train_pred = 0, [], []
                
        for data in tqdm(data_loader, disable=(rank != 0), desc=f"Epoch {epoch}/{train_cfg['epoch']}"):
            model.module.total_steps += 1 
            
            aug_input, no_aug_input, aug_input_phase, no_aug_input_phase, tf_label, cmp_label = data

            aug_input = aug_input.to(device, non_blocking=True)
            no_aug_input = no_aug_input.to(device, non_blocking=True)
            aug_input_phase = aug_input_phase.to(device, non_blocking=True)
            no_aug_input_phase = no_aug_input_phase.to(device, non_blocking=True)

            tf_label = tf_label.to(device, non_blocking=True) 
            cmp_label = cmp_label.to(device, non_blocking=True)

            num_paired_samples = torch.sum(cmp_label)
            input_tensor = torch.cat([no_aug_input[cmp_label], 
                              aug_input[cmp_label], 
                              aug_input[~cmp_label]], dim=0)
            input_phase_tensor = torch.cat([no_aug_input_phase[cmp_label], 
                              aug_input_phase[cmp_label], 
                              aug_input_phase[~cmp_label]], dim=0)
            tf_label_combined = torch.cat([tf_label[cmp_label], 
                                   tf_label[cmp_label], 
                                   tf_label[~cmp_label]], dim=0)
            mask_label = torch.ones(len(input_tensor), device=device, dtype=torch.bool)
            mask_label[num_paired_samples*2:] = False

            out, fea = model(input_tensor, input_phase_tensor)
            if epoch < 6:
                tf_loss = loss_fn(out.squeeze(-1), tf_label_combined)
            else:
                logits_o = out[:num_paired_samples]
                logits_c = out[num_paired_samples : num_paired_samples * 2]
                logits_unpair = out[num_paired_samples * 2 :]

                labels_paired = tf_label[cmp_label]
                labels_unpaired = tf_label[~cmp_label]
                
                if num_paired_samples > 0:
                    prl_loss = loss_prl(logits_o, logits_c, labels_paired)
                else:
                    prl_loss = torch.tensor(0.0, device=device) 

                if len(labels_unpaired) > 0:
                    unpair_loss = loss_fn(logits_unpair.squeeze(-1), labels_unpaired)
                else:
                    unpair_loss = torch.tensor(0.0, device=device)
                tf_loss = prl_loss + unpair_loss

            hard_tuples_x = mining_func(fea, tf_label_combined )
            ct_loss = ct(fea, tf_label_combined , hard_tuples_x) / (train_cfg['batch_size']/world_size)
            loss = tf_loss + ct_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred.extend(torch.argmax(out.detach(), dim=1).flatten().cpu().tolist())
            train_true.extend(tf_label_combined.detach().cpu().flatten().tolist())
            
            train_loss += (tf_loss.item() + ct_loss.item())
            
            if rank == 0 and model.module.total_steps % train_cfg['loss_freq'] == 0:
                print(f"Tf loss: {tf_loss.item()} ct_loss: {ct_loss.item()}, at step: {model.module.total_steps}")
                train_writer.add_scalar('loss_tf', tf_loss, model.module.total_steps)
                train_writer.add_scalar('loss_ct', ct_loss, model.module.total_steps)

        # 1. Aggregate training loss
        total_train_loss = torch.tensor(train_loss, device=device)
        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)

        # 2. Aggregate predictions
        all_train_true, all_train_pred = [None] * world_size, [None] * world_size
        dist.all_gather_object(all_train_true, train_true)
        dist.all_gather_object(all_train_pred, train_pred)

        model.eval()
        acc, ap = validate(model, val_cfg, world_size, rank)[:2]
        accs9, aps9, avg_acc9, avg_ap9 = test_mygen9GANs(model, test_cfg, world_size, rank, True)
        accs8, aps8, avg_acc8, avg_ap8 = test_8GANs(model, test_cfg, world_size, rank, True)

        # Validation
        if rank == 0:
            final_train_true = np.array([item for sublist in all_train_true for item in sublist])
            final_train_pred = np.array([item for sublist in all_train_pred for item in sublist])
            num_train_samples = len(data_loader.dataset)
            final_train_true = final_train_true[:num_train_samples]
            final_train_pred = final_train_pred[:num_train_samples]
            train_acc = accuracy_score(final_train_true, final_train_pred > 0.5)

            avg_train_loss = total_train_loss.item() / (world_size * len(data_loader))
            val_writer.add_scalar('accuracy', acc, model.module.total_steps)
            val_writer.add_scalar('ap', ap, model.module.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

            log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, val acc：{:.4f}, val ap：{:.4f}".format(
                            epoch+1,
                            train_cfg['epoch'],
                            avg_train_loss,
                            train_acc,
                            acc,
                            ap,
                            )

            save_filename = 'model_epoch_latest.pth' 
            state_dict = {'epoch':epoch, 'model': model.module.state_dict(),'optimizer': optimizer.state_dict()}
            save_path = os.path.join(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), save_filename)
            torch.save(state_dict, save_path)
            
            all_acc_avg = 0
            
            for key, value in accs9.items():
                val_writer.add_scalar(key + "acc" + "cmp50", value, epoch)
            for key, value in aps9.items():
                val_writer.add_scalar(key + "ap" + "cmp50", value, epoch)
            val_writer.add_scalar("avg_acc_9GANs" + "cmp50", avg_acc9, epoch)
            val_writer.add_scalar("avg_ap_9GANs" + "cmp50", avg_ap9, epoch)
            print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test9GAN CMP",'Mean', avg_acc9, avg_ap9))
            log_text+="\n 9GANs acc: {:.4f}, 9GANs ap: {:.4f}".format(avg_acc9, avg_ap9)
            all_acc_avg += avg_acc9
            
            
            for key, value in accs8.items():
                val_writer.add_scalar(key + "acc" + "cmp50", value, epoch)
            for key, value in aps8.items():
                val_writer.add_scalar(key + "ap" + "cmp50", value, epoch)
            val_writer.add_scalar("avg_acc_8GANs" + "cmp50", avg_acc8, epoch)
            val_writer.add_scalar("avg_ap_8GANs" + "cmp50", avg_ap8, epoch)
            print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format("test8GAN CMP",'Mean', avg_acc8, avg_ap8))
            
            all_acc_avg = (all_acc_avg + avg_acc8) / 2
            log_text+=" 8GANs acc: {:.4f}, 8GANs ap: {:.4f}\n ALL AVG：{:.4f}".format(avg_acc8, avg_ap8, all_acc_avg)
            
            if len(weight_dict) < n_weight:
                save_filename = 'model_{}_9+8Gan_{:.2f}.pth'.format(epoch, all_acc_avg) 
                state_dict = {'epoch':epoch, 'model': model.module.state_dict(),'optimizer': optimizer.state_dict()}
                save_path = os.path.join(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), save_filename)
                weight_dict[save_path] = all_acc_avg
                torch.save(state_dict, save_path)
                last_acc = min(weight_dict[k] for k in weight_dict)

            elif all_acc_avg >= last_acc:
                worst_path_to_delete = min(weight_dict, key=weight_dict.get)
                del weight_dict[worst_path_to_delete]
                if os.path.exists(worst_path_to_delete):
                    os.remove(worst_path_to_delete)
                save_filename = f'model_{epoch}_9+8Gan_{all_acc_avg:.2f}.pth'
                save_path = os.path.join(os.path.join(train_cfg['checkpoints_dir'], train_cfg['name']), save_filename)
                state_dict = {'epoch':epoch, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state_dict, save_path)
                weight_dict[save_path] = all_acc_avg
                last_acc = min(weight_dict[k] for k in weight_dict)

            
            logger.info(log_text)
            early_stopping(acc, epoch, model.module, optimizer)
            if early_stopping.early_stop:
                model.module.adjust_learning_rate(optimizer)
                print("Learning rate adjusted, continue training...")
                early_stopping = EarlyStopping(patience=train_cfg['earlystop_epoch'], delta=-0.0001, verbose=True)
        dist.barrier()
        