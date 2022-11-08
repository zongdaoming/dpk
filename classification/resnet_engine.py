
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from timm.utils import accuracy, AverageMeter
from util import get_grad_norm, reduce_tensor
from utils.log_helper import default_logger as logger
from logger import create_logger
from utils.masking_generator import RandomMaskingGenerator
import torch.distributed as dist
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util import  get_grad_norm, auto_resume_helper, reduce_tensor, sim_dis_compute, loss_kl_neck_single, save_last_checkpoint

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    # for idx, (images, target) in enumerate(data_loader):
    #     images = images.cuda(non_blocking=True)
    #     target = target.cuda(non_blocking=True)        
    for idx, (samples, target) in enumerate(data_loader):    
        if len(samples) == 2:
            images = samples[0]
        else:
            images = samples
        images= images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)
        if len(output) == 2:
            output = output[-1]
            
        loss = criterion(output, target)
        acc1, acc3, acc5 = accuracy(output, target, topk=(1,3,5))

        acc1 = reduce_tensor(acc1)
        acc3 = reduce_tensor(acc3)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc3_meter.update(acc3.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@3 {acc3_meter.val:.3f} ({acc3_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@3 {acc3_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc3_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return
def patchify(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 8
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

def unpatchify(x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 8
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
def resnet_epoch(config, teacher, student, device, normlize_target, criterion, soft_criterion, data_loader, optimizer_s, epoch, mixup_fn, lr_scheduler_s, T, tb_logger):
    #three loss type: criterion、soft_criterion、feat_criterion
    student.train()
    teacher.eval()
    optimizer_s.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hard_loss_meter = AverageMeter()
    soft_loss_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    loss_func = nn.MSELoss()
    N =  num_steps
    M =  num_steps 
    for idx, (samples_masks, targets) in enumerate(data_loader):
        # samples[0].shape torch.Size([128,3,224,224])
        # samples[1].shape torch.Size([128,196])
        samples = samples_masks[0]
        #mask images
        samples = patchify(samples)
        N, L, D = samples.shape
        mask_len = int(L * config.TRAIN.MASK_RATIO)
        mask_idx = np.random.permutation(L)[:mask_len]
        samples[:,mask_idx] = 0
        samples = unpatchify(samples)
        
        # torch.Size([128, 3, 224, 224])     
        samples= samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
          
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        feats_s, outputs_student = student(samples)  
                  
        with torch.no_grad():
            feats_t, outputs_teacher = teacher(samples)
            
        outputs_S = F.log_softmax(outputs_student/T, dim=1)
        outputs_T = F.softmax(outputs_teacher/T, dim=1)
        
        
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*1000
            
            loss_feats = loss_func(feats_t[-1].detach(), feats_s[-1]) 
            
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard  +  config.TRAIN.DISTILLATION.ALPHA * loss_soft  + config.TRAIN.DISTILLATION.BETA * loss_feats
           
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
               
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
               
                optimizer_s.step()
                
                optimizer_s.zero_grad()
                # lr_scheduler_s.step_update(epoch * num_steps + idx)
                
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_feats = 0.
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.BETA * loss_feats 
            optimizer_s.zero_grad()                      
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
              
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer_s.step()
            lr_scheduler_s.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        
        if loss_feats is not None:
            feat_loss_meter.update(loss_feats.item(), targets.size(0))        
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer_s.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            if config.TRAIN.DISTILLATION.TYPE != 'logits':

                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_features {feat_loss_meter.val:.4f} ({feat_loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1

    
            else:
    
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
                    f'loss_features {feat_loss_meter.val:.4f} ({feat_loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
def train_resnet_epoch(config, student, device, normlize_target, criterion, data_loader, optimizer, epoch, mixup_fn, tb_logger):
    student.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    soft_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    N =  num_steps
    M =  num_steps 
    
    for idx, (samples, targets) in enumerate(data_loader):
        # samples.shape torch.Size([b,3,224,224])
       
        samples= samples[0].cuda(non_blocking=True)
       
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
        outputs_student = student(samples)
        if len(outputs_student) == 2:
            outputs_student = outputs_student[-1]
        ########################################################################################## Loss Criterion #######################################################################################
        
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            
            loss = criterion(outputs_student, targets)          
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                # lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs_student, targets)
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer.step()
            # lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        loss_meter.update(loss.item(), targets.size(0))       
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
        curr_step = epoch * len(data_loader) + idx + 1
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")