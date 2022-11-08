
from errno import EPIPE
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
import copy


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
    
def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


def at_loss(g_s, g_t, p=2):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])


def get_att_dis(stu, tea):
    b, stu_chans, stu_dims = stu.shape[0], stu.shape[1], stu.shape[2]
    b, tea_chans, tea_dims = tea.shape[0], tea.shape[1], tea.shape[2]
    assert stu_dims == tea_dims, "Student dimensions equal to Teacher dimensions"
    attention_distribution = torch.zeros(size=(b,stu_chans,tea_chans)) # torch.Size([32, 128, 512])
    for i in range(stu.shape[1]):
        for j in range(tea.shape[1]):
            attention_score = torch.cosine_similarity(stu[:,i,:], tea[:,j,:])                    
            attention_distribution[:,i,j] = attention_score
    return attention_distribution.mean()

def train_resnet_epoch(config, teacher, student, device, normlize_target, criterion, soft_criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, T, tb_logger):
    student.train()
    teacher.eval()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hard_loss_meter = AverageMeter()
    soft_loss_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    N =  num_steps
    M =  num_steps 
    loss_f = nn.MSELoss()
    
    for idx, (samples, targets) in enumerate(data_loader):
        bool_masked_pos = samples[1].cuda(non_blocking=True).to(torch.bool)
        samples= samples[0].cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        feats_s, outputs_student = student(samples)            

        with torch.no_grad():
            feats_t, outputs_teacher = teacher(samples)
                
        outputs_S = F.log_softmax(outputs_student/T, dim=1)
        outputs_T = F.softmax(outputs_teacher/T, dim=1)
        loss_feats = None
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*1000
            if config.TRAIN.DISTILLATION.TYPE == 'at':
                loss_feats =  config.TRAIN.DISTILLATION.BETA * at_loss(feats_s, feats_t, 2)
            
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard +  config.TRAIN.DISTILLATION.ALPHA * loss_soft + loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft       
            loss = loss / config.TRAIN.ACCUMULATION_STEPS                       
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
             
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*1000
            if config.TRAIN.DISTILLATION.TYPE == 'fitnet':
                loss_feats = config.TRAIN.DISTILLATION.BETA * at_loss(feats_s, feats_t, 2) 
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft + loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft       
            
            optimizer.zero_grad()
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer.step()

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0)) 
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        if loss_feats != None:
            feat_loss_meter.update(loss_feats.item(), targets.size(0))      
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if loss_feats == None:
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
                    # f'loss_features {feat_loss_meter.val:.4f} ({feat_loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
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
    

def train_cifar_one_epoch(config, teacher, student, masked_autoencoder, device, normlize_target, criterion, soft_criterion, feat_criterion, data_loader, optimizer_s, optimizer_mae, epoch, mixup_fn, lr_scheduler_s, lr_scheduler_mae, T, tb_logger):
    #three loss type: criterion、soft_criterion、feat_criterion
    student.train()
    masked_autoencoder.train()
    
    teacher.eval()
    optimizer_s.zero_grad()
    optimizer_mae.zero_grad()
    
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
    hsic_matrix = torch.zeros(N,3)
    
    def HSIC(K, L, device):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()    
    mask_ratio = []
    for idx, (samples, targets) in enumerate(data_loader):
        # samples[0].shape torch.Size([128,3,224,224])
        # samples[1].shape torch.Size([128,196])
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
        index = config.TRAIN.DISTILLATION.FEAT_INDEX
        images = feats_s[index]
        target_images = feats_t[index]
        
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*100
            
            loss_feats = None
            # [B,CHW]
            X  = images.flatten(1)
            # [B,B]
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            hsic_matrix[idx, 0] += HSIC(K, K, device) / num_steps                        
            
            
            Y = feats_t[index].flatten(1)
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
            
            if config.MASKED_AUTOENCODER.TYPE == "dynamic":
                
                dynamic_ratio = 1.0 - torch.sigmoid(torch.as_tensor(HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device))))) # the larger the similarity score, the smaller the teacher participant
            elif config.MASKED_AUTOENCODER.TYPE == "cka":
                tmp = HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device)))
                if np.isnan(tmp) or np.isinf(tmp):
                    dynamic_ratio = 0.85
                else:
                    dynamic_ratio = max(1.0 - torch.as_tensor(tmp), 0.10)
            else:
                dynamic_ratio = config.TRAIN.MASK_RATIO

            if dynamic_ratio <= 0 or dynamic_ratio >= 1.0:
                dynamic_ratio = 0.75
            
            num_patches = target_images.shape[2] * target_images.shape[3] 
            num_mask = int(num_patches * dynamic_ratio)
            bool_masked_pos_dynamic = []
            num_mask = max(num_mask, 4)
            for i in range(target_images.shape[0]):
                mask = np.hstack([ np.zeros(num_patches - num_mask),np.ones(num_mask)])
                np.random.shuffle(mask)
                mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                bool_masked_pos_dynamic.append(mask)
                
            bool_masked_pos = torch.stack(bool_masked_pos_dynamic)  
            mask_ratio.append(dynamic_ratio)  
            
            hsic_matrix[idx, 1] += HSIC(K, L, device) / num_steps
            hsic_matrix[idx, 2] += HSIC(L, L, device) / num_steps
            if images.shape[2:] != target_images.shape[2:]:
                images = F.interpolate(images, size=target_images.size()[2:], mode='bilinear', align_corners=True)
                                                              
            # torch.Size([32, 49, 1024])                        
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                if config.TRAIN.DISTILLATION.TYPE == 'feat':                    
                    with torch.no_grad():
                        # calculate the predict label
                        unnorm_images = target_images     
                        if normlize_target:
                            images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            # images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                            #     ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                            images_norm  = F.normalize(images_squeeze, p=2, dim=-1)
                            
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                        else:
                            images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                       
                        B, _, C = images_patch.shape # torch.Size([128, hw, p1p2c])
                    
                    # import pdb; pdb.set_trace() [b,196,4096]                       
                    outputs = masked_autoencoder(images, target_images, bool_masked_pos)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    
                    loss_feats = loss_func(input=outputs, target=images_patch.detach()) * config.TRAIN.DISTILLATION.BETA 
                    # loss_feats = None
            
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft + loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft                        
                    
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer_s.step()
                optimizer_mae.step()
                optimizer_s.zero_grad()
                optimizer_mae.zero_grad()
                lr_scheduler_mae.step_update(epoch * num_steps + idx)
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_student, outputs_teacher)
            
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft  
            optimizer_s.zero_grad()                      
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            optimizer_s.step()
            optimizer_mae.step()
            lr_scheduler_mae.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
        
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
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
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
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1
    hsic_matrix = hsic_matrix[:,1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:,2].sqrt())
    assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"HSIC_MATRIX shape {hsic_matrix.shape}")
    return hsic_matrix, mask_ratio


def train_cifar_two_epoch(config, teacher, student, masked_autoencoder_1, masked_autoencoder_2, device, normlize_target, criterion, soft_criterion, feat_criterion, data_loader, optimizer_s, optimizer_mae_1, optimizer_mae_2, epoch, mixup_fn, lr_scheduler_s, lr_scheduler_mae_1, lr_scheduler_mae_2, T, tb_logger):
    #three loss type: criterion、soft_criterion、feat_criterion
    student.train()
    masked_autoencoder_1.train()
    masked_autoencoder_2.train()
    teacher.eval()
    optimizer_s.zero_grad()
    optimizer_mae_1.zero_grad()
    optimizer_mae_2.zero_grad()
    
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
    hsic_matrix = torch.zeros(N,3)
    
    def HSIC(K, L, device):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()    
    mask_ratio = []
    for idx, (samples_masks, targets) in enumerate(data_loader):
        # samples[0].shape torch.Size([128,3,224,224])
        # samples[1].shape torch.Size([128,196])
        samples = samples_masks
        samples= samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        feats_s, outputs_student = student(samples)  
        
        with torch.no_grad():
            feats_t, outputs_teacher = teacher(samples)
            
        if feats_s[-2].shape[2:] != feats_t[-2].shape[2:]:
            feats_s[-2] = F.interpolate(feats_s[-2], size=feats_t[-2].size()[2:], mode='bilinear', align_corners=True)
        if feats_s[-3].shape[2:] != feats_t[-3].shape[2:]:
            feats_s[-3] = F.interpolate(feats_s[-3], size=feats_t[-3].size()[2:], mode='bilinear', align_corners=True)
        h1, w1 = feats_s[-2].shape[2:]
        h2, w2 = feats_s[-3].shape[2:]
        bool_masked_pos_1 = torch.ones(batch_size, h1, w1)
        bool_masked_pos_2 = torch.ones(batch_size, h2, w2)
        bool_masked_pos_1 = bool_masked_pos_1.to(device, non_blocking=True).flatten(1).to(torch.bool) # torch.Size([128, 49])  
        bool_masked_pos_2 = bool_masked_pos_2.to(device, non_blocking=True).flatten(1).to(torch.bool) # torch.Size([128, 49])           
        outputs_S = F.log_softmax(outputs_student/T, dim=1)
        outputs_T = F.softmax(outputs_teacher/T, dim=1)
        
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*100
            loss_feats = None
            # images.shape                                                        │  
            index = config.TRAIN.DISTILLATION.FEAT_INDEX
            # [B,CHW]
            X_1  = feats_s[index].flatten(1)
            # [B,B]
            K_1 = X_1 @ X_1.t()
            K_1.fill_diagonal_(0.0)
            hsic_matrix[idx, 0] += HSIC(K_1, K_1, device) / num_steps                        
            images_1 =  feats_s[-2]
            images_2 =  feats_s[-3]
            Y_1 = feats_t[index].flatten(1)
            L_1 = Y_1 @ Y_1.t()
            L_1.fill_diagonal_(0.0)
            assert K_1.shape == L_1.shape, f"Feature shape mistach! {K_1.shape}, {L_1.shape}"
            
            X_2  = feats_s[index-1].flatten(1)
            # [B,B]
            K_2 = X_2 @ X_2.t()
            K_2.fill_diagonal_(0.0)
            
            Y_2 = feats_t[index-1].flatten(1)
            L_2 = Y_2 @ Y_2.t()
            L_2.fill_diagonal_(0.0)
            hsic_matrix[idx, 1] += HSIC(K_1, L_1, device) / num_steps
            hsic_matrix[idx, 2] += HSIC(L_1, L_1, device) / num_steps
            target_images_1 = feats_t[-2].detach()
            target_images_2 = feats_t[-3].detach()
            if config.MASKED_AUTOENCODER.TYPE == "cka":
                a = np.abs(HSIC(K_1, L_1, device)) + 1e-5
                b = np.sqrt(HSIC(K_1, K_1, device) + 1e-5) * np.sqrt(HSIC(L_1, L_1, device)+1e-5)
                tmp = a / b
                if np.isnan(tmp) or np.isinf(tmp) or tmp <= 0.1 or tmp >= 1:
                    mask_ratio_1 = 0.9
                else:
                    mask_ratio_1 = max(1.0 - torch.as_tensor(tmp), 0.15)
                a = np.abs(HSIC(K_2, L_2, device)) + 1e-5
                b = np.sqrt(HSIC(K_2, K_2, device) + 1e-5) * np.sqrt(HSIC(L_2, L_2, device)+1e-5)
                tmp = a / b
                if np.isnan(tmp) or np.isinf(tmp) or tmp <= 0.1 or tmp >= 1:
                    mask_ratio_2 = 0.9
                else:
                    mask_ratio_2 = max(1.0 - torch.as_tensor(tmp), 0.15)
            else:
                mask_ratio_1 = mask_ratio_2 = config.TRAIN.MASK_RATIO
            mask_ratio.append([mask_ratio_1,mask_ratio_2])
            if config.MASKED_AUTOENCODER.TYPE != None:
                num_patches_1 = target_images_1.shape[2]*target_images_1.shape[3]
                num_patches_2 = target_images_2.shape[2]*target_images_2.shape[3] 
                num_mask_1 = int(num_patches_1 * mask_ratio_1)
                num_mask_2 = int(num_patches_2 * mask_ratio_2)
                num_mask_1 = max(num_mask_1, 2)
                num_mask_2 = max(num_mask_2, 2)
                bool_masked_pos_dynamic_1 = []
                bool_masked_pos_dynamic_2 = []
                for i in range(target_images_1.shape[0]):
                    mask_1 = np.hstack([np.zeros(num_patches_1 - num_mask_1),np.ones(num_mask_1)])
                    np.random.shuffle(mask_1)
                    mask_1 = torch.as_tensor(mask_1, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_1.append(mask_1)
                    mask_2 = np.hstack([np.zeros(num_patches_2 - num_mask_2),np.ones(num_mask_2)])
                    np.random.shuffle(mask_2)
                    mask_2 = torch.as_tensor(mask_2, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_2.append(mask_2)
                bool_masked_pos_1 = torch.stack(bool_masked_pos_dynamic_1)  
                bool_masked_pos_2 = torch.stack(bool_masked_pos_dynamic_2)                       
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                if config.TRAIN.DISTILLATION.TYPE == 'feat':                    
                    with torch.no_grad():
                        # calculate the predict label
                        unnorm_images_1 = target_images_1 
                        unnorm_images_2 = target_images_2
                        if normlize_target:
                            images_squeeze_1 = rearrange(unnorm_images_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_norm_1  = F.normalize(images_squeeze_1, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_1 = rearrange(images_norm_1, 'b n p c -> b n (p c)')
                            
                            images_squeeze_2 = rearrange(unnorm_images_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_norm_2  = F.normalize(images_squeeze_2, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_2 = rearrange(images_norm_2, 'b n p c -> b n (p c)')
                        else:
                            images_patch_1 = rearrange(unnorm_images_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_patch_2 = rearrange(unnorm_images_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                       
                        B, _, C = images_patch_1.shape # torch.Size([128, hw, p1p2c])                     
                    outputs_1 = masked_autoencoder_1(images_1, target_images_1, bool_masked_pos_1)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    outputs_2 = masked_autoencoder_2(images_2, target_images_2, bool_masked_pos_2)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    
                    images_patch_1 = images_patch_1.detach()
                    images_patch_2 = images_patch_2.detach()
                    loss_feats = loss_func(input=outputs_1, target=images_patch_1)  + loss_func(input=outputs_2, target=images_patch_2) * 0.6
            
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft + config.TRAIN.DISTILLATION.BETA * loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft                        
                    
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder_1.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                # import pdb 
                # pdb.set_trace()
                optimizer_s.step()
                optimizer_mae_1.step()
                optimizer_mae_2.step()
                optimizer_s.zero_grad()
                optimizer_mae_1.zero_grad()
                optimizer_mae_2.zero_grad()
                lr_scheduler_mae_1.step_update(epoch * num_steps + idx)
                lr_scheduler_mae_2.step_update(epoch * num_steps + idx)
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_student, outputs_teacher)
            
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft  
            optimizer_s.zero_grad()                      
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder_1.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer_s.step()
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
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
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
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
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                curr_step = epoch * len(data_loader) + idx + 1
    hsic_matrix = hsic_matrix[:,1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:,2].sqrt())
    assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"HSIC_MATRIX shape {hsic_matrix.shape}")
    return hsic_matrix, mask_ratio
    
def train_resnet_one_epoch(config, teacher, student, masked_autoencoder, device, normlize_target, criterion, soft_criterion, feat_criterion, data_loader, optimizer_s, optimizer_mae, epoch, mixup_fn, lr_scheduler_s, lr_scheduler_mae, T, tb_logger):
    #three loss type: criterion、soft_criterion、feat_criterion
    student.train()
    masked_autoencoder.train()
    teacher.eval()
    optimizer_s.zero_grad()
    optimizer_mae.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hard_loss_meter = AverageMeter()
    soft_loss_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    g_meter = AverageMeter()
    start = time.time()
    end = time.time()
    loss_func = nn.MSELoss()
    
    N =  num_steps
    M =  num_steps
    hsic_matrix = torch.zeros(N,3)
    
    def HSIC(K, L, device):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()    
    
    mask_ratio = []
   
    for idx, (samples_masks, targets) in enumerate(data_loader):
        samples = samples_masks[0] 
        samples= samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        feats_s, outputs_student = student(samples)  
        if 'efficientnet' in config.TEACHER_MODEL.TYPE:
            with torch.no_grad():
                feats_t = teacher.module.forward_features(samples)
                out = teacher.module.global_pool(feats_t)
                outputs_teacher = teacher.module.classifier(out)
                feats_t = [feats_t]
        else:
            with torch.no_grad():
                feats_t, outputs_teacher = teacher(samples)
        outputs_S = F.log_softmax(outputs_student/T, dim=1)
        outputs_T = F.softmax(outputs_teacher/T, dim=1)
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*1000
            loss_feats = None                              
            index = config.TRAIN.DISTILLATION.FEAT_INDEX
            # [B,CHW]
            X  = feats_s[index].flatten(1)
            # [B,B]
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            hsic_matrix[idx, 0] += HSIC(K, K, device) / num_steps                        
            images =  feats_s[index]
            Y = feats_t[index].flatten(1)
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
            hsic_matrix[idx, 1] += HSIC(K, L, device) / num_steps
            hsic_matrix[idx, 2] += HSIC(L, L, device) / num_steps
            target_images = feats_t[index].detach()             
            bool_masked_pos = None
            if config.MASKED_AUTOENCODER.TYPE == "sigmoid":
                dynamic_ratio = 1.0 - torch.sigmoid(torch.as_tensor(HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device))))) 
            elif config.MASKED_AUTOENCODER.TYPE == "random":
                dynamic_ratio = torch.rand(1)
            elif config.MASKED_AUTOENCODER.TYPE == "step":
                dynamic_ratio = (1.0 - epoch / 200.0) * 0.95 
            elif config.MASKED_AUTOENCODER.TYPE == 'others':
                dynamic_ratio = 0.5 + torch.sigmoid(-torch.as_tensor(HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device)))))
            elif config.MASKED_AUTOENCODER.TYPE == "exp":
                dynamic_ratio =  0.99 ** epoch
            elif config.MASKED_AUTOENCODER.TYPE == "cka":
                tmp = HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device)))
                if np.isnan(tmp) or np.isinf(tmp):
                    dynamic_ratio = 0.85
                else:
                    dynamic_ratio = max(1.0 - torch.as_tensor(tmp), 0.10)
            elif config.MASKED_AUTOENCODER.TYPE == "cos":
                similarity = torch.cosine_similarity(images.flatten(2),target_images.flatten(2),-1).mean()       
                dynamic_ratio = 1.0 - similarity.detach().item() 
            else:
                dynamic_ratio = config.TRAIN.MASK_RATIO
            
            mask_ratio.append(dynamic_ratio)
            if bool_masked_pos == None:
                
                bool_masked_pos_dynamic = []
                num_patches = target_images.shape[2]*target_images.shape[3]
                dynamic_ratio = min(dynamic_ratio, 0.95)
                num_mask = int(num_patches * dynamic_ratio)
                num_mask = max(num_mask, 2)
                num_mask = min(num_mask, num_patches - 2)
                for i in range(target_images.shape[0]):
                    mask = np.hstack([np.zeros(num_patches - num_mask),np.ones(num_mask)])
                    np.random.shuffle(mask)
                    mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic.append(mask)
                bool_masked_pos = torch.stack(bool_masked_pos_dynamic)                        
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                if config.TRAIN.DISTILLATION.TYPE == 'feat':                    
                    with torch.no_grad():
                        unnorm_images = target_images  
                        if normlize_target:
                            images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            # images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                            #     ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                            images_norm  = F.normalize(images_squeeze, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                        else:
                            images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                       
                        B, _, C = images_patch.shape # torch.Size([128, hw, p1p2c])   
                    
                    outputs = masked_autoencoder(images, target_images, bool_masked_pos) 
                    loss_feats = loss_func(input=outputs, target=images_patch.detach()) 
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft + config.TRAIN.DISTILLATION.BETA * loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft                         
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer_s.step()
                optimizer_mae.step()
                
                optimizer_s.zero_grad()
                optimizer_mae.zero_grad()
                lr_scheduler_mae.step_update(epoch * num_steps + idx)
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_student, outputs_teacher)
            
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft  
            optimizer_s.zero_grad()                      
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer_s.step()

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
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
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
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
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1
    
    hsic_matrix = hsic_matrix[:,1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:,2].sqrt())
    assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"HSIC_MATRIX shape {hsic_matrix.shape}")
    return hsic_matrix, mask_ratio

                
def train_resnet_two_epoch(config, teacher, student, masked_autoencoder_1, masked_autoencoder_2, device, normlize_target, criterion, soft_criterion, feat_criterion, data_loader, optimizer_s, optimizer_mae_1, optimizer_mae_2, epoch, mixup_fn, lr_scheduler_s, lr_scheduler_mae_1, lr_scheduler_mae_2, T, tb_logger):
    #three loss type: criterion、soft_criterion、feat_criterion
    student.train()
    masked_autoencoder_1.train()
    masked_autoencoder_2.train()
    teacher.eval()
    
    optimizer_s.zero_grad()
    optimizer_mae_1.zero_grad()
    optimizer_mae_2.zero_grad()
    
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
    hsic_matrix = torch.zeros(N,3)
    
    def HSIC(K, L, device):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()    
    
    mask_ratio = []
    
    for idx, (samples_masks, targets) in enumerate(data_loader):
        # samples[0].shape torch.Size([128,3,224,224])
        # samples[1].shape torch.Size([128,196])
        samples = samples_masks[0]
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
            
            loss_feats = None
            # images.shape                                                        │  
            index = config.TRAIN.DISTILLATION.FEAT_INDEX
            # [B,CHW]
            X_1  = feats_s[index].flatten(1)
            # [B,B]
            K_1 = X_1 @ X_1.t()
            K_1.fill_diagonal_(0.0)
            hsic_matrix[idx, 0] += HSIC(K_1, K_1, device) / num_steps                        
            images_1 =  feats_s[index]
            images_2 =  feats_s[index-1]
            
            Y_1 = feats_t[index].flatten(1)
            L_1 = Y_1 @ Y_1.t()
            L_1.fill_diagonal_(0.0)
            assert K_1.shape == L_1.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
            
            hsic_matrix[idx, 1] += HSIC(K_1, L_1, device) / num_steps
            hsic_matrix[idx, 2] += HSIC(L_1, L_1, device) / num_steps
            
            target_images_1 = feats_t[index].detach()
            target_images_2 = feats_t[index-1].detach()
            
            X_2  = feats_s[index-1].flatten(1)
            # [B,B]
            K_2 = X_2 @ X_2.t()
            K_2.fill_diagonal_(0.0)
            
            Y_2 = feats_t[index-1].flatten(1)
            L_2 = Y_2 @ Y_2.t()
            L_2.fill_diagonal_(0.0)
            
            bool_masked_pos_1 = None
            if config.MASKED_AUTOENCODER.TYPE == "sigmoid":
                dynamic_ratio_1 = 1.0 - torch.sigmoid(torch.as_tensor(HSIC(K_1, L_1, device) / (np.sqrt(HSIC(K_1, K_1, device)) * np.sqrt(HSIC(L_1, L_1, device))))) # the larger the similarity score, the smaller the teacher participant
                dynamic_ratio_2 = 1.0 - torch.sigmoid(torch.as_tensor(HSIC(K_2, L_2, device) / (np.sqrt(HSIC(K_2, K_2, device)) * np.sqrt(HSIC(L_2, L_2, device))))) 
            elif config.MASKED_AUTOENCODER.TYPE == "cka":
                # the larger the similarity score, the smaller the teacher participant
                tmp = HSIC(K_1, L_1, device) / (np.sqrt(HSIC(K_1, K_1, device)) * np.sqrt(HSIC(L_1, L_1, device)))
                if np.isnan(tmp) or np.isinf(tmp) or tmp <= 0.1 or tmp >= 1:
                    dynamic_ratio_1 = 0.8
                else:
                    dynamic_ratio_1 = max(1.0 - torch.as_tensor(tmp), 0.1)
                tmp = HSIC(K_2, L_2, device) / (np.sqrt(HSIC(K_2, K_2, device)) * np.sqrt(HSIC(L_2, L_2, device)))
                if np.isnan(tmp) or np.isinf(tmp) or tmp <= 0.1 or tmp >= 1:
                    dynamic_ratio_2 = 0.8
                else:
                    dynamic_ratio_2 = max(1.0 - torch.as_tensor(tmp), 0.1) 
            else:
                dynamic_ratio_1 = config.TRAIN.MASK_RATIO
                dynamic_ratio_2 = config.TRAIN.MASK_RATIO
            mask_ratio.append([dynamic_ratio_1, dynamic_ratio_2])
            if bool_masked_pos_1 == None:
                num_patches_1 = target_images_1.shape[2]*target_images_1.shape[3]
                num_patches_2 = target_images_2.shape[2]*target_images_2.shape[3] 
                num_mask_1 = int(num_patches_1 * dynamic_ratio_1)
                num_mask_2 = int(num_patches_2 * dynamic_ratio_2)
                num_mask_1 = max(num_mask_1, 4)
                num_mask_2 = max(num_mask_2, 4)
                bool_masked_pos_dynamic_1 = []
                bool_masked_pos_dynamic_2 = []
                for i in range(target_images_1.shape[0]):
                    mask_1 = np.hstack([np.zeros(num_patches_1 - num_mask_1),np.ones(num_mask_1)])
                    np.random.shuffle(mask_1)
                    mask_1 = torch.as_tensor(mask_1, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_1.append(mask_1)
                    mask_2 = np.hstack([np.zeros(num_patches_2 - num_mask_2),np.ones(num_mask_2)])
                    np.random.shuffle(mask_2)
                    mask_2 = torch.as_tensor(mask_2, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_2.append(mask_2)
                bool_masked_pos_1 = torch.stack(bool_masked_pos_dynamic_1)  
                bool_masked_pos_2 = torch.stack(bool_masked_pos_dynamic_2)  
                                  
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                if config.TRAIN.DISTILLATION.TYPE == 'feat':                    
                    with torch.no_grad():
                        # calculate the predict label
                        unnorm_images_1 = target_images_1 
                        unnorm_images_2 = target_images_2
                        if normlize_target:
                            images_squeeze_1 = rearrange(unnorm_images_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_norm_1  = F.normalize(images_squeeze_1, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_1 = rearrange(images_norm_1, 'b n p c -> b n (p c)')
                            
                            images_squeeze_2 = rearrange(unnorm_images_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_norm_2  = F.normalize(images_squeeze_2, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_2 = rearrange(images_norm_2, 'b n p c -> b n (p c)')
                        else:
                            images_patch_1 = rearrange(unnorm_images_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_patch_2 = rearrange(unnorm_images_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                       
                        B, _, C = images_patch_1.shape # torch.Size([128, hw, p1p2c])
                    
                    # import pdb; pdb.set_trace() [b,196,4096]                       
                    outputs_1 = masked_autoencoder_1(images_1, target_images_1, bool_masked_pos_1)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    outputs_2 = masked_autoencoder_2(images_2, target_images_2, bool_masked_pos_2)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    images_patch_1 = images_patch_1.detach()
                    images_patch_2 = images_patch_2.detach()
                    loss_feats = (loss_func(input=outputs_1, target=images_patch_1) + loss_func(input=outputs_2, target=images_patch_2) ) * config.TRAIN.DISTILLATION.BETA
            
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft +  loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft                        
                    
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder_1.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                # import pdb 
                # pdb.set_trace()
                optimizer_s.step()
                optimizer_mae_1.step()
                optimizer_mae_2.step()
                optimizer_s.zero_grad()
                optimizer_mae_1.zero_grad()
                optimizer_mae_2.zero_grad()
                lr_scheduler_mae_1.step_update(epoch * num_steps + idx)
                lr_scheduler_mae_2.step_update(epoch * num_steps + idx)
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_student, outputs_teacher)
            
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft  
            optimizer_s.zero_grad()                      
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder_1.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            optimizer_s.step()
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
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
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
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
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1
    
    hsic_matrix = hsic_matrix[:,1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:,2].sqrt())
    assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"HSIC_MATRIX shape {hsic_matrix.shape}")
    return hsic_matrix, mask_ratio

   
def train_resnet_three_epoch(config, teacher, student, masked_autoencoder_1, masked_autoencoder_2, masked_autoencoder_3, device, normlize_target, criterion, soft_criterion, feat_criterion, data_loader, optimizer_s, optimizer_mae_1, optimizer_mae_2, optimizer_mae_3, epoch, mixup_fn, lr_scheduler_s, lr_scheduler_mae_1, lr_scheduler_mae_2, lr_scheduler_mae_3, T, tb_logger):
    #three loss type: criterion、soft_criterion、feat_criterion
    student.train()
    masked_autoencoder_1.train()
    masked_autoencoder_2.train()
    masked_autoencoder_3.train()
    teacher.eval()
    
    optimizer_s.zero_grad()
    optimizer_mae_1.zero_grad()
    optimizer_mae_2.zero_grad()
    optimizer_mae_3.zero_grad()
    
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
    hsic_matrix = torch.zeros(N,3)
    
    def HSIC(K, L, device):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()    
    
    mask_ratio = []
    
    for idx, (samples_masks, targets) in enumerate(data_loader):
        # samples[0].shape torch.Size([128,3,224,224])
        # samples[1].shape torch.Size([128,196])
        samples = samples_masks[0]

        # torch.Size([128, 3, 224, 224])     
        samples= samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = samples.shape[0]
          
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        feats_s, outputs_student = student(samples)  
        h1, w1 = feats_s[-1].shape[2:] #7
        h2, w2 = feats_s[-2].shape[2:] #14
        h3, w3 = feats_s[-3].shape[2:]  #28 
        bool_masked_pos_1 = torch.ones(batch_size, h1, w1)
        bool_masked_pos_2 = torch.ones(batch_size, h2//2, w2//2)
        bool_masked_pos_3 = torch.ones(batch_size, h3//4, w3//4)
        
        bool_masked_pos_1 = bool_masked_pos_1.to(device, non_blocking=True).flatten(1).to(torch.bool) # torch.Size([128, 49])  
        bool_masked_pos_2 = bool_masked_pos_2.to(device, non_blocking=True).flatten(1).to(torch.bool) # torch.Size([128, 49])          
        bool_masked_pos_3 = bool_masked_pos_3.to(device, non_blocking=True).flatten(1).to(torch.bool) # torch.Size([128, 49])          
                  
        with torch.no_grad():
            feats_t, outputs_teacher = teacher(samples)
                
        outputs_S = F.log_softmax(outputs_student/T, dim=1)
        outputs_T = F.softmax(outputs_teacher/T, dim=1)
        
        ########################################################################################## Loss Criterion #######################################################################################
        if config.TRAIN.ACCUMULATION_STEPS > 1: 
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_S, outputs_T)*T*T*1000
            
            loss_feats = None
            # images.shape                                                        │  
            index = config.TRAIN.DISTILLATION.FEAT_INDEX
            # [B,CHW]
            X  = feats_s[index].flatten(1)
            # [B,B]
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            hsic_matrix[idx, 0] += HSIC(K, K, device) / num_steps 
                                   
            images_1 =  feats_s[index]
            images_2 =  feats_s[-2]
            images_3 =  feats_s[-3]
            
            Y = feats_t[index].flatten(1)
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
            
            
            hsic_matrix[idx, 1] += HSIC(K, L, device) / num_steps
            hsic_matrix[idx, 2] += HSIC(L, L, device) / num_steps
            
            target_images_1 = feats_t[index].detach()
            target_images_2 = feats_t[-2].detach()
            target_images_3 = feats_t[-3].detach()
            
            X_1  = feats_s[index].flatten(1)
            # [B,B]
            K_1 = X_1 @ X_1.t()
            K_1.fill_diagonal_(0.0)
            
            Y_1 = feats_t[index].flatten(1)
            L_1 = Y_1 @ Y_1.t()
            L_1.fill_diagonal_(0.0)
            
            X_2  = feats_s[index-1].flatten(1)
            # [B,B]
            K_2 = X_2 @ X_2.t()
            K_2.fill_diagonal_(0.0)
            
            Y_2 = feats_t[index-1].flatten(1)
            L_2 = Y_2 @ Y_2.t()
            L_2.fill_diagonal_(0.0)
            
            X_3  = feats_s[index-2].flatten(1)
            # [B,B]
            K_3 = X_3 @ X_3.t()
            K_3.fill_diagonal_(0.0)
            
            Y_3 = feats_t[index-2].flatten(1)
            L_3 = Y_3 @ Y_3.t()
            L_3.fill_diagonal_(0.0)
            
            mask_ratio_1 = max(1.0 - torch.sigmoid(torch.as_tensor(HSIC(K_1, L_1, device) / (np.sqrt(HSIC(K_1, K_1, device)) * np.sqrt(HSIC(L_1, L_1, device))))), 0.1)
            mask_ratio_2 = max(1.0 - torch.sigmoid(torch.as_tensor(HSIC(K_2, L_2, device) / (np.sqrt(HSIC(K_2, K_2, device)) * np.sqrt(HSIC(L_2, L_2, device))))), 0.1)
            mask_ratio_3 = max(1.0 - torch.sigmoid(torch.as_tensor(HSIC(K_3, L_3, device) / (np.sqrt(HSIC(K_3, K_3, device)) * np.sqrt(HSIC(L_3, L_3, device))))), 0.1)
            
            if config.MASKED_AUTOENCODER.TYPE == "dynamic":
                mask_ratio = mask_ratio_1 #1.0 - torch.sigmoid(torch.as_tensor(HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device))))) # the larger the similarity score, the smaller the teacher participant
            elif config.MASKED_AUTOENCODER.TYPE == "random":
                mask_ratio = torch.rand(1)
            elif config.MASKED_AUTOENCODER.TYPE == "step":
                mask_ratio = (1.0 - epoch / 200.0) * 0.95
            elif config.MASKED_AUTOENCODER.TYPE == 'others':
                mask_ratio = 0.5 + torch.sigmoid(-torch.as_tensor(HSIC(K, L, device) / (np.sqrt(HSIC(K, K, device)) * np.sqrt(HSIC(L, L, device)))))
            else:
                channel_mask_ratio = config.TRAIN.MASK_RATIO
                # images = images[:,torch.randperm(images.shape[1])]
           
            
            if config.MASKED_AUTOENCODER.TYPE != None:
                # Dynamic Masked Ratio
                mask_ratio.append(channel_mask_ratio)
                dynamic_ratio = channel_mask_ratio
                
                num_patches_1 = bool_masked_pos_1.shape[1] # 512 / 2048
                num_mask_1 = int(num_patches_1 * mask_ratio_1)
                if num_mask_1 <= 0:
                    num_mask_1 = 2
                bool_masked_pos_dynamic_1 = []
                for i in range(bool_masked_pos_1.shape[0]):
                    mask = np.hstack([ np.zeros(num_patches_1 - num_mask_1),np.ones(num_mask_1)])
                    np.random.shuffle(mask)
                    mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_1.append(mask)
                bool_masked_pos_1 = torch.stack(bool_masked_pos_dynamic_1)  
                
                num_patches_2 = bool_masked_pos_2.shape[1] # 512 / 2048
                num_mask_2 = int(num_patches_2 * mask_ratio_2)
                if num_mask_2 <= 0:
                    num_mask_2 = 2
                bool_masked_pos_dynamic_2 = []
                for i in range(bool_masked_pos_2.shape[0]):
                    mask = np.hstack([ np.zeros(num_patches_2 - num_mask_2),np.ones(num_mask_2)])
                    np.random.shuffle(mask)
                    mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_2.append(mask)
                bool_masked_pos_2 = torch.stack(bool_masked_pos_dynamic_2)
                
                num_patches_3 = bool_masked_pos_3.shape[1]  # 512 / 2048
                num_mask_3 = int(num_patches_3 * mask_ratio_3)
                if num_mask_3 <= 0:
                    num_mask_3 = 2
                bool_masked_pos_dynamic_3 = []
                for i in range(bool_masked_pos_3.shape[0]):
                    mask = np.hstack([ np.zeros(num_patches_3 - num_mask_3),np.ones(num_mask_3)])
                    np.random.shuffle(mask)
                    mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic_3.append(mask)
                bool_masked_pos_3 = torch.stack(bool_masked_pos_dynamic_3)   
                                
            if config.TRAIN.DISTILLATION.TYPE != 'logits':
                if config.TRAIN.DISTILLATION.TYPE == 'feat':                    
                    with torch.no_grad():
                        # calculate the predict label
                        unnorm_images_1 = target_images_1 
                        unnorm_images_2 = target_images_2
                        unnorm_images_3 = target_images_3
                        if normlize_target:
                            images_squeeze_1 = rearrange(unnorm_images_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_norm_1  = F.normalize(images_squeeze_1, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_1 = rearrange(images_norm_1, 'b n p c -> b n (p c)')
                            
                            images_squeeze_2 = rearrange(unnorm_images_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE*2, p2=config.TRAIN.PATCH_SIZE*2)
                            images_norm_2  = F.normalize(images_squeeze_2, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_2 = rearrange(images_norm_2, 'b n p c -> b n (p c)')
                            
                            images_squeeze_3 = rearrange(unnorm_images_3, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=config.TRAIN.PATCH_SIZE*4, p2=config.TRAIN.PATCH_SIZE*4)
                            images_norm_3  = F.normalize(images_squeeze_3, p=2, dim=-1)
                            # we find that the mean is about 0.48 and standard deviation is about 0.08.
                            images_patch_3 = rearrange(images_norm_3, 'b n p c -> b n (p c)')
                        else:
                            images_patch_1 = rearrange(unnorm_images_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE, p2=config.TRAIN.PATCH_SIZE)
                            images_patch_2 = rearrange(unnorm_images_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE*2, p2=config.TRAIN.PATCH_SIZE*2)
                            images_patch_3 = rearrange(unnorm_images_3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=config.TRAIN.PATCH_SIZE*4, p2=config.TRAIN.PATCH_SIZE*4)
                       
                        B, _, C = images_patch_1.shape # torch.Size([128, hw, p1p2c])
                    
                    # import pdb; pdb.set_trace() [b,196,4096]                       
                    outputs_1 = masked_autoencoder_1(images_1, target_images_1, bool_masked_pos_1)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    outputs_2 = masked_autoencoder_2(images_2, target_images_2, bool_masked_pos_2)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    outputs_3 = masked_autoencoder_3(images_3, target_images_3, bool_masked_pos_3)  # bool_masked_pos.shape ([128,49]) # torch.Size([128, 49, xxx])
                    
                    images_patch_1 = images_patch_1.detach()
                    images_patch_2 = images_patch_2.detach()
                    images_patch_3 = images_patch_3.detach()
                    loss_feats = loss_func(input=outputs_1, target=images_patch_1) + loss_func(input=outputs_2, target=images_patch_2) + loss_func(input=outputs_3, target=images_patch_3) * 0.6
                    
            
            if loss_feats is not None:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft + config.TRAIN.DISTILLATION.BETA * loss_feats
            else:
                loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft                        
                    
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                # import pdb 
                # pdb.set_trace()
                optimizer_s.step()
                optimizer_mae_1.step()
                optimizer_mae_2.step()
                optimizer_mae_3.step()
                
                optimizer_s.zero_grad()
                optimizer_mae_1.zero_grad()
                optimizer_mae_2.zero_grad()
                optimizer_mae_3.zero_grad()
                lr_scheduler_mae_1.step_update(epoch * num_steps + idx)
                lr_scheduler_mae_2.step_update(epoch * num_steps + idx)
                lr_scheduler_mae_3.step_update(epoch * num_steps + idx)
        else:
            loss_hard = criterion(outputs_student, targets)
            loss_soft = soft_criterion(outputs_student, outputs_teacher)
            
            loss = config.TRAIN.DISTILLATION.GAMMA * loss_hard + config.TRAIN.DISTILLATION.ALPHA * loss_soft  
            optimizer_s.zero_grad()                      
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.TRAIN.CLIP_GRAD)
                grad_norm_mae = torch.nn.utils.clip_grad_norm_(masked_autoencoder.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(student.parameters())
            
            optimizer_s.step()
            # lr_scheduler_s.step()

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        soft_loss_meter.update(loss_soft.item(), targets.size(0))
        hard_loss_meter.update(loss_hard.item(), targets.size(0))
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
                    f'loss_soft {soft_loss_meter.val:.4f} ({soft_loss_meter.avg:.4f})\t'
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
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                
                curr_step = epoch * len(data_loader) + idx + 1
    
    hsic_matrix = hsic_matrix[:,1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:,2].sqrt())
    assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"HSIC_MATRIX shape {hsic_matrix.shape}")
    return hsic_matrix, mask_ratio