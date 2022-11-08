import os
import time
import random
import argparse
import datetime
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import optim as optim
import torch.nn.functional as F
from timm.utils import accuracy, AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import yaml
import utils.misc_n as utils
from resnet_config import get_config
from models.builder import MODELS
import modeling_finetune, modeling_pretrain, modeling_pretrain_tokens
from timm.models import create_model
from timm.utils.model import unwrap_model
from data import build_loader
# from logger import create_logger
from utils.log_helper import default_logger as logger
from optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from engine import validate, train_resnet_epoch, train_resnet_one_epoch
from util import load_vit_checkpoint, load_teacher_checkpoint, load_student_checkpoint, load_mae_checkpoint, save_checkpoint, save_cka, auto_resume_helper, mask_save_last_checkpoint, mask_save_checkpoint, save_mask_ratio
from util import sim_dis_compute,loss_kl_neck_single,save_last_checkpoint
from timm.models import create_model
from torch.optim import lr_scheduler
from resnet_engine import train, resnet_epoch
# from tensorboardX import SummaryWriter

def main(args):
    logger.info(f'Torch Version: {torch.__version__}')
    logger.info(f'Cuda Is Available: {torch.cuda.is_available()}')
    logger.info(f'Cuda number: {torch.cuda.device_count()}')
    logger.info(f'GPU Version: {torch.cuda.get_device_name()}')
    with open(args.config) as f:
        config = yaml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    utils.init_distributed_mode(args)
    # logger.info("git:\n  {}\n".format(utils.get_sha()))
    args.output = os.path.join('results',args.config.split('/')[-1].split('.')[0].upper())
    os.makedirs(args.output,exist_ok=True)
    utils.print_args(args)    
    # fix the seed for reproducibility
    device = torch.device(args.device)
    normlize_target = args.normlize_target    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Define Your Model
    config = get_config(args)
    print(config) 
    lr = 0.1
    step_size = 25
    gamm = 0.5
    teacher = create_model(config.TEACHER_MODEL.TYPE)
    
    student = create_model(config.STUDENT_MODEL.TYPE)
    
    t_state_dict = torch.load(config.TEACHER_MODEL.RESUME)
    
    f = t_state_dict.get('model',False)
    if f == False:
        state = teacher.load_state_dict(t_state_dict, strict=False)
    else:
        state = teacher.load_state_dict(t_state_dict['model'], strict=False) 
         
    print("teacher load state:", state)
    
    if config.TEACHER_MODEL.TYPE in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']:
        lr = 0.02
        step_size = 25
        gamm = 0.5
    optimizer_s = None    
    
    s_state_dict = torch.load(config.STUDENT_MODEL.RESUME)
    f = s_state_dict.get('model',False)
    if f == False:
        state = student.load_state_dict(s_state_dict, strict=False)
    else:
        state = student.load_state_dict(s_state_dict['model'], strict=False)
    print('student state:', state)
    
    masked_autoencoder = None
    
    if config.TRAIN.DISTILLATION.TYPE != 'logits' and config.TRAIN.DISTILLATION.TYPE != 'fitnet':
       masked_autoencoder = create_model(args.model)
       if config.MASKED_AUTOENCODER.RESUME != 'None':
           mask_state_dict = torch.load(config.MASKED_AUTOENCODER.RESUME)
           masked_autoencoder.load_state_dict(mask_state_dict['model'], strict=False)
           logger.info(f"mask network load from {config.MASKED_AUTOENCODER.RESUME}")
       print(masked_autoencoder)
       masked_autoencoder.cuda()
    teacher.cuda()
    student.cuda()
    
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    if optimizer_s == None:
        epochs = 200
        optimizer_s = optim.SGD(params = student.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        lr_scheduler_s = lr_scheduler.StepLR(optimizer_s, step_size=step_size, gamma=gamm)
    if config.TRAIN.DISTILLATION.TYPE == 'logits' or config.TRAIN.DISTILLATION.TYPE == 'at':
        epochs = 100
        optimizer_s = optim.SGD(params = student.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        lr_scheduler_s = lr_scheduler.MultiStepLR(optimizer_s, milestones=[30, 60, 90], gamma=0.1)
    print(epochs)
    
    student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
    teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False)
    
    if masked_autoencoder != None:
        masked_autoencoder = torch.nn.parallel.DistributedDataParallel(masked_autoencoder, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        optimizer_mae = build_optimizer(config, masked_autoencoder)
        lr_scheduler_mae = build_scheduler(config, optimizer_mae, len(data_loader_train))#len(data_loader_train))
        masked_autoencoder_without_ddp = masked_autoencoder.module
        if config.MASKED_AUTOENCODER.RESUME != 'None':
            state = load_mae_checkpoint(config, masked_autoencoder_without_ddp, optimizer_mae, lr_scheduler_mae, logger)
            print('mae state:', state)
    student_without_ddp = student.module
    teacher_without_ddp = teacher.module
    
    n_parameters_student = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_parameters_teacher = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    if masked_autoencoder != None:
        n_parameters_masked = sum(p.numel() for p in masked_autoencoder.parameters() if p.requires_grad)
        logger.info("masked_autoencoder of params: {} M".format(n_parameters_masked/1e6))
   
    logger.info("student of params: {} M".format(n_parameters_student/1e6))
  
    logger.info("teacher of params: {} M".format(n_parameters_teacher/1e6))
    
    logger.info(f"Distillation Type: {config.TRAIN.DISTILLATION.TYPE}")

    if config.AUG.MIXUP > 0.0:
        criterion = SoftTargetCrossEntropy()
    elif config.STUDENT_MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.STUDENT_MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    soft_criterion = torch.nn.KLDivLoss() 
    
    feat_criterion = None
    if config.TRAIN.DISTILLATION.ALIGN == 'similarity':
        feat_criterion = sim_dis_compute
    elif config.TRAIN.DISTILLATION.ALIGN == 'L2':
        feat_criterion = torch.nn.MSELoss()
    elif config.TRAIN.DISTILLATION.ALIGN == 'KL':
        feat_criterion = torch.nn.KLDivLoss()
    elif config.TRAIN.DISTILLATION.ALIGN == 'gang':
        feat_criterion = loss_kl_neck_single
    elif config.TRAIN.DISTILLATION.ALIGN == 'regressor':
        feat_criterion = torch.nn.MSELoss()
    
    if dist.get_rank() == 0:
        pass
    #     tb_logger = SummaryWriter(os.path.join(config.STUDENT_MODEL.OUTPUT, 'tb_log'))
        tb_logger= None
        logger.info(f"Tensorboard Launched!")
    else:
        tb_logger = None 
         
    ori_accuracy = 0.0 

    T = config.TRAIN.DISTILLATION.TEMPERATURE # Temperature
    
    logger.info(f"====>Channel align: {config.CHANNEL_S}")
    logger.info(f"====>Distillation type: {config.TRAIN.DISTILLATION.TYPE}")
    logger.info(f"====>Distillation align: {config.TRAIN.DISTILLATION.ALIGN}")
    logger.info(f"====>KD temperature: {config.TRAIN.DISTILLATION.TEMPERATURE}")
    logger.info(f"====>Classification loss weight: {config.TRAIN.DISTILLATION.GAMMA}")
    logger.info(f"====>KD logits loss weight: {config.TRAIN.DISTILLATION.ALPHA}")
    logger.info(f"====>KD features loss weight: {config.TRAIN.DISTILLATION.BETA}")
    logger.info(f"====>Mask Ratio: {config.TRAIN.MASK_RATIO}")
    logger.info(f"====>Mask type: {config.MASKED_AUTOENCODER.TYPE}")
    logger.info(f"====>Early stop distillation: {config.TRAIN.EARLYSTOP}")
    logger.info("==============================================================")
    logger.info("Start Training!")
    
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(config.TRAIN.START_EPOCH, epochs):
        data_loader_train.sampler.set_epoch(epoch)
        if config.TRAIN.DISTILLATION.TYPE == 'logits' or config.TRAIN.DISTILLATION.TYPE == 'fitnet':
            train_resnet_epoch(config, teacher, student, device, normlize_target, criterion, soft_criterion, data_loader_train, optimizer_s, epoch, mixup_fn, lr_scheduler_s, T, tb_logger)
        else:
            hsic_matrix, mask_ratio = train_resnet_one_epoch(config, teacher, student, masked_autoencoder, device, normlize_target, 
                            criterion, soft_criterion, feat_criterion, data_loader_train, 
                            optimizer_s, optimizer_mae, epoch, mixup_fn, lr_scheduler_s, lr_scheduler_mae, T, tb_logger)
            save_cka(config, epoch, hsic_matrix, logger)
            if len(mask_ratio):
                save_mask_ratio(config, epoch, mask_ratio, logger)
        lr_scheduler_s.step()  
        acc1, acc3, acc5, loss = validate(config, data_loader_val, student)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_last_checkpoint(config, epoch, student_without_ddp, acc1, optimizer_s, lr_scheduler_s, logger)
            if masked_autoencoder != None:
                mask_save_last_checkpoint(config, epoch, masked_autoencoder_without_ddp, acc1, optimizer_mae, lr_scheduler_mae, logger)
        if dist.get_rank() == 0 and (acc1 > max_accuracy):
            save_checkpoint(config, epoch, student_without_ddp, acc1, optimizer_s, lr_scheduler_s, logger)  
            if masked_autoencoder != None: 
                mask_save_checkpoint(config, epoch, masked_autoencoder_without_ddp, acc1, optimizer_mae, lr_scheduler_mae, logger)
                     
        max_accuracy = max(max_accuracy, acc1)
        
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}")
        logger.info(f"Max accuracy: {max_accuracy:.3f}")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test', add_help=False)
    parser.add_argument('--config', type=str, required=True, metavar="FILE", help='path to config file', )    
    args = parser.parse_args() 
    main(args)