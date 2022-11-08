
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# from utils.log_helper import default_logger as logger
 
def load_mutil_mae_checkpoint(config, path, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MASKED_AUTOENCODER.RESUME}....................")
    if config.MASKED_AUTOENCODER.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        if config.TRAIN.CONTINUE:
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        else:
            config.TRAIN.START_EPOCH = 0
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.STUDENT_MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy
def load_mae_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MASKED_AUTOENCODER.RESUME}....................")
    if config.MASKED_AUTOENCODER.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MASKED_AUTOENCODER.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MASKED_AUTOENCODER.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        if config.TRAIN.CONTINUE:
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        else:
            config.TRAIN.START_EPOCH = 0
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.STUDENT_MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy
def load_vit_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.STUDENT_MODEL.RESUME}....................")
    if config.TRAIN.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.STUDENT_MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        print('from load :',config.TRAIN.RESUME)
        checkpoint = torch.load(config.TRAIN.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        if config.TRAIN.CONTINUE:
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        else:
            config.TRAIN.START_EPOCH = 0
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.TRAIN.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy
 
def load_student_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.STUDENT_MODEL.RESUME}....................")
    if config.STUDENT_MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.STUDENT_MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.STUDENT_MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        if config.TRAIN.CONTINUE:
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        else:
            config.TRAIN.START_EPOCH = 0
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.STUDENT_MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_teacher_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.TEACHER_MODEL.RESUME}....................")
    if config.TEACHER_MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.TEACHER_MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.TEACHER_MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        if config.TRAIN.CONTINUE:
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        else:
            config.TRAIN.START_EPOCH = 0
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.TEACHER_MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def mask1_save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask1_best.pth')
    # save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'ckpt_epoch_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def mask2_save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask2_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    
def mask3_save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask3_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
def mask_save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    
def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'student_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
def save_mask_ratio(config, epoch, mask_ratio, logger):
    # logger.info(f"hsic_matrix {hsic_matrix}")
    save_state = {'mask_ratio_matrix': mask_ratio}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask_ratio_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def save_cka(config, epoch, hsic_matrix, logger):
    # logger.info(f"hsic_matrix {hsic_matrix}")
    save_state = {'hsic_matrix': hsic_matrix}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'cka_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def L2(f_):
    return (((f_**2).sum(1))**0.5).reshape(f_.shape[0], 1, f_.shape[2]) + 1e-8

def similarity(feat):
    feat = feat.permute(0, 2, 1).float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    return torch.einsum('icm, icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    '''
    input shape:
        f_S: B * C * L
        f_T: B * C * L
    '''
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/(f_T.shape[1]**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


def loss_kl_neck_single(neck_st, neck_th, size_average = True, norm = 'softmax'):
        # crit = nn.KLDivLoss(reduction ='mean')
        relu = nn.ReLU()
        s = relu(neck_st)
        t = relu(neck_th)

        if norm == 'softmax':
            s = F.log_softmax(s, dim=1)
            t = F.softmax(t, dim=1)
            t.detach_()
            #loss = crit(s, t)
            loss = t * (t.log()-s)
            loss = loss.mean()
        elif norm == 'l2':
            loss = torch.sum(t/torch.sum(t) * (torch.log((t/torch.sum(t)+1e-6)/(s/torch.sum(s)+1e-6))))
        else:
            assert False
            return None
        return loss
def mask1_save_last_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(), 
                  'lr_scheduler': lr_scheduler.state_dict(), 
                  'epoch': epoch, 
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask1_last_ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!") 

def mask2_save_last_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(), 
                  'lr_scheduler': lr_scheduler.state_dict(), 
                  'epoch': epoch, 
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask2_last_ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!") 
    
def mask3_save_last_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(), 
                  'lr_scheduler': lr_scheduler.state_dict(), 
                  'epoch': epoch, 
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask3_last_ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    
def mask_save_last_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(), 
                  'lr_scheduler': lr_scheduler.state_dict(), 
                  'epoch': epoch, 
                  'config': config}
    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'mask_last_ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")   

def save_last_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(), 
                  'lr_scheduler': lr_scheduler.state_dict(), 
                  'epoch': epoch, 
                  'config': config}

    save_path = os.path.join(config.STUDENT_MODEL.OUTPUT, f'student_last_ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")    