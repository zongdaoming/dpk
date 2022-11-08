

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 256
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224


# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 12
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
# _C.MODEL.TYPE = 'teacher'
_C.RECTIFIER_DEPTH = 2
# TEACHER MODEL
_C.TEACHER_MODEL = CN()
# Model type
_C.TEACHER_MODEL.TYPE = 'resnet34'
# Model name
_C.TEACHER_MODEL.NAME = 'resnet34'
# Checkpoint to resume, could be overwritten by command line argument
_C.TEACHER_MODEL.RESUME = 'teacher/resnet34.pth'
# Number of classes, overwritten in data preparation
_C.TEACHER_MODEL.NUM_CLASSES = 1000
# Label Smoothing
_C.TEACHER_MODEL.LABEL_SMOOTHING = 0.
# Teacher output
_C.TEACHER_MODEL.OUTPUT = ''
# Teacher feature dims
_C.TEACHER_MODEL.NUM_DIMS = []

# STUDENT MODEL
_C.STUDENT_MODEL = CN()
# Model type
_C.STUDENT_MODEL.TYPE = 'resnet18'
# Model name
_C.STUDENT_MODEL.NAME = 'resnet18'
# Checkpoint to resume, could be overwritten by command line argument
_C.STUDENT_MODEL.RESUME = 'teacher/resnet18.pth'
# Number of classes
_C.STUDENT_MODEL.NUM_CLASSES = 1000
# Label Smoothing
_C.STUDENT_MODEL.LABEL_SMOOTHING = 0.
# Student output
_C.STUDENT_MODEL.OUTPUT = ''
# Student feature dims
_C.STUDENT_MODEL.NUM_DIMS = []


# Maksed Autoencoder
_C.MASKED_AUTOENCODER = CN()
_C.MASKED_AUTOENCODER.RESUME = 'None'
_C.MASKED_AUTOENCODER.TYPE = 'MAE'
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Important Parameters
_C.TRAIN.NAME = 'resnet' #VIT
_C.TRAIN.WINDOW_SIZE = (7,7)
_C.TRAIN.MASK_RATIO = 0.75
_C.TRAIN.PATCH_SIZE = 1
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 250
_C.TRAIN.WARMUP_EPOCHS=20
_C.TRAIN.COOLDOWN_EPOCHS=10
_C.TRAIN.PATIENCE_EPOCHS=10
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 1e-6
# _C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 1e-5
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.CONTINUE = False
_C.TRAIN.EARLYSTOP = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1


# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.95)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.99999



# Distillation
_C.TRAIN.DISTILLATION = CN()
# distillation type: feature/attn/patch embedding
_C.TRAIN.DISTILLATION.TYPE = 'feat'
# align
_C.TRAIN.DISTILLATION.ALIGN = 'L2'
# temperature for KD distillation
_C.TRAIN.DISTILLATION.TEMPERATURE = 1.0
# weight for classification
_C.TRAIN.DISTILLATION.GAMMA = 1.0
# weight balance for KD
_C.TRAIN.DISTILLATION.ALPHA = 1.0
# weight balance for other losses
_C.TRAIN.DISTILLATION.BETA = 0.3
# which layers to calculate KD loss
_C.TRAIN.DISTILLATION.FEAT_INDEX = -1
_C.CHANNEL_S = 'none'
_C.CHANNEL_MASK = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0

# _C.AUG.MIXUP = 0.8
# # Cutmix alpha, cutmix enabled if > 0
# _C.AUG.CUTMIX = 1.0
# # Cutmix min/max ratio, overrides alpha and enables cutmix if set
# _C.AUG.CUTMIX_MINMAX = None
# # Probability of performing mixup or cutmix when either/both is enabled
# _C.AUG.MIXUP_PROB = 1.0
# # Probability of switching to cutmix when both mixup and cutmix enabled
# _C.AUG.MIXUP_SWITCH_PROB = 0.5
# # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
# _C.AUG.MIXUP_MODE = 'batch'

# --------------
_C.AUG.MIXUP = 0.0
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 0.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = None
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = None
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = None
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = 'O0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 5
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()
    
def update_config(config, args):
    # update config from files first
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if args.img_size:
        config.DATA.IMG_SIZE= args.img_size
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.student_resume:
        config.STUDENT_MODEL.RESUME = args.student_resume
    if args.teacher_resume:
        config.TEACHER_MODEL.RESUME = args.teacher_resume
    if args.teacher_type:
        config.TEACHER_MODEL.TYPE = args.teacher_type
    if args.student_type:
        config.STUDENT_MODEL.TYPE = args.student_type
        
    if args.masked_autoencoder_resume != None:
        config.MASKED_AUTOENCODER.RESUME = args.masked_autoencoder_resume
    if args.masked_auto_encoder_type:
        config.MASKED_AUTOENCODER.TYPE = args.masked_auto_encoder_type 
    if args.kd_align:
        config.TRAIN.DISTILLATION.ALIGN = args.kd_align
    if args.mask_ratio:
        config.TRAIN.MASK_RATIO = args.mask_ratio
    if args.channel_s != None:
        config.CHANNEL_S = args.channel_s
    if args.channel_mask!= None:
        config.CHANNEL_MASK = args.channel_mask
    
    ########################################################################
    ########################################################################
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
        
    # distillation parameters
    if args.kd_type!=None:
        config.TRAIN.DISTILLATION.TYPE = args.kd_type
    if args.kd_align!=None:
        config.TRAIN.DISTILLATION.ALIGN = args.kd_align
    if args.kd_T!=None:
        config.TRAIN.DISTILLATION.TEMPERATURE = args.kd_T
    
    if args.gamma!=None:
        config.TRAIN.DISTILLATION.GAMMA = args.gamma
    if args.alpha!=None:
        config.TRAIN.DISTILLATION.ALPHA = args.alpha
    if args.beta!=None:
        config.TRAIN.DISTILLATION.BETA = args.beta
    if args.feat_index!=None:
        config.TRAIN.DISTILLATION.FEAT_INDEX = args.feat_index
    
    if args.continue_train:
        config.TRAIN.CONTINUE = True
    if args.earlystop:
        config.TRAIN.EARLYSTOP = True
    
    # ** set local rank for distributed training (is very important) **
    config.LOCAL_RANK = args.local_rank
    
    # output folder
    config.STUDENT_MODEL.OUTPUT = os.path.join(config.OUTPUT, config.STUDENT_MODEL.NAME, config.TAG)
    config.TEACHER_MODEL.OUTPUT = os.path.join(config.OUTPUT, config.TEACHER_MODEL.NAME, config.TAG)
    os.makedirs(config.TEACHER_MODEL.OUTPUT,exist_ok=True)
    os.makedirs(config.STUDENT_MODEL.OUTPUT,exist_ok=True)
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
