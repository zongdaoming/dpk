import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector, build_backbone
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from einops import rearrange

@DISTILLER.register_module()
class MAE_Distiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 mae_cfg, 
                 index=[2,3],
                 where=False,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=True):

        super(MAE_Distiller, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.index = index[::-1]
        self.mae = []
        self.mae_cfg = mae_cfg
        for mae in mae_cfg.mae_cfg:
            self.mae.append(build_backbone(mae).cuda())
        self.init_weights_teacher(teacher_pretrained)
        self.where = where
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses]+self.mae)


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
        print('teacher load state:', checkpoint)

    def HSIC(self, K, L, device):
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
    
    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
       

        with torch.no_grad():
            self.teacher.eval()
            feat_t = self.teacher.extract_feat(img)
           
        student_loss = self.student.forward_train(img, img_metas, **kwargs)
        
        feat_s = self.student.extract_feat(img) # feat从大到小排
        device = 'cuda'
        bool_masked_pos = []
        for (index, mae_cfg) in zip(self.index, self.mae_cfg.mae_cfg):
            X  = feat_t[index].flatten(1)
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            Y = feat_s[index].flatten(1)
            L = Y @ Y.t()
            L.fill_diagonal_(0.0)
            x_a = np.abs(self.HSIC(K, L, device))+ 1e-5
            x_b = np.abs(self.HSIC(K, K, device)) + 1e-5
            x_c = np.abs(self.HSIC(L, L, device)) + 1e-5
            
            p = mae_cfg.patch_size
            t =  np.sqrt(x_b * x_c)
            dynamic_ratio = 1.0 - x_a / t
            if np.isnan(dynamic_ratio) or np.isinf(dynamic_ratio):
                dynamic_ratio = 0.5
            if dynamic_ratio >= 1.0:
                dynamic_ratio = 0.85
            if dynamic_ratio <= 0:
                dynamic_ratio = 0.15
            
            num_patches = (feat_t[index].shape[2]*feat_t[index].shape[3]) // (p*p)
            num_mask = int(num_patches * dynamic_ratio)
            num_mask = max(num_mask, 40)
            
            if self.where:
                target_images = rearrange(feat_t[index], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p) # [b,h*w,c*p*p]
                tea_att_score = torch.sum(target_images, dim=-1) # sum up along the channel dim || y_feats_sum.shape  [b,h*w]
                ids_shuffle = torch.argsort(tea_att_score, descending=True, dim=1)
                ids_restore = torch.argsort(ids_shuffle, descending=True, dim=1) 
                ids_keep = ids_shuffle[:, :num_mask]
                mask = torch.zeros([tea_att_score.shape[0], tea_att_score.shape[-1]], device=target_images.device)
                # generate the binary mask: 1 is keep, 0 is remove
                mask[:, :num_mask] = 1
                bool_masked_pos.append(torch.gather(mask, dim=1, index=ids_restore).to(torch.bool))
                
            else: 
                bool_masked_pos_dynamic = []
                for i in range(feat_t[index].shape[0]):
                    mask = np.hstack([np.zeros(num_patches - num_mask),np.ones(num_mask)])
                    np.random.shuffle(mask)
                    mask = torch.as_tensor(mask, dtype=torch.bool).to(device)
                    bool_masked_pos_dynamic.append(mask)
                bool_masked_pos.append(torch.stack(bool_masked_pos_dynamic))   
        buffer_dict = dict(self.named_buffers())
        i = 0
        for item_loc in self.distill_cfg:
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]
            if i < len(self.index):
                out = self.mae[i](student_feat, teacher_feat, bool_masked_pos[i])
            else:
                out = student_feat
            i += 1
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                
                student_loss[loss_name] = self.distill_losses[loss_name](out, teacher_feat, kwargs['gt_bboxes'], img_metas)
        
        
        return student_loss
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)