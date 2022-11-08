import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm import create_model
import numpy as np

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'pretrain_mae_base_patch16_224', 
    'pretrain_mae_large_patch16_224', 
]

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) 
            
        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 student_encoder_in_chans=3, 
                 teacher_encoder_in_chans=3,
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.patch = patch_size
        self.stu_encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=student_encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)
        self.tea_encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=teacher_encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)


        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.stu_encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)


        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # masked token is trainable parameters
        self.pos_embed = get_sinusoid_encoding_table(self.stu_encoder.patch_embed.num_patches, decoder_embed_dim)
        # trunc_normal_(self.mask_token, std=.02)
        assert self.stu_encoder.patch_embed.num_patches == self.tea_encoder.patch_embed.num_patches, "Check the patch embeddings part！"

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token', 'mask_token'}
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, y, mask):   
          
        x_vis = self.stu_encoder(x, mask) # [B, N_vis, 1024]
        
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, 192] :: x_vis.shape torch.Size([128, 13, 384])
        B, N, C = x_vis.shape
        
        y_vis = self.tea_encoder(y,~mask) #[b,N_vis,1024]
        
        y_vis = self.encoder_to_decoder(y_vis) # torch.Size([b, N_vis, 384])  
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach() # torch.Size([128, 49, 384]) 
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C) # torch.Size([128, 13, 384]) 
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C) # torch.Size([128, 36, 384]) 
        x_full = torch.zeros(size=(x.shape[0], x.shape[2]*x.shape[3]//self.patch//self.patch ,C)).to(x.device) #【b,196,192】
        
        index_x = (mask == False).nonzero() 
        index_y = (mask == True).nonzero()
        x_vis_new = x_vis + pos_emd_vis
        x_vis_new = x_vis_new.reshape(-1,C) 
        y_vis_new = y_vis + pos_emd_mask
        y_vis_new = y_vis_new.reshape(-1,C) 
        x_full[index_x[:,0],index_x[:,1],:] = x_vis_new
        x_full[index_y[:,0],index_y[:,1],:] = y_vis_new  
        x = self.decoder(x_full, 0) 
        return x 

@register_model
def resnet50_18_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_18_1_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, #512
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_18_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=2, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024*2*2,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet50_18_3(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=28,
        patch_size=4, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=512*4*4,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet101_18_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, #1024
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model  

@register_model
def resnet101_18_1_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet101_18_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=2, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024*2*2,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 


@register_model
def resnet101_18_3(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=28,
        patch_size=4, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=512, #1024 3.24 before 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=512*4*4,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet152_18_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, #1024
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 


@register_model
def resnet152_18_1_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, #1024
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet152_18_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=2, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, #1024
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024*2*2,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet152_18_3(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=28,
        patch_size=4, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=512, #1024
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=512*4*4,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 


@register_model
def convnext_large_resnet18(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=1536,        
        # encoder settings 
        encoder_embed_dim=640, #1024
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1536,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def convnext_large_resnet18_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=768,        
        # encoder settings 
        encoder_embed_dim=512, #1024
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=640,
        decoder_num_classes=768,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def pretrain_mae_base_patch4_28(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=28,
        patch_size=4, 
        student_encoder_in_chans=96,
        teacher_encoder_in_chans=384,        
        # encoder settings 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=384,
        decoder_num_classes=384*4*4,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def resnet34_mae_patch14_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 


 
@register_model
def resnet34_18_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet34_mae_patch14_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=2, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=256*2*2,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet34_mae_patch28_4(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=28,
        patch_size=4, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128*4,
        decoder_num_classes=128*4*4,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 
@register_model
def resnet34_18_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet110_32_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet110_32_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=32,
        teacher_encoder_in_chans=32,        
        # encoder settings 
        encoder_embed_dim=64, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=64,
        decoder_num_classes=32,
        decoder_depth=6,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet110_20_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=64, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 
@register_model
def resnet110_20_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=32,
        teacher_encoder_in_chans=32,        
        # encoder settings 
        encoder_embed_dim=64, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=32,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet56_20_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=80,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def resnet56_20_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=32,
        teacher_encoder_in_chans=32,        
        # encoder settings 
        encoder_embed_dim=64, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=48,
        decoder_num_classes=32,
        decoder_depth=6,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 
  

@register_model
def vgg13_8_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=4,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=320, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def vgg13_8_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=160, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=6,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model 


@register_model
def resnet32_shufflev2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=464,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet32_shufflev2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=232,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn40_shufflev1_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=960,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=320, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn40_shufflev1_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=480,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn40_16_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn40_16_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn40_40_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=80, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def wrn4_2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=160, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn4_2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=96, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn2_2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=160, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn2_2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=64,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vgg13_wrn16_2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=4,
        patch_size=1, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vgg13_wrn16_2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=160, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_wrn16_2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=4,
        patch_size=1, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_wrn16_2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=64,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def wrn40_40_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=32,
        teacher_encoder_in_chans=64,        
        # encoder settings 
        encoder_embed_dim=64, 
        encoder_depth=6, 
        encoder_num_heads=4,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=80,
        decoder_num_classes=64,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_vgg_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=4,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1536,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_vgg_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_mobile_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=4,
        patch_size=1, 
        student_encoder_in_chans=160,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1536,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def resnet50_mobile_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=48,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=640,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vgg13_mobile_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=4,
        patch_size=1, 
        student_encoder_in_chans=160,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=320, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def vgg13_mobile_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=48,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=6, 
        encoder_num_heads=4,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet32_shufflev1_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=960,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet32_shufflev1_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=480,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=256, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet32_8_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=8,
        patch_size=1, 
        student_encoder_in_chans=256,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=180, 
        encoder_depth=6, 
        encoder_num_heads=4,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet32_8_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=16,
        patch_size=1, 
        student_encoder_in_chans=128,
        teacher_encoder_in_chans=128,        
        # encoder settings 
        encoder_embed_dim=160, 
        encoder_depth=6, 
        encoder_num_heads=4,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=128,
        decoder_num_classes=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def resnet_mobilev2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=1280,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet_mobilev3(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=432,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1024,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet18_shufflenetv2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=1024,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet18_shufflenetv2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=96,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=128, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet50_shufflenetv2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=1024,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=6, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1280,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet34_mobilev2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=1024,
        teacher_encoder_in_chans=512,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=512,
        decoder_num_classes=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet34_mobilev2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=256,        
        # encoder settings 
        encoder_embed_dim=320, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=256,
        decoder_num_classes=256,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet_mobilev2_1(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=7,
        patch_size=1, 
        student_encoder_in_chans=1024,
        teacher_encoder_in_chans=2048,        
        # encoder settings 
        encoder_embed_dim=640, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=1536,
        decoder_num_classes=2048,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def resnet_mobilev2_2(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=14,
        patch_size=1, 
        student_encoder_in_chans=512,
        teacher_encoder_in_chans=1024,        
        # encoder settings 
        encoder_embed_dim=512, 
        encoder_depth=4, 
        encoder_num_heads=8,
        encoder_num_classes=0,
        # decoder settings
        decoder_embed_dim=640,
        decoder_num_classes=1024,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

