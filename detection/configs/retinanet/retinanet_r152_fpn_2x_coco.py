_base_ = './retinanet_r50_fpn_2x_coco.py'
model = dict(pretrained='torchvision://resnet152', backbone=dict(depth=152))
