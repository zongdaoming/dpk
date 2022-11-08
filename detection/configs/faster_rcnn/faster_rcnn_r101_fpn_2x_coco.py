_base_ = './faster_rcnn_r50_fpn_2x_coco.py'
model = dict(pretrained='ttorchvision://resnet101', backbone=dict(depth=101))
