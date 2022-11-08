# Maskfeat
## Install MMDetection and MS COCO2017
  - Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection and make sure you can run it successfully.
  - This repo uses mmdet==2.11.0 and mmcv-full==1.2.4
  - If you want to use higher mmdet version, you may have to change the optimizer in apis/train.py and build_detector in tools/train.py.
  - For mmdet>=2.12.0, if you want to use inheriting strategy, you have to initalize the student with teacher's parameters after model.init_weights().
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmdetectin's codes.
  - Add the mmdet/distillation/. in our codes to the mmdet/ in mmdetectin's codes.
  - Replace the mmdet/apis/train.py and tools/train.py in mmdetection's codes with mmdet/apis/train.py and tools/train.py in our codes.
  - Add pth_transfer.py to mmdetection's codes.
  - Unzip COCO dataset into data/coco/
## Train

```
#single GPU
python tools/train_kd.py configs/distillers/Maskfeat/Maskfeat_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py

#multi GPU
bash tools/dist_train.sh configs/distillers/Maskfeat/Maskfeat_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py 8
```

## Transfer
```
# Tansfer the Maskfeat model into mmdet model
python pth_transfer.py --p $fgd_ckpt --out $new_mmdet_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/retinanet/retinanet_r50_fpn_2x_coco.py $new_mmdet_ckpt --eval bbox

#multi GPU
bash tools/dist_test.sh configs/retinanet/retinanet_r50_fpn_2x_coco.py $new_mmdet_ckpt 8 --eval bbox
```

## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).

Thanks to the work [FGD](https://github.com/yzd-v/FGD) and [mmetection-distiller](https://github.com/pppppM/mmdetection-distiller).