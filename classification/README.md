# Maskfeat

### Installation

Environments:

- Python 3.6
- PyTorch 1.10.1
- torchvision 0.11.2

Install the package:

```
pip install -r requirements.txt
```
### Getting started
1. Download all checkpoints from model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
- please download the checkpoints to './teacher'
- If test the models on ImageNet, please download the dataset at <https://image-net.org/> and put them to `./data/images/`

### 
2. Training on CIFAR-100
```bash
./scripts/mutil_cifar.sh $PARTITION $GPUS configs/2022_03_24/2022_03_24_resnet50_mobilenetv2_last_two.yaml

or change our code to DataParallel
```

3. Training on ImageNet
```bash
# multi stage distillation
./scripts/mutil_stage.sh $PARTITION $GPUS configs/2022_03_17/2022_03_17_resnet34_last_two_cka.yaml

# last one stage distillation
./scripts/resne_kd.sh $PARTITION $GPUS $configs

```

### Custom Distillation Method
1. register model at `modeling_pretrain.py`, like below:
  ```python
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
  ```
2. create a yaml file at `configs/`, like `configs/2022_03_24/2022_03_24_resnet50_mobilenetv2_last_two.yaml`.

# License
# Acknowledgement