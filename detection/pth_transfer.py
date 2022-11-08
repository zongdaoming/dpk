# -*- coding: utf-8 -*-

import torch
import argparse
from collections import OrderedDict

def change_model(args):
    fgd_model = torch.load(args.p)
    all_name = []
    for name, v in fgd_model["state_dict"].items():
        if name.startswith("student."):
            all_name.append((name[8:], v))
        else:
            continue
    state_dict = OrderedDict(all_name)
    fgd_model['state_dict'] = state_dict
    torch.save(fgd_model, args.out) 
    print("transfer is done!!!!!")
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('--p', type=str, default='work_dirs/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco/epoch_24.pth', 
                        metavar='N',help='fgd_model path')
    parser.add_argument('--out', type=str, default='retina_res50_new.pth',metavar='N', 
                        help = 'pair path')
    args = parser.parse_args()
    change_model(args)
