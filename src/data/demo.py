import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data


### [y]
from utility import log_initialize


class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        ### [y]
        print("run Demo's __init__()")
        #srdata_log_dp = "/home/v5/yh/Eclipse_ws/edsr/study_edsr/experiment/yh_gen_log"
        #self.demo_log = log_initialize("Demo-py", srdata_log_dp)
        
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        
        ### [y]
        #self.demo_log.debug("demo-py, filename={0}".format(filename))
        
        lr = imageio.imread(self.filelist[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

