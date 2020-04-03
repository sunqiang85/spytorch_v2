# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.path_cfgs import PATH
# from path_cfgs import PATH

import os, torch, random
import numpy as np
from types import MethodType
import os


class Cfgs(PATH):
    def __init__(self):
        super(Cfgs, self).__init__()

        # Set Devices
        # If use multi-gpu training, set e.g.'0, 1, 2' instead
        self.gpu = '2'

        # Set RNG For CPU And GPUs
        self.seed = 2020

        # Dataset
        self.batch_size = 512
        self.num_workers = 2
        self.pin_memory = True

        # Train
        self.max_epoch = 10




    @property
    def ckpt_path(self):
        return "{}/model.pth_best".format(self.exp_dir)

    @property
    def log_path(self):
        return "{}/{}_log.txt".format(self.exp_dir, self.mode)

    @property
    def tensorboard_path(self):
        return "{}/tensorboard".format(self.exp_dir)


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


    def proc(self):
        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.n_gpu = len(self.gpu.split(','))
        self.devices = [_ for _ in range(self.n_gpu)]
        torch.set_num_threads(2)

        self.init_seed()


        if not os.path.exists(self.exp_dir):
            os.mkdir(self.EXP_DIR)


    def init_seed(self):
        # ------------ Seed setup
        # fix pytorch seed
        torch.manual_seed(self.seed)
        if self.n_gpu < 2:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.seed)

        # fix random seed
        random.seed(self.seed)

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''

#
#
# if __name__ == '__main__':
#     __C = Cfgs()
#     __C.proc()





