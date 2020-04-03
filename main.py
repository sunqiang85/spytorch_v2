# from datasets.CIFAR10 import CIFAR10
import torch
from cfgs.base_cfgs import Cfgs
import argparse
import yaml
from datasets.CIFAR10 import get_dataloader
from exec import Engine
from sutils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="cfgs/yaml/cifar10_v1.yaml")
    parser.add_argument('--mode', type=str, default="train", help="train, test")
    parser.add_argument('--init', type=str, default="scratch", help="scratch, pretrained, resume")
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    __C = Cfgs()

    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    with open(args.conf, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    logger = Logger(__C)
    logger.filelogger.info('Hyper Parameters:')
    logger.filelogger.info(str(__C))

    # data
    if __C.mode == 'test':
        train_loader = None
    else:
        train_loader = get_dataloader(__C, 'train')

    if __C.mode == 'train':
        val_loader = get_dataloader(__C, 'valid')
    elif __C.mode == 'test':
        val_loader = get_dataloader(__C, 'test')

    # train or validation
    execution = Engine(__C, logger)
    execution.run(train_loader, val_loader)


