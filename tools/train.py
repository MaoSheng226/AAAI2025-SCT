# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # print("________________________________________________begin_____________________________")
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume
    # print("______________________runnner_begin____________")
    # build the runner from config

    """根据源码分析来看，后续的构建整个网络所需的dataloder、model等等都是通过构建Runner对象开始的。from_cfg函数是直接调用cls来获取config文件里面的参数
    从而得到对象，并在后续进行初始化，也就是说调用该函数可以生成一个Runner对象（利用Runner的_init_的初始化函数）
    但是在构建一个并行的对象时却一直卡在了模型构建阶段，因为不论是不是分布式还是单个GPU，都会调用wrap_model函数来初始化model对象
    根据distributed属性来判断是不是并行的模型。launcher参数指定了是否需要分布式训练，只要不是None就是将distributed属性改为True即
    是分布式的模型。
    但是当构建分布式的模型的时候，卡在了以下代码当中
    model = MMDistributedDataParallel(
                module=model,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
                
    可能出现的问题：
    1、broadcas_buffers、find_unused_parameters两个参数的设置，是不是对的？
    一个是控制广播，即将一个GPU得到的参数是否覆盖，还有就是对于冻住的参数是不是不进行反向传播
    2、多线程的问题，个人认为这个问题是最有可能的。很可能出现问题的就是该问题，有可能出现多线程的线程互斥造成的死锁
    还有可能是多线程导致的资源占用，从而导致两个线程之间出现共同资源的抢占的情况。
    
    """
    if 'runner_type' not in cfg:
        # build the default runner
        # print("__________runner_type___________")
        runner = Runner.from_cfg(cfg)
        # print("__________end__________")
        # print(runner)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    print("___________________________next_________________")
    print(runner)
    # start training
    runner.train()


if __name__ == '__main__':
    main()
