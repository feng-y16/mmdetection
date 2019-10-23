from __future__ import division
import re
from collections import OrderedDict
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
from mmdet.apis.inference import LoadImage
from mmdet import datasets
from mmdet.datasets.pipelines import Compose
from mmdet.core import (CocoDistEvalmAPHook, CocoDistEvalRecallHook,
                        DistEvalmAPHook, DistOptimizerHook, Fp16OptimizerHook)
from mmdet.datasets import DATASETS, build_dataloader
from mmdet.models import RPN
from .env import get_root_logger
import pdb
from tools.attack import visualize_img, load_model
import datetime
import numpy as np
from tqdm import tqdm
import os
from skimage import transform
from tqdm import *


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in dataset
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = DATASETS.get(val_dataset_cfg.type)
            if issubclass(dataset_type, datasets.CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def attack_detector(args, model, cfg, dataset):
    cfg.data.workers_per_gpu = 0
    cfg.data.imgs_per_gpu = 1
    infer_model = load_model(args)
    attack_loader = build_dataloader(dataset, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, cfg.gpus, dist=False)
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    if args.clear_output:
        file_list = os.listdir(args.save_path)
        for f in file_list:
            os.remove(os.path.join(args.save_path, f))
    max_batch = min(attack_loader.__len__(), args.max_attack_batches)
    pbar_outer = tqdm(total=max_batch)
    pbar_inner = tqdm(total=args.num_attack_iter)
    acc_before_attack = 0
    acc_under_attack = 0
    for i, data in enumerate(attack_loader):
        if i >= max_batch:
            break
        imgs = data['img'].data[0].cuda()
        raw_imgs = data['img'].data[0]
        raw_filename = data['img_meta'].data[0][0]['filename']
        imgs_mean = data['img_meta'].data[0][0]['img_norm_cfg']['mean']
        imgs_std = data['img_meta'].data[0][0]['img_norm_cfg']['std']
        pbar_inner.reset()
        acc_list = []
        for _ in range(args.num_attack_iter):
            imgs = imgs.detach()
            imgs.requires_grad = True
            result = model(imgs, data['img_meta'], return_loss=True,
                           gt_bboxes=data['gt_bboxes'], gt_labels=data['gt_labels'])
            pdb.set_trace()
            keys = list(result.keys())
            acc_list.append(result['acc'])
            keys.remove('acc')
            for key in keys:
                if type(result[key]) is list:
                    for loss in result[key]:
                        loss.backward(retain_graph=True)
                else:
                    result[key].backward(retain_graph=True)
            imgs = imgs + args.epsilon / args.num_attack_iter * imgs.grad / torch.max(torch.abs(imgs.grad))
            result[keys[0]][0].backward()
            model.zero_grad()
            pbar_inner.update(1)
        acc_before_attack += acc_list[0]
        acc_under_attack += acc_list[-1]
        imgs = imgs.detach().cpu().numpy()[0]
        raw_imgs = raw_imgs.numpy()[0]
        for k in range(0, 3):
            imgs[k] = imgs[k] * imgs_std[k] + imgs_mean[k]
            raw_imgs[k] = raw_imgs[k] * imgs_std[k] + imgs_mean[k]
        imgs = imgs[[2, 1, 0]]
        raw_imgs = raw_imgs[[2, 1, 0]]
        imgs = imgs.transpose(1, 2, 0)
        raw_imgs = raw_imgs.transpose(1, 2, 0)
        (_, filename) = os.path.split(raw_filename)
        visualize_img(infer_model, raw_imgs, args.save_path + filename)
        visualize_img(infer_model, imgs, args.save_path + 'attack_' + filename)
        pbar_outer.update(1)
    pbar_outer.close()
    pbar_inner.close()
    acc_before_attack /= max_batch
    acc_under_attack /= max_batch
    print("Accuracy before attack = %g" % acc_before_attack)
    print("Accuracy under attack = %g" % acc_under_attack)
    print("Accuracy decrease = %g" % (acc_before_attack - acc_under_attack))
