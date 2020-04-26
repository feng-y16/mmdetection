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
from tools.attack import load_model, visualize_all_images, visualize_all_images_plus_acc, visualize_modification, \
    generate_data
import datetime
import numpy as np
from tqdm import tqdm
import os
from skimage import transform
import copy
import threading
from datetime import datetime
import scipy.stats as st
import shutil
from mmdet.core import eval_map_attack
import sys
sys.path.append('../')
sys.path.append('../yolov3')
from yolov3.test import test_attack


class ThreadingWithResult(threading.Thread):

    def __init__(self, func, args=()):
        super(ThreadingWithResult, self).__init__()
        self.func = func
        self.args = args
        self.result = [-1 * np.ones(6), None]

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


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


def conv_layer(args):
    assert (args.kernel_size - 1) % 2 == 0
    if args.kernel == 'Gaussian':
        x = np.linspace(-3, 3, args.kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        weight = np.stack([kernel, kernel, kernel])
        weight = torch.Tensor(weight).unsqueeze(0).cuda()
    elif args.kernel == 'Linear':
        x = np.linspace(0, 1, (args.kernel_size + 1) // 2)
        kern1d = np.concatenate((x, x[np.size(x) - 2::-1]))
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        weight = np.stack([kernel, kernel, kernel])
        weight = torch.Tensor(weight).unsqueeze(0).cuda()
    elif args.kernel == 'Uniform':
        kern1d = np.ones(args.kernel_size)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        weight = np.stack([kernel, kernel, kernel])
        weight = torch.Tensor(weight).unsqueeze(0).cuda()

    def conv(input_data):
        return torch.nn.functional.conv2d(input_data, weight, bias=None, stride=1, padding=(args.kernel_size - 1) // 2)
    return conv


def attack_detector(args, model, cfg, dataset):
    print(str(datetime.now()) + ' - INFO - GPUs: ', cfg.gpus)
    print(str(datetime.now()) + ' - INFO - Imgs per GPU: ', cfg.data.imgs_per_gpu)
    print(str(datetime.now()) + ' - INFO - Workers per GPU: ', cfg.data.workers_per_gpu)
    print(str(datetime.now()) + ' - INFO - Momentum: ', args.momentum)
    print(str(datetime.now()) + ' - INFO - Epsilon: ', args.epsilon)
    infer_model = load_model(args)
    attack_loader = build_dataloader(dataset, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, cfg.gpus, dist=False)
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    if args.experiment_index == args.resume_experiment:
        if os.path.exists('/home/fengyao/yolov3/data/coco_before_attack.txt'):
            os.remove('/home/fengyao/yolov3/data/coco_before_attack.txt')
        if os.path.exists('/home/fengyao/yolov3/data/coco_under_attack.txt'):
            os.remove('/home/fengyao/yolov3/data/coco_under_attack.txt')
    if args.clear_output:
        file_list = os.listdir(args.save_path)
        for f in file_list:
            if os.path.isdir(os.path.join(args.save_path, f)):
                shutil.rmtree(os.path.join(args.save_path, f))
            else:
                os.remove(os.path.join(args.save_path, f))
    class_names = infer_model.CLASSES
    num_of_classes = len(infer_model.CLASSES)
    max_batch = min(attack_loader.__len__(), args.max_attack_batches)
    pbar_outer = tqdm(total=max_batch)
    pbar_inner = tqdm(total=args.num_attack_iter)
    assert max_batch > 0
    acc_before_attack = 0
    acc_under_attack = 0
    statistics = np.zeros(6)
    number_of_images = 0
    dot_product = 0
    conv_kernel = None
    with_mask = hasattr(model.module, 'mask_head') and model.module.mask_head is not None
    MAP_data = [[[None] * num_of_classes, [None] * num_of_classes], [[None] * num_of_classes, [None] * num_of_classes]]
    if args.kernel_size != 0:
        conv_kernel = conv_layer(args)
    for i, data in enumerate(attack_loader):
        epsilon = args.epsilon / max(data['img_meta'].data[0][0]['img_norm_cfg']['std'])
        if i >= max_batch:
            break
        raw_imgs = copy.deepcopy(data['img'])
        imgs = data['img']
        for j in range(0, len(imgs.data)):
            imgs.data[j] = imgs.data[j].cuda()
            if args.visualize:
                if args.model_name == 'rpn_r50_fpn_1x':
                    visualize_modification(args, infer_model, copy.deepcopy(imgs.data[j]), j,
                                           data['img_meta'].data[j], data['gt_bboxes'].data[j])
                else:
                    visualize_modification(args, infer_model, copy.deepcopy(imgs.data[j]), j,
                                           data['img_meta'].data[j], data['gt_bboxes'].data[j],
                                           data['gt_labels'].data[j])
            if args.generate_data and args.experiment_index == args.resume_experiment:
                assert args.model_name != 'rpn_r50_fpn_1x'
                generate_data(args, copy.deepcopy(imgs.data[j]), False, data['img_meta'].data[j],
                              data['gt_bboxes'].data[j], data['gt_labels'].data[j])
            imgs.data[j] = imgs.data[j].detach()
            imgs.data[j].requires_grad = True
            number_of_images += imgs.data[j].size()[0]
        pbar_inner.reset()
        last_update_direction = list(range(0, len(imgs.data)))
        for _ in range(args.num_attack_iter):
            if args.model_name == 'rpn_r50_fpn_1x':
                result = model(imgs, data['img_meta'], return_loss=True, gt_bboxes=data['gt_bboxes'])
            elif with_mask:
                result = model(imgs, data['img_meta'], return_loss=True, gt_bboxes=data['gt_bboxes'],
                               gt_labels=data['gt_labels'], gt_masks=data['gt_masks'])
            else:
                result = model(imgs, data['img_meta'], return_loss=True,
                               gt_bboxes=data['gt_bboxes'], gt_labels=data['gt_labels'])
            loss = 0
            for key in args.loss_keys:
                if type(result[key]) is list:
                    for losses in result[key]:
                        loss += losses.sum()
                else:
                    loss += result[key].sum()
            loss.backward()
            for j in range(0, len(imgs.data)):
                if args.momentum == 0:
                    update_direction = imgs.data[j].grad
                    if conv_kernel:
                        update_direction = conv_kernel(update_direction)
                    l1_per_img = torch.sum(torch.abs(update_direction), (1, 2, 3), keepdim=True)
                    l1_per_img = l1_per_img.expand(update_direction.size())
                    update_direction = update_direction / l1_per_img
                    imgs.data[j] = imgs.data[j] + epsilon / args.\
                        num_attack_iter * torch.sign(update_direction)
                else:
                    if _ == 0:
                        update_direction = imgs.data[j].grad
                        if conv_kernel:
                            update_direction = conv_kernel(update_direction)
                        l1_per_img = torch.sum(torch.abs(update_direction), (1, 2, 3), keepdim=True)
                        l1_per_img = l1_per_img.expand(update_direction.size())
                        update_direction = update_direction / l1_per_img
                        imgs.data[j] = imgs.data[j] + epsilon / args. \
                            num_attack_iter * torch.sign(update_direction)
                    else:
                        update_direction = imgs.data[j].grad
                        if conv_kernel:
                            update_direction = conv_kernel(update_direction)
                        l1_per_img = torch.sum(torch.abs(update_direction), (1, 2, 3), keepdim=True)
                        l1_per_img = l1_per_img.expand(update_direction.size())
                        update_direction = update_direction / l1_per_img
                        update_direction += args.momentum * last_update_direction[j]
                        imgs.data[j] = imgs.data[j] + epsilon / args. \
                            num_attack_iter * torch.sign(update_direction)
                imgs.data[j] = imgs.data[j].detach()
                if args.visualize:
                    if args.model_name == 'rpn_r50_fpn_1x':
                        visualize_modification(args, infer_model, copy.deepcopy(imgs.data[j]), j,
                                               data['img_meta'].data[j], data['gt_bboxes'].data[j])
                    else:
                        visualize_modification(args, infer_model, copy.deepcopy(imgs.data[j]), j,
                                               data['img_meta'].data[j], data['gt_bboxes'].data[j],
                                               data['gt_labels'].data[j])
                imgs.data[j].requires_grad = True
                if args.visualize and _ > 0:
                    dot_product += torch.sum(update_direction.view(-1) /
                                             torch.norm(update_direction.view(-1)) *
                                             last_update_direction[j].view(-1) /
                                             torch.norm(last_update_direction[j].view(-1)))
                last_update_direction[j] = update_direction
            model.zero_grad()
            pbar_inner.update(1)
        # for j in range(0, cfg.gpus):
        #     if args.model_name == 'rpn_r50_fpn_1x':
        #         t = ThreadingWithResult(visualize_all_images_plus_acc, args=(args, infer_model,
        #                                                                      imgs.data[j], raw_imgs.data[j],
        #                                                                      data['img_meta'].data[j],
        #                                                                      data['gt_bboxes'].data[j],
        #                                                                      MAP_data))
        #     else:
        #         t = ThreadingWithResult(visualize_all_images_plus_acc, args=(args, infer_model,
        #                                                                      imgs.data[j], raw_imgs.data[j],
        #                                                                      data['img_meta'].data[j],
        #                                                                      data['gt_bboxes'].data[j],
        #                                                                      MAP_data,
        #                                                                      data['gt_labels'].data[j]))
        #     t.start()
        #     t.join()
        #     statistics_result = t.get_result()
        #     if statistics_result[0][0] >= 0:
        #         statistics += statistics_result[0]
        #         MAP_data = statistics_result[1]
        #     else:
        #         print("Error! Results were not fetched!")
        for j in range(0, len(imgs.data)):
            generate_data(args, copy.deepcopy(imgs.data[j]), True, data['img_meta'].data[j],
                          data['gt_bboxes'].data[j], data['gt_labels'].data[j])
        pbar_outer.update(1)
    pbar_outer.close()
    pbar_inner.close()
    if args.generate_data and args.experiment_index == args.resume_experiment:
        args.MAP_before_attack, args.class_accuracy_before_attack, args.IoU_accuracy_before_attack,\
            args.IoU_accuracy_before_attack2 = test_attack(False)
    args.MAP_under_attack, args.class_accuracy_under_attack\
        , args.IoU_accuracy_under_attack, args.IoU_accuracy_under_attack2 = test_attack()
    if args.visualize and args.num_attack_iter > 1:
        dot_product /= (args.num_attack_iter - 1) * number_of_images / args.imgs_per_gpu
        print("average normalized dot product = ", dot_product)
    acc_before_attack /= max_batch
    acc_under_attack /= max_batch
    statistics /= number_of_images

    if args.neglect_raw_stat and args.experiment_index > args.resume_experiment:
        pass
    else:
        pass
        # args.class_accuracy_before_attack = 100 * statistics[0]
        # args.IoU_accuracy_before_attack = 100 * statistics[1]
        # args.IoU_accuracy_before_attack2 = 100 * statistics[2]
        # if MAP_data[0] is None:
        #     args.MAP_before_attack = 0
        # else:
        #     args.MAP_before_attack = eval_map_attack(MAP_data[0][0], MAP_data[0][1], len(class_names),
        #                                              scale_ranges=None, iou_thr=0.5,
        #                                              dataset=class_names, print_summary=True)[0]
    # args.class_accuracy_under_attack = 100 * statistics[3]
    # args.IoU_accuracy_under_attack = 100 * statistics[4]
    # args.IoU_accuracy_under_attack2 = 100 * statistics[5]
    # if MAP_data[1] is None:
    #     args.MAP_under_attack = 0
    # else:
    #     args.MAP_under_attack = eval_map_attack(MAP_data[1][0], MAP_data[1][1], len(class_names),
    #                                             scale_ranges=None, iou_thr=0.5,
    #                                             dataset=class_names, print_summary=True)[0]
    args.class_accuracy_decrease = args.class_accuracy_before_attack - args.class_accuracy_under_attack
    args.IoU_accuracy_decrease = args.IoU_accuracy_before_attack - args.IoU_accuracy_under_attack
    args.IoU_accuracy_decrease2 = args.IoU_accuracy_before_attack2 - args.IoU_accuracy_under_attack2
    args.MAP_decrease = 100 * (args.MAP_before_attack - args.MAP_under_attack)
    print("Class & IoU accuracy before attack = %g %g" % (args.class_accuracy_before_attack,
                                                          args.IoU_accuracy_before_attack))
    print("Class & IoU accuracy under attack = %g %g" % (args.class_accuracy_under_attack,
                                                         args.IoU_accuracy_under_attack))
    print("Class & IoU accuracy decrease = %g %g" % (args.class_accuracy_decrease,
                                                     args.IoU_accuracy_decrease))
    print("MAP before attack = %g" % args.MAP_before_attack)
    print("MAP under attack = %g" % args.MAP_under_attack)
    print("MAP decrease = %g" % args.MAP_decrease)
    # torch.cuda.empty_cache()
    return args
