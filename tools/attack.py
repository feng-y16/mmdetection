from __future__ import division
import argparse
import os
from mmdet import __version__
import torch
from mmcv import Config
from parsing import parse_args
from train import *
from mmdet.apis import init_detector, inference_detector, show_result, show_result_plus_acc
from mmdet.apis import get_root_logger, init_dist, set_random_seed
from mmdet.apis.train import *
import pdb
import os
import numpy as np
from tqdm import trange
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
import xlwt
import pandas as pd
import datetime
import itertools
import mmcv
from mmcv.runner import load_checkpoint


def load_model(args):
    if args.train:
        train(args)
        model = torch.load(args.work_dir + '/' + args.model_name + '.pth')
    else:
        if args.model_path is None:
            print('Model path missing!')
            return
        if args.black_box_model_path is not None:
            model = init_detector(args.config_black_box, args.black_box_model_path, device='cuda:0')
        else:
            model = init_detector(args.config, args.model_path, device='cuda:0')
    model.eval()
    return model


def visualize():
    args = parse_args()
    model = load_model(args)
    root = '/home/fengyao/MSCOCO2017dataset/test/test2017/'
    save_path = '/home/fengyao/MSCOCO2017dataset/test/test2017/output/'
    imgs = ['000000000001.jpg',
            '000000000016.jpg']
    for index in range(0, len(imgs)):
        result = inference_detector(model, root + imgs[index])
        show_result(root + imgs[index], result, model.CLASSES, show=False, out_file=save_path + imgs[index])
    print('[INFO]Done.')


def visualize_modification(args, model, imgs, index, metadata, gt_bboxes, gt_labels=None):
    imgs = imgs.permute(0, 2, 3, 1)[:, :, :, [2, 1, 0]]
    num_of_imgs = imgs.size()[0]
    imgs = imgs.detach().cpu().numpy()
    for img_index in range(num_of_imgs):
        img_mean = metadata[img_index]['img_norm_cfg']['mean'][::-1]
        img_std = metadata[img_index]['img_norm_cfg']['std'][::-1]
        imgs[img_index] = imgs[img_index] * img_std + img_mean
        # result = inference_detector(model, imgs[img_index])
        if not os.path.exists(args.save_path + str(index) + '/' + str(img_index)):
            os.makedirs(args.save_path + str(index) + '/' + str(img_index))
        save_path = args.save_path + str(index) + '/' + str(img_index) + '/' + str(datetime.datetime.now()) + '.jpg'
        img = mmcv.imread(imgs[img_index])
        result = inference_detector(model, img)
        show_result_plus_acc(img, result, model.CLASSES, gt_bboxes[img_index], gt_labels[img_index],
                             [[[None] * num_of_classes, [None] * num_of_classes],
                              [[None] * num_of_classes, [None] * num_of_classes]], show=False,
                             out_file=save_path)
        # show_result(imgs[img_index], result, model.CLASSES, show=False, out_file=save_path)
        # mmcv.imwrite(img, save_path)


def generate_data(args, imgs, is_attack, metadata, gt_bboxes, gt_labels=None):
    imgs = imgs.permute(0, 2, 3, 1)[:, :, :, [2, 1, 0]]
    num_of_imgs = imgs.size()[0]
    imgs = imgs.detach().cpu().numpy()
    for img_index in range(num_of_imgs):
        img_mean = metadata[img_index]['img_norm_cfg']['mean'][::-1]
        img_std = metadata[img_index]['img_norm_cfg']['std'][::-1]
        imgs[img_index] = imgs[img_index] * img_std + img_mean
        height, width, _ = np.shape(imgs[img_index])
        if is_attack:
            if not os.path.exists(args.save_path[:-7] + 'images/attack'):
                os.makedirs(args.save_path[:-7] + 'images/attack')
            save_path = args.save_path[:-7] + 'images/attack/' + metadata[img_index]['filename'][-16:]
        else:
            with open('/home/fengyao/yolov3/data/coco_before_attack.txt', mode='a') as f:
                f.writelines('/home/fengyao/mmdetection' + args.save_path[1:-7] + 'images/original'
                             + '/' + metadata[img_index]['filename'][-16:])
                f.writelines('\n')
            with open('/home/fengyao/yolov3/data/coco_under_attack.txt', mode='a') as f:
                f.writelines('/home/fengyao/mmdetection' + args.save_path[1:-7] + 'images/attack'
                             + '/' + metadata[img_index]['filename'][-16:])
                f.writelines('\n')
            if not os.path.exists(args.save_path[:-7] + 'images/original'):
                os.makedirs(args.save_path[:-7] + 'images/original')
            save_path = args.save_path[:-7] + 'images/original/' + metadata[img_index]['filename'][-16:]
            if not os.path.exists(args.save_path[:-7] + 'labels/original'):
                os.makedirs(args.save_path[:-7] + 'labels/original')
            file_handle = open(args.save_path[:-7] + 'labels/original/' + metadata[img_index]['filename'][-16:-4] +
                               '.txt', mode='w')
            for label_index in range(gt_bboxes[img_index].size()[0]):
                assert gt_labels[img_index][label_index] >= 1
                assert gt_labels[img_index][label_index] <= 80
                file_handle.writelines(str(int(gt_labels[img_index][label_index] - 1)) + ' ')
                file_handle.writelines(str(float((gt_bboxes[img_index][label_index][0] +
                                                  gt_bboxes[img_index][label_index][2]) / (2 * width))) + ' ')
                file_handle.writelines(str(float((gt_bboxes[img_index][label_index][1] +
                                                  gt_bboxes[img_index][label_index][3]) / (2 * height))) + ' ')
                file_handle.writelines(str(float((gt_bboxes[img_index][label_index][2] -
                                                  gt_bboxes[img_index][label_index][0]) / width)) + ' ')
                file_handle.writelines(str(float((gt_bboxes[img_index][label_index][3] -
                                                  gt_bboxes[img_index][label_index][1]) / height)) + ' ')
                file_handle.writelines('\n')
            file_handle.close()
        img = mmcv.imread(imgs[img_index])
        mmcv.imwrite(img, save_path)


def visualize_img(model, img, save_path):
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, show=False, out_file=save_path)
    return


def visualize_all_images(args, model, imgs, raw_imgs, metadata):
    imgs = imgs.detach().cpu().numpy()
    raw_imgs = raw_imgs.numpy()
    imgs = imgs.transpose(0, 2, 3, 1)
    raw_imgs = raw_imgs.transpose(0, 2, 3, 1)
    imgs = imgs[:, :, :, [2, 1, 0]]
    raw_imgs = raw_imgs[:, :, :, [2, 1, 0]]
    for index in range(0, np.shape(imgs)[0]):
        raw_filename = metadata[index]['filename']
        imgs_mean = metadata[index]['img_norm_cfg']['mean']
        imgs_std = metadata[index]['img_norm_cfg']['std']
        for k in range(0, 3):
            imgs[index][:, :, k] = imgs[index][:, :, k] * imgs_std[2 - k] + imgs_mean[2 - k]
            raw_imgs[index][:, :, k] = raw_imgs[index][:, :, k] * imgs_std[2 - k] + imgs_mean[2 - k]
        (_, filename) = os.path.split(raw_filename)
        filename = filename.split('.', 2)[0]
        visualize_img(model, raw_imgs[index], args.save_path + filename + '.jpg')
        visualize_img(model, imgs[index], args.save_path + filename + '_attack' + '.jpg')
    return


def visualize_img_plus_acc(model, img, metadata, gt_bboxes, gt_labels, save_path, map_data):
    img_mean = metadata['img_norm_cfg']['mean'][::-1]
    img_std = metadata['img_norm_cfg']['std'][::-1]
    img = img * img_std + img_mean
    result = inference_detector(model, img)
    return show_result_plus_acc(img, result, model.CLASSES, gt_bboxes, gt_labels,
                                map_data, show=False, out_file=save_path)


def visualize_all_images_plus_acc(args, model, imgs, raw_imgs, metadata, gt_bboxes, map_data, gt_labels=None):
    imgs = imgs.permute(0, 2, 3, 1)[:, :, :, [2, 1, 0]]
    raw_imgs = raw_imgs.permute(0, 2, 3, 1)[:, :, :, [2, 1, 0]]
    imgs = imgs.detach().cpu().numpy()
    raw_imgs = raw_imgs.numpy()
    raw_class_acc, raw_iou_acc = 0, 0
    class_acc, iou_acc = 0, 0
    raw_iou_acc2 = 0
    iou_acc2 = 0
    if gt_labels is None:
        gt_labels = [-1] * np.shape(imgs)[0]
    for index in range(0, np.shape(imgs)[0]):
        raw_filename = metadata[index]['filename']
        (_, filename) = os.path.split(raw_filename)
        filename = filename.split('.', 2)[0]
        if torch.rand(1) < args.save_ratio:
            raw_class_acc_image, raw_iou_acc_image, raw_iou_acc_image2, map_data[0] = visualize_img_plus_acc(
                model, raw_imgs[index], metadata[index], gt_bboxes[index], gt_labels[index],
                args.save_path + filename, map_data[0])
            class_acc_image, iou_acc_image, iou_acc_image2, map_data[1] = visualize_img_plus_acc(
                model, imgs[index], metadata[index], gt_bboxes[index], gt_labels[index],
                args.save_path + filename + '_attack', map_data[1])
        else:
            if args.neglect_raw_stat and args.experiment_index > args.resume_experiment:
                raw_class_acc_image = 0
                raw_iou_acc_image = 0
                raw_iou_acc_image2 = 0
            else:
                raw_class_acc_image, raw_iou_acc_image, raw_iou_acc_image2, map_data[0] = visualize_img_plus_acc(
                    model, raw_imgs[index], metadata[index], gt_bboxes[index], gt_labels[index], None, map_data[0])
            class_acc_image, iou_acc_image, iou_acc_image2, map_data[1] = visualize_img_plus_acc(
                model, imgs[index], metadata[index], gt_bboxes[index], gt_labels[index], None, map_data[1])
        raw_class_acc += raw_class_acc_image
        raw_iou_acc += raw_iou_acc_image
        raw_iou_acc2 += raw_iou_acc_image2
        class_acc += class_acc_image
        iou_acc += iou_acc_image
        iou_acc2 += iou_acc_image2
    return np.array([raw_class_acc, raw_iou_acc, raw_iou_acc2, class_acc, iou_acc, iou_acc2]), map_data


def attack(args, datasets):
    cfg = Config.fromfile(args.config)
    cfg.data.workers_per_gpu = args.workers_per_gpu
    cfg.data.imgs_per_gpu = args.imgs_per_gpu
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        args.work_dir = cfg.work_dir
    args.save_path = args.work_dir + '/output/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.model_path)
    if datasets is None:
        datasets = [build_dataset(cfg.data.train)]
    # if len(cfg.workflow) == 2:
    # datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__, config=cfg.text, CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    args = attack_detector(args, model, cfg, datasets[0])
    return args, datasets


def save_to_excel(dict_list, file_name):
    data = pd.DataFrame(dict_list)
    file_path = pd.ExcelWriter(file_name)
    data.fillna(' ', inplace=True)
    data.to_excel(file_path, encoding='utf-8', index=False)
    file_path.save()


if __name__ == "__main__":
    result_dict_list = []
    args_raw = parse_args()
    save_keys = ['epsilon', 'loss_keys', 'num_attack_iter', 'momentum', 'kernel', 'kernel_size', 'MAP_decrease',
                 'class_accuracy_decrease', 'IoU_accuracy_decrease', 'IoU_accuracy_decrease2', 'MAP_before_attack',
                 'MAP_under_attack', 'class_accuracy_before_attack', 'class_accuracy_under_attack',
                 'IoU_accuracy_before_attack', 'IoU_accuracy_under_attack', 'IoU_accuracy_before_attack2',
                 'IoU_accuracy_under_attack2', 'model_name', 'config', 'work_dir', 'gpus', 'imgs_per_gpu',
                 'max_attack_batches', 'seed', 'model_path', 'save_path']
    search_dict = ['epsilon', 'loss_keys', 'num_attack_iter', 'momentum', 'kernel', 'kernel_size']
    search_values = [[16.0],
                     [['loss_rpn_bbox', 'loss_cls']],
                     [1, 10, 20],
                     [0, 1, 2],
                     ['Uniform', 'Linear', 'Gaussian'],
                     [0, 5, 11, 15]]
    if args_raw.DIM:
        search_values = [[16.0],
                         [['loss_rpn_bbox', 'loss_cls']],
                         [10],
                         [1],
                         ['Uniform', 'Linear', 'Gaussian'],
                         [0, 5, 11, 15]]
    if args_raw.model_name == 'retinanet_r50_fpn_1x':
        search_values[1] = [['loss_cls']]
    # search_values = [[16.0],
    #                  [['loss_rpn_bbox', 'loss_cls']],
    #                  [10],
    #                  [2],
    #                  ['Gaussian'],
    #                  [15]]
    if args_raw.model_name == 'rpn_r50_fpn_1x':
        search_values = [[16.0],
                         [['loss_rpn_bbox', 'loss_rpn_cls']],
                         [1, 10, 20],
                         [0, 1, 2],
                         ['Uniform', 'Linear', 'Gaussian'],
                         [0, 5, 11, 15]]
    args_raw.MAP_before_attack = None
    args_search = copy.deepcopy(args_raw)
    if args_search.black_box_model_path is None:
        save_file_name = str(datetime.datetime.now()) + '.xlsx'
    else:
        save_file_name = str(datetime.datetime.now()) + '_attack_' + str(args_search.black_box_model_name) + '.xlsx'
    loaded_datasets = None
    experiment_index = 0
    for search_value in itertools.product(*search_values):
        save_dict = {}
        if args_raw.neglect_raw_stat:
            args_search = copy.deepcopy(args_search)
        else:
            args_search = copy.deepcopy(args_raw)
        for i in range(0, len(search_dict)):
            exec('args_search.' + search_dict[i] + ' = search_value[i]')
        if args_search.num_attack_iter == 1 and args_search.momentum > 0:
            continue
        if experiment_index < args_raw.resume_experiment:
            experiment_index = experiment_index + 1
            continue
        args_search.experiment_index = experiment_index
        args_search, loaded_datasets = attack(args_search, loaded_datasets)
        args_dict = vars(args_search)
        for key in save_keys:
            save_dict[key] = args_dict[key]
        result_dict_list.append(save_dict)
        save_to_excel(result_dict_list, args_search.work_dir + save_file_name)
        experiment_index = experiment_index + 1
