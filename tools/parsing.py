import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path', default='/home/fengyao/mmdetection/configs/')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--max_attack_batches', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--model_name', help='name of loaded model', default='mask_rcnn_r50_fpn_1x')
    parser.add_argument('--train', action='store_true', help='whether or not to train')
    parser.add_argument('--visualize', action='store_true', help='whether or not visualize modification')
    parser.add_argument('--clear_output', action='store_true', help='whether or not to clear output path')
    parser.add_argument('--neglect_raw_stat', action='store_true', help='whether or not to neglect stat '
                                                                        'calculation of raw data')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--black_box_model_path', type=str, default=None)
    parser.add_argument('--black_box_model_name', type=str, default=None)
    parser.add_argument('--num_attack_iter', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=5.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--save_ratio', type=float, default=0.03)
    parser.add_argument('--imgs_per_gpu', type=int, default=4)
    parser.add_argument('--workers_per_gpu', type=int, default=0)
    parser.add_argument('--kernel', type=str, default="Gaussian")
    parser.add_argument('--kernel_size', type=int, default=0)
    parser.add_argument('--resume_experiment', type=int, default=0)
    parser.add_argument('--generate_data', action='store_true', help='whether or not to generate data')
    args = parser.parse_args()
    if args.black_box_model_name is None:
        args.black_box_model_name = args.model_name
    args.config_black_box = args.config + args.black_box_model_name + '_jun9.py'
    args.config += args.model_name + '_jun9.py'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(1)
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(22225)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '101.6.240.88'
    if 'NPROC_PER_NODE' not in os.environ:
        os.environ['NPROC_PER_NODE'] = str(args.gpus)
    return args
