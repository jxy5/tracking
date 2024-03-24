import os
import sys
import time
import argparse
import functools

sys.path.append("/home/yan/tracking/HiFT")  # 这里放项目的绝对路径，就能找到toolkit和pysot啦

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import UAV10Dataset, UAV20Dataset, DTBDataset, UAVDataset
from toolkit.evaluation import OPEBenchmark
from toolkit.visualization import draw_success_precision

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    # parser.add_argument('--dataset_dir', default='',type=str, help='dataset root directory')
    # parser.add_argument('--dataset', default='DTB70',type=str, help='dataset name')
    # parser.add_argument('--tracker_result_dir',default='', type=str, help='tracker result root')
    # parser.add_argument('--trackers',default='general_model', nargs='+')
    # parser.add_argument('--vis', default='',dest='vis', action='store_true')
    # parser.add_argument('--show_video_level', default=' ',dest='show_video_level', action='store_true')
    # parser.add_argument('--num', default=1, type=int, help='number of processes to eval')
    # args = parser.parse_args()
    # /home/ubuntu/Share/QY/Dataset_UAV123_10fps/UAV10fps
    parser = argparse.ArgumentParser(description='tracking evaluation')
    parser.add_argument('--tracker_path', '-p', default='/home/yan/tracking/HiFT/tools/results', type=str,
                        help='tracker result path')
    parser.add_argument('--dataset', '-d', default='UAV10fps', type=str, help='dataset name')
    parser.add_argument('--num', '-n', default=1, type=int, help='number of thread to eval')
    parser.add_argument('--tracker_prefix', '-t', default='', type=str, help='tracker name')
    parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.set_defaults(show_video_level=True)
    args = parser.parse_args()

    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix + '*'))
    print(os.path.join(args.tracker_path,
                       args.dataset,
                       args.tracker_prefix + '*'))
    # trackers = [x.split('/')[-1] for x in trackers]
    trackers = [os.path.basename(x) for x in trackers]
    print("trackers", len(trackers))
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                          '../testing_dataset'))
    root = os.path.join('/home/yan/tracking/HiFT/test_dataset/UAV123_10fps', 'data_seq/UAV123_10fps')

    # trackers=args.tracker_prefix

    if 'UAV10fps' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)

        benchmark = OPEBenchmark(dataset)
        # success
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        # precision
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        # norm precision
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                                trackers), desc='eval norm precision', total=len(trackers), ncols=18):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret.update(ret),
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret,
                                       norm_precision_ret=norm_precision_ret)
    elif 'UAV20l' in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'DTB70' in args.dataset:
        dataset = DTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    elif 'UAV123' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
    else:
        print('dataset error')
