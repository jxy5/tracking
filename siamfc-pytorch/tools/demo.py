from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('~/data/OTB100/Crossing/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = 'pretrained/siamfc_alexnet_e50.pth'  # 用训练好的siamfc_alexnet_e50.pth模型进行tracking
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
    # img_files：视频序列；anno[0]就是第一帧中的ground truth bbox；visualize：可视化demo
