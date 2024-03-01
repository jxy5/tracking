from __future__ import absolute_import

import os
from got10k.experiments import *

from pysot_toolkit.trackers import Tracker


if __name__ == '__main__':
    # net_path = '/home/yan/TransT/pytracking/networks/transt.pth'
    tracker = Tracker('/home/yan/TransT/pytracking/networks/transt.pth')

    root_dir = os.path.expanduser('~/data/OTB100')
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker, visualize=True)
    e.report([tracker.name])