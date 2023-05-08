# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class PrimitiveDataset(Dataset):
    def __init__(self,root = './data/pour_dset', npoints=2048, split='train', pred_start_only=True):
        self.npoints = npoints
        self.root = root
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        self.datapath = {}

        self.pred_start_only = pred_start_only
    
        ctr = 0
        for idx, fn in enumerate(sorted(os.listdir('%s/%s'%(self.root, split)))):
            data = np.load('%s/%s/%s'%(self.root, split, fn), allow_pickle=True).item()
            self.datapath[idx] = os.path.join('%s/%s/%s'%(self.root, split, fn))

    def __getitem__(self, index):
        if index in self.cache:
            if self.pred_start_only:
                point_set, colors, kpt_labels, cls_labels, start_waypt, start_rot = self.cache[index]
            else:
                point_set, colors, kpt_labels, cls_labels, start_waypt, end_waypt, start_rot, end_rot = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.load(fn, allow_pickle=True).item()

            point_set = data['xyz']
            colors = data['xyz_color']
            kpt_labels = data['xyz_kpt']
            start_waypt = data['start_waypoint']
            cls_labels = data['cls']
            start_rot = data['start_ori']

            if not self.pred_start_only:
                end_waypt = data['end_waypoint']
                end_rot = data['end_ori']

            if len(self.cache) < self.cache_size:
                if self.pred_start_only:
                    self.cache[index] = (point_set, colors, kpt_labels, cls_labels, start_waypt, start_rot)
                else:
                    self.cache[index] = (point_set, colors, kpt_labels, cls_labels, start_waypt, end_waypt, start_rot, end_rot)

        point_set[:, 0:3], centroid, m = pc_normalize(point_set[:, 0:3])
        start_waypt, _, _ = pc_normalize(start_waypt, centroid, m)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
    
        point_set = point_set[choice, :]
        kpt_labels = kpt_labels[choice]
        colors = colors[choice, :]
        cls_labels = cls_labels[choice].T

        inp = np.vstack((point_set.T, colors.T, kpt_labels.T)).T
        if self.pred_start_only:
            return inp, cls_labels, start_waypt, start_rot
        else:
            return inp, cls_labels, start_waypt, end_waypt, start_rot, end_rot

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    dset = PrimitiveDataset('data/dset_toolcabinet', npoints=20000, split='train')
    for i in range(10):
        inp, offsets, start, start_rot = dset[i]

    dset = PrimitiveDataset('data/dset_pour', npoints=20000, split='train', pred_start_only=False)
    for i in range(10):
        inp, offsets, start, start_rot, end, end_rot = dset[i]
