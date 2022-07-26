#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/1 21:19
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import h5py
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset


class ERPDataset(Dataset):
    def __init__(self, path, train=True, sep=561615, device="cpu"):
        with h5py.File(path, 'r') as hf:
            self.features = np.array(hf.get("features"), dtype="float32")
            self.erp_labels = np.array(hf.get("erp_labels"), dtype="long")
        if train:
            self.features = self.features[:sep]
            self.erp_labels = self.erp_labels[:sep]
        else:
            self.features = self.features[sep:]
            self.erp_labels = self.erp_labels[sep:]
        self.erp_labels = torch.from_numpy(self.erp_labels).to(device)
        self.features = torch.from_numpy(self.features.reshape(
            (self.features.shape[0], 1, self.features.shape[1],
             self.features.shape[2])
        )).to(device)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.erp_labels[index]

    def __len__(self):
        return len(self.erp_labels)
