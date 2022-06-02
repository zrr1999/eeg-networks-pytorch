#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/5/31 0:23
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import h5py
import numpy as np
import os

import torch
import tqdm
from sklearn.preprocessing import OneHotEncoder
from eeg_inception import EEGInception
from dataset import ERPDataset
from torch.utils.data import DataLoader
from torch import nn, optim

torch.__version__ = "1.2"
device = "cpu"

train_dataset = ERPDataset(device=device)
val_dataset = ERPDataset(train=False, device=device)
train_loader = DataLoader(train_dataset, 1024)
val_loader = DataLoader(val_dataset, 1024)
model = EEGInception().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=False)

try:
    model.load_state_dict(torch.load("./last.pth"))
    print("读取权重")
except FileNotFoundError:
    print("未读取权重")

with tqdm.trange(500) as t:
    for epoch in t:
        for feat, label in train_loader:
            out = model(feat)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t.set_postfix(loss=loss)
    val_loss_list = []
    for epoch in t:
        for feat, label in val_loader:
            out = model(feat)
            val_loss_list.append(loss_fn(out, label))
    val_mean_loss = sum(val_loss_list)/len(val_loss_list)
    torch.save(model.state_dict(), f"./epoch{epoch}_{val_mean_loss}.pth")
    torch.save(model.state_dict(), f"./last.pth")
