#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/5 17:30
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import torch
import typer
from torch.utils.data import DataLoader
from torch import nn
from eeg_networks import EEGInception, EEGSpatialLSTM, ERPDataset
from utils import ModelName


def validate(model, val_loader, loss_fn, total):
    val_loss_list = []
    val_acc_sum = 0
    with torch.no_grad():
        for feat, label in val_loader:
            out = model(feat)
            val_loss_list.append(loss_fn(out, label))
            val_acc_sum += torch.eq(out.argmax(dim=1), label).sum()

    val_mean_loss = sum(val_loss_list) / total
    val_mean_acc = val_acc_sum / total

    return val_mean_loss, val_mean_acc


def main(
        model_path: str = None,
        model_name: ModelName = "inception",
        device: str = "cuda",
        dataset_path: str = './GIB-UVA ERP-BCI.hdf5',
        batch_size: int = 8192,
):
    val_dataset = ERPDataset(dataset_path, train=False, device=device)
    val_loader = DataLoader(val_dataset, batch_size)
    if model_name == "inception":
        model = EEGInception(2).to(device)
    else:
        model = EEGSpatialLSTM(2, 16, 16).to(device)
    loss_fn = nn.CrossEntropyLoss()

    if model_path is None:
        model.load_state_dict(torch.load(f"./weights/{model_name}/last.pth"))
    else:
        model.load_state_dict(torch.load(model_path))

    model.eval()
    val_mean_loss, val_mean_acc = validate(model, val_loader, loss_fn, len(val_dataset))
    print(val_mean_loss, val_mean_acc)


if __name__ == "__main__":
    typer.run(main)
