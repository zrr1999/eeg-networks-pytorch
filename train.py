#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/5/31 0:23
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import torch
import tqdm
import typer
from nets import EEGInception, EEGSpatialLSTM
from dataset import ERPDataset
from torch.utils.data import DataLoader
from torch import nn, optim
from enum import Enum


class Model(str, Enum):
    inception = "inception"
    lstm = "lstm"


def main(
        model: Model = "inception",
        device: str = "cpu",
        dataset_path: str = './GIB-UVA ERP-BCI.hdf5',
        model_path: str = ""
):
    train_dataset = ERPDataset(dataset_path, device=device)
    val_dataset = ERPDataset(dataset_path, train=False, device=device)
    train_loader = DataLoader(train_dataset, 1024)
    val_loader = DataLoader(val_dataset, 1024)
    if model == "inception":
        model = EEGInception().to(device)
    else:
        model = EEGSpatialLSTM(2, 16, 16).to(device)
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
                t.set_postfix(loss=loss.item())
        val_loss_list = []
        with torch.no_grad():
            for epoch in t:
                for feat, label in val_loader:
                    out = model(feat)
                    val_loss_list.append(loss_fn(out, label))
        val_mean_loss = sum(val_loss_list) / len(val_loss_list)
        torch.save(model.state_dict(), f"./epoch{epoch}_{val_mean_loss}.pth")
        torch.save(model.state_dict(), f"./last.pth")


if __name__ == "__main__":
    typer.run(main)
