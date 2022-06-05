#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/5/31 0:23
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import torch
import tqdm
import typer
from torch.utils.data import DataLoader
from torch import nn, optim
from eeg_networks import EEGInception, EEGSpatialLSTM, ERPDataset
from utils import Model


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
        model: Model = "inception",
        device: str = "cuda",
        dataset_path: str = './GIB-UVA ERP-BCI.hdf5',
        batch_size: int = 8192 + 2048,
        max_epoch: int = 500,
        model_path: str = "./weights"
):
    train_dataset = ERPDataset(dataset_path, device=device)
    val_dataset = ERPDataset(dataset_path, train=False, device=device)
    train_loader = DataLoader(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size)
    if model == "inception":
        eeg_model = EEGInception(2).to(device)
    else:
        eeg_model = EEGSpatialLSTM(2, 16, 16).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(eeg_model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=False)

    try:
        eeg_model.load_state_dict(torch.load(f"./weights/{model}/last.pth"))
        print("读取权重")
    except FileNotFoundError:
        print("未读取权重")

    for epoch in range(max_epoch):
        eeg_model.train()
        with tqdm.tqdm(train_loader) as t:
            for feat, label in t:
                out = eeg_model(feat)
                loss = loss_fn(out, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(
                    loss=loss.item() / batch_size,
                    acc=torch.eq(out.argmax(dim=1), label).sum().item() / len(label),
                    epoch=f"{epoch}/{max_epoch}"
                )
        eeg_model.eval()
        val_mean_loss, val_mean_acc = validate(eeg_model, val_loader, loss_fn, len(val_dataset))
        print(val_mean_loss, val_mean_acc)
        torch.save(eeg_model.state_dict(), f"./weights/{model}/epoch{epoch}_{val_mean_acc * 100:.2f}%.pth")
        torch.save(eeg_model.state_dict(), f"./weights/{model}/last.pth")


if __name__ == "__main__":
    typer.run(main)
