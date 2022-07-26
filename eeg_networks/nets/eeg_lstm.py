#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/2 13:08
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import torch
import torch.nn as nn
import torch.nn.functional


class EEGSpatialLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers=2,
                 num_spatial=8, num_times=None, num_channels=8, dropout_rate=0.5, dependent=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.len_seq = None if num_times is None else num_times // input_size
        self.num_channels = num_channels
        self.num_layers = num_layers
        if dependent:
            self.lstm = nn.ModuleList([
                nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                for _ in range(num_channels)
            ])
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.block = nn.Sequential(
            nn.Conv2d(1, num_spatial, (num_channels, 1)),
            nn.BatchNorm2d(num_spatial),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        self.cls = nn.Sequential(
            nn.Linear(num_spatial * hidden_size // 4, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        # shape of input: Nx1xTxC
        x = x.permute(dims=(0, 1, 3, 2))
        # shape of x: Nx1xCxT
        if self.len_seq is None:
            x = x.view(x.shape[0], self.num_channels, x.shape[-1] // self.input_size, self.input_size)
        else:
            x = x.view(x.shape[0], self.num_channels, self.len_seq, self.input_size)
        # shape of x: NxCxLxI
        if isinstance(self.lstm, nn.ModuleList):
            x = torch.cat([
                lstm(x[:, i])[0][:, -1].unsqueeze(1) for i, lstm in enumerate(self.lstm)
                # shape of output: NxLxH -> Nx1xH
            ], dim=1)
        else:
            x = torch.cat([
                self.lstm(x[:, i])[0][:, -1].unsqueeze(1) for i in range(self.num_channels)
            ], dim=1)
        x = x.unsqueeze(1)
        # shape of x: Nx1xCxH
        x = self.block(x)
        # shape of x: NxSx1xH/4
        x = torch.flatten(x, start_dim=1)
        return self.cls(x)


if __name__ == '__main__':
    input = torch.zeros(10, 1, 128, 8)
    print(EEGSpatialLSTM(2, input_size=16, hidden_size=16)(input)[0].shape)
    # print(nn.LSTM(input_size=8, hidden_size=16)(input.squeeze(1))[1][0].shape)
    # print(nn.LSTM(input_size=8, hidden_size=16)(input.squeeze(1))[1][1].shape)
    print(torch.arange(12).view(4, 3))
