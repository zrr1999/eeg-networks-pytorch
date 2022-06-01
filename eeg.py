#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/1 18:55
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import torch
import torch.nn as nn


class DepthWiseConv2d(nn.Module):
    def __init__(self, channel, kernel_size, depth_multiplier, bias=False):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size, bias=bias, groups=channel)
            for _ in range(depth_multiplier)
        ])

    def forward(self, x):
        output = torch.cat([net(x) for net in self.nets], 1)
        return output


class EEGInception(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True), n_classes=2, learning_rate=0.001):
        super().__init__()
        # ============================= CALCULATIONS ============================= #
        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]

        # ================================ INPUT ================================= #
        # input_layer = Input((input_samples, ncha, 1))

        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    1, filters_per_branch, (scales_sample, 1),
                    padding="same"
                    # padding=((scales_sample - 1) // 2, 0)
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
                DepthWiseConv2d(8, (1, ncha), 2),
                nn.BatchNorm2d(filters_per_branch*2),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])
        self.avg_pool1 = nn.AvgPool2d((4, 1))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    len(scales_samples)*2*filters_per_branch,
                    filters_per_branch, (scales_sample // 4, 1),
                    bias=False,
                    padding="same"
                    # padding=((scales_sample // 4 - 1) // 2, 0)
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(
            nn.Conv2d(24, filters_per_branch * len(scales_samples) // 2, (8, 1), bias=False, padding='same'),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(12, filters_per_branch * len(scales_samples) // 4, (4, 1), bias=False, padding='same'),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),
        )

        # Output layer
        self.cls = nn.Sequential(
            nn.Linear(4 * 1 * 6, n_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception1], 1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], 1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, 1)
        return self.cls(x)


if __name__ == '__main__':
    print(EEGInception()(torch.zeros(10, 1, 128, 8)).shape)
    print(DepthWiseConv2d(3, (3, 3), 2)(torch.zeros(10, 3, 512, 512)).shape)
    print(nn.AvgPool2d((2, 1))(torch.zeros(10, 3, 512, 512)).shape)
