#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/1 18:55
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import torch
import torch.nn as nn
import torch.nn.functional


class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(x, self.padding)


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
    def __init__(self, num_classes, fs=1282, num_channels=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True)):
        super().__init__()
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    1, filters_per_branch, (scales_sample, 1),
                    padding="same"
                    # padding=((scales_sample - 1) // 2, 0)
                ) if torch.__version__ >= "1.9" else nn.Sequential(
                    CustomPad((0, 0,scales_sample // 2 - 1, scales_sample // 2, )),
                    nn.Conv2d(
                        1, filters_per_branch, (scales_sample, 1)
                    )
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
                DepthWiseConv2d(8, (1, num_channels), 2),
                nn.BatchNorm2d(filters_per_branch * 2),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])
        self.avg_pool1 = nn.AvgPool2d((4, 1))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    len(scales_samples) * 2 * filters_per_branch,
                    filters_per_branch, (scales_sample // 4, 1),
                    bias=False,
                    padding="same"
                    # padding=((scales_sample // 4 - 1) // 2, 0)
                ) if torch.__version__ >= "1.9" else nn.Sequential(
                    CustomPad((0, 0, scales_sample // 8 - 1, scales_sample // 8, )),
                    nn.Conv2d(
                        len(scales_samples) * 2 * filters_per_branch,
                        filters_per_branch, (scales_sample // 4, 1),
                        bias=False
                    )
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(
            nn.Conv2d(
                24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                bias=False, padding='same'
            ) if torch.__version__ >= "1.9" else nn.Sequential(
                CustomPad((0, 0, 4, 3)),
                nn.Conv2d(
                    24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                    bias=False
                )
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(
                12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                bias=False, padding='same'
            ) if torch.__version__ >= "1.9" else nn.Sequential(
                CustomPad((0, 0, 2, 1)),
                nn.Conv2d(
                    12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                    bias=False
                )
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),
        )
        self.cls = nn.Sequential(
            nn.Linear(4 * 1 * 6, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception1], dim=1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], dim=1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, start_dim=1)
        return self.cls(x)
