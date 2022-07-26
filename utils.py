#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/3 11:29
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
from enum import Enum


class ModelName(str, Enum):
    inception = "inception"
    lstm = "lstm"
