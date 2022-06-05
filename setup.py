#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/6/5 14:30
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import eeg_networks
from setuptools import setup, find_packages

with open("README.md", "r", encoding='UTF-8') as file:
    long_description = file.read()

setup(
    name="EEGNetworks",
    version=eeg_networks.__version__,
    author="六个骨头",
    author_email="2742392377@qq.com",
    description="""
    基于PyTorch的各种用于EEG分类任务的简单网络的实现。
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zrr1999/eeg-networks-pytorch",
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages(exclude=["tests", "_*"]),
    package_data={
        '': ['*.txt'],
        # 包含demo包data文件夹中的 *.dat文件
        'demo': ['data/*.dat'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    platforms="any",
    install_requires=["pytorch", "tqdm", "numpy", "h5py"],
    # extras_require={
    #     'server': ['fastapi', 'pydantic', 'uvicorn>=0.3.0'],
    #     'parse': ['pyparsing'],
    #     'command': ['typer'],
    # },
    entry_points={
        # 'console_scripts': [
        #     'bonetex=bonetex:main',
        # ]
    }
)

