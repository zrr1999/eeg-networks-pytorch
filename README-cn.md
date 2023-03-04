[English](README.md) | 简体中文
# EEG-PyTorch
基于PyTorch的各种用于EEG分类任务的简单网络的实现。

## 已实现网络
- [x] eeg-inception
- [x] eeg-lstm
- [ ] eeg-gcn

## 使用方法
### 安装
使用 PYPI 安装。
```sh
pip install EEGNetworks
```

如果不想安装包，也可以克隆本项目并安装依赖包。
```sh
git clone https://github.com/zrr1999/eeg-networks-pytorch
cd eeg-networks-pytorch
pip install -r requirements.txt
```
### 训练
可以通过以下命令训练模型
```sh
python train.py --model_name inception --device cpu --dataset_path ./GIB-UVA ERP-BCI.hdf5 --model_path ./weights
```
通过以下命令获取更多详细信息
```sh
python train.py --help
```
### 验证
可以通过以下命令验证模型
```sh
python val.py --model_path ./weights/last.pth --model_name inception
```
通过以下命令获取更多详细信息
```sh
python val.py --help
```

## 参考资料（Reference）
[GIB-UVa ERP-BCI dataset](https://www.kaggle.com/datasets/esantamaria/gibuva-erpbci-dataset?resource=download)
