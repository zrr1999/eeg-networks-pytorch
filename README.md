English | [简体中文](README_ch.md)
# EEG-PyTorch
Implementation of various simple networks based on PyTorch to classify EEG signals.

## Implemented networks
- [x] eeg-inception
- [x] eeg-lstm
- [ ] eeg-gcn

## Usage
### Install
Install via PYPI.
```sh
pip install EEGNetworks
```

如果不想安装包，也可以克隆本项目并安装依赖包。
Clone repo and install requirements.txt.
```sh
pip install -r requirements.txt
```
### Training
可以通过以下命令训练模型
```sh
python train.py --model_name inception --device cpu --dataset_path ./GIB-UVA ERP-BCI.hdf5 --model_path ./weights
```
通过以下命令获取更多详细信息
```sh
python train.py --help
```
### Validation
You can validate the model with the following command.
```sh
python val.py --model_path ./weights/last.pth --model_name inception
```
Get more details with the following command.
```sh
python val.py --help
```

## Reference
[GIB-UVa ERP-BCI dataset](https://www.kaggle.com/datasets/esantamaria/gibuva-erpbci-dataset?resource=download)
