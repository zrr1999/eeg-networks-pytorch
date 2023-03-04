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

If you do not want to install packages, you can also clone this project and install dependent packages.
```sh
git clone https://github.com/zrr1999/eeg-networks-pytorch
cd eeg-networks-pytorch
pip install -r requirements.txt
```
### Training
You can train the model with the following command.
```sh
python train.py --model_name inception --device cpu --dataset_path ./GIB-UVA ERP-BCI.hdf5 --model_path ./weights
```
Get more details with the following command.
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
