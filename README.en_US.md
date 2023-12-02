English | [简体中文](README.zh-CN.md)

# EEG-PyTorch

Implementation of various simple networks for EEG classification tasks based on PyTorch.

## Implemented Networks

- [x] eeg-inception
- [x] eeg-lstm
- [ ] eeg-gcn

## Usage

### Installation

Install using PYPI.

```sh
pip install eeg-networks
```

If you do not want to install the package, you can also clone this project and install the required packages.

```sh
git clone https://github.com/zrr1999/eeg-networks-pytorch
cd eeg-networks-pytorch
pip install -r requirements.txt
```

### Training

You can train the model using the following command.

```sh
python train.py --model_name inception --device cpu --dataset_path ./GIB-UVA ERP-BCI.hdf5 --model_path ./weights
```

Use the following command to get more detailed information.

```sh
python train.py --help
```

### Validation

You can validate the model using the following command.

```sh
python val.py --model_path ./weights/last.pth --model_name inception
```

Use the following command to get more detailed information.

```sh
python val.py --help
```

## References

[GIB-UVa ERP-BCI dataset](https://www.kaggle.com/datasets/esantamaria/gibuva-erpbci-dataset?resource=download)

