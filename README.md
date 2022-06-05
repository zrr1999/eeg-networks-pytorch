# EEG-PyTorch

基于PyTorch的各种用于EEG分类任务的简单网络的实现。
Implementation of various simple networks based on PyTorch to classify EEG signals.

## 已实现网络（Implemented networks）
- [x] eeg-inception
- [x] eeg-lstm
- [ ] eeg-gcn

## 使用方法（Usage）
### 准备工作

下载数据集 [GIB-UVa ERP-BCI dataset](https://www.kaggle.com/datasets/esantamaria/gibuva-erpbci-dataset?resource=download)



安装依赖包

```sh
pip install -r requirements.txt
```

### 训练

```sh
python train.py --model inception
```

```sh
python train.py --model lstm
```