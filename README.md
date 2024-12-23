# TDCNet Transparent Objects Depth Completion with CNN-Transformer Dual-Branch Parallel Network

[[Paper]](https://arxiv.org/abs/2412.14961)

PyTorch implementation of paper 'TDCNet Transparent Objects Depth Completion with CNN-Transformer Dual-Branch Parallel Network'

## Dataset Preparation

### ClearGrasp Dataset

ClearGrasp can be downloaded at their [official website](https://sites.google.com/view/cleargrasp/data)

### Omniverse Object Dataset

Omniverse Object Dataset can be downloaded [here](https://drive.google.com/drive/folders/1wCB1vZ1F3up5FY5qPjhcfSfgXpAtn31H?usp=sharing). 

## TransCG Dataset

TransCG dataset is now available on [official page](https://graspnet.net/transcg). 

## Requirements

The code has been tested under

- Ubuntu 22.04 + NVIDIA GeForce RTX 4090
- PyTorch 1.11.0

System dependencies can be installed by:

```bash
sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
sudo apt install libopenexr-dev zlib1g-dev openexr
```

Python environment dependencies can be installed by

```bash
pip install -r requirements.txt
```

## Testing

We provide pre-trained weight files on the TransCG dataset ’. /TransCG_checkpoint.7z‘, please unzip the file before use.
