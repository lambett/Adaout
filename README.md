# Adaout
Adaout is a practical and flexible regularization method with high generalization and interpretability.

## Requirements
- python 3.6 (Anaconda version >=5.2.0 is recommended)
- torch (torch version >=1.1.0 is recommended)
- torchvision (torchvision version >=0.3.0 is recommended)
- pandas
- numpy
- NVIDIA GPU + CUDA CuDNN

## Datasets
- CIFAR-10, CIFAR-100, SVHN, ImageNet and others

## Getting started
- Download datasets and extract it inside  `data`
- Train: `python train.py`, `python train100.py` or `python train_svhn.py`
- Evaluate:
  - Pretrained models for CIFAR-10 and CIFAR-100 are available at this [link](https://drive.google.com/file/d/1eJwEdoTtvt00_3d7SgdtSYxbEjJuWLnv/view?usp=sharing). Download and extract them in the `save_model/resnet56_10` or `save_model/resnet56_100` directory.
  - You should achieve about 94.63% accuracy on CIFAR-10, and 74.18% accuracy on CIFAR-100 datasets.
