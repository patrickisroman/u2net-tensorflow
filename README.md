# u2net-tensorflow

A tensorflow implementation of the [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf) using Keras & Functional API

Based on the [PyTorch version](https://github.com/NathanUA/U-2-Net) by NathanUA

```
virtualenv venv
source venv/bin/activate
pip install tensorflow matplotlib opencv-python
```

The `U2NET` class can be used to instatiate a modular instance of the U2-Net network
```python
u2net = U2NET()
out = u2net(inp)
```

## Training

Download the [DUTS-TR](http://saliencydetection.net/duts/#org3aad434) dataset and unzip into the `data` directory
```
wget http://saliencydetection.net/duts/download/DUTS-TR.zip
unzip DUTS-TR.zip -d data
```

Then start training

```
python train.py
```

## Citation
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```