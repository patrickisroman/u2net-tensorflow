# u2net-tensorflow

A tensorflow implementation of the [U^2-Net [...] for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf) using Keras & Functional API

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

### Test
```
python simpletest.py
```