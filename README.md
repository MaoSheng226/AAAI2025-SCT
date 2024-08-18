# SCT: Structure Contextual Learning for ORSI Salient Object Detection with Transformer
## Introduction

This is the code for the paper "SCT: Structure Contextual Learning for ORSI Salient Object Detection with Transformer," which relies on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for its implementation.
To run the code, you can follow the installation process of mmsegmentation, and additionally, pay special attention to installing the package [PySODMetrics](https://github.com/lartpang/PySODMetrics) for evaluation metrics. If you have the mmsegmentation version 1.2.0 installed, you can directly add the modifications on top of it.\\
The Checkpoint is
URL: https://pan.baidu.com/s/15yKZC5-nIyk2wNZ-Ymd8bg?pwd=vdh9\\
The directories to be modified are as follows:
* configs
* mmseg
  * datasets
  * evaluation
  * models
    * backbones
    * decode_heads
    * losses
## Guide to Use
For network training, enter the following command:
```
python tools/train.py configs/sd/SCDecoderHead_OSTb5_1xb8-320k_EORSSD-512x512.py
```
For testing, enter the following command:
```
python tools/test.py configs/sd/SCDecoderHead_OSTb5_1xb8-320k_EORSSD-512x512.py [checkpoint_path] --tta --out [result_save_path]
```
Evaluate the specific values of each metric:
```
python tools/evalution2.py
```
