# ECE 197 Z Deep Learning - Assignment 2: Object Detection
### Fine-tuning a pre-trained torchvision object detection model on a drinks dataset
--------------------------------------------------------------------------------
<br>

**Philip Luis D. Tuason III**

**2018-08149**

**BS Electronics Engineering**

*Electrical and Electronics Engineering Institute (EEEI),*

*College of Engineering,*

*University of the Philippines Diliman*

--------------------------------------------------------------------------------
<br>

The goal of this assignment was to arbitrarily select a pre-trained object detection model from the **torchvision** library and fine-tune it with the provided COCO drinks dataset. I chose to use the **Faster R-CNN model with a ResNet-50 FPN backbone** as it had, for me, the best tradeoff between accuracy and train time according to this *[PyTorch reference](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)*. The official reference for this model can be found *[here](https://arxiv.org/abs/1506.01497)* (arXiv).

In developing this project, I mostly relied on the following references:
  - *[TorchVision object detection fine-tuning tutorial](https://pytorch.org/)*
  - *[PyTorch object detection reference training scripts](https://github.com/pytorch/vision/tree/main/references/detection)*
  - PyTorch and torchvision documentation

--------------------------------------------------------------------------------

<p align='center'><img src='imgs/demo.gif'></p>

--------------------------------------------------------------------------------

<br>

## How to run

It is assumed that CUDA-enabled PyTorch is installed in your environment via conda. Other prerequisites can be found in [requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

<br>

### Testing

After satisfying all the prerequisites, you may evaluate the model through
```
python test.py
```

The dataset and model checkpoints should automatically be downloaded as you run the script.

<br>

### Training

If you wish to replicate the model, you may do so through
```
python train.py
```

<br>

### Demo

To test the model's performance to its limits, you can perform real-time inference through the demo script.
```
python demo.py
```
Note that the frame rate is dependent on how fast the model performs in your environment.