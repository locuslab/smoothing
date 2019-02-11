# Experiments

This document describes how to replicate our results.

First, train the models on ImageNet and CIFAR-10:

```
python train.py imagenet resnet50  models/imagenet/resnet50/noise_0.00 --batch 400 --noise 0.0
python train.py imagenet resnet50  models/imagenet/resnet50/noise_0.25 --batch 400 --noise 0.25
python train.py imagenet resnet50  models/imagenet/resnet50/noise_0.50 --batch 400 --noise 0.5
python train.py imagenet resnet50  models/imagenet/resnet50/noise_1.00 --batch 400 --noise 1.0

python train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_0.00 --batch 400 --noise 0.00 --gpu [num]
python train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_0.12 --batch 400 --noise 0.12 --gpu [num]
python train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_0.25 --batch 400 --noise 0.25 --gpu [num]
python train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_0.50 --batch 400 --noise 0.50 --gpu [num]
python train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_1.00 --batch 400 --noise 1.00 --gpu [num]
```
On ImageNet, `train.py` uses all available GPUs in synchronous SGD; on CIFAR-10 it just uses one GPU.

Then, certify a subsample of the test set on ImageNet and CIFAR-10:

```
python code/certify.py imagenet models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25 --skip 100 --batch 400
python code/certify.py imagenet models/imagenet/resnet50/noise_0.50/checkpoint.pth.tar 0.50 data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50 --skip 100 --batch 400
python code/certify.py imagenet models/imagenet/resnet50/noise_1.00/checkpoint.pth.tar 1.00 data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00 --skip 100 --batch 400

python code/certify.py cifar10 models/cifar10/resnet110/noise_0.12/checkpoint.pth.tar 0.25 data/predict/cifar10/resnet110/noise_0.12/test/sigma_0.12 --skip 20 --batch 400
python code/certify.py cifar10 models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 data/predict/cifar10/resnet110/noise_0.25/test/sigma_0.25 --skip 20 --batch 400
python code/certify.py cifar10 models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 data/predict/cifar10/resnet110/noise_0.50/test/sigma_0.50 --skip 20 --batch 400
python code/certify.py cifar10 models/cifar10/resnet110/noise_1.00/checkpoint.pth.tar 1.00 data/predict/cifar10/resnet110/noise_1.00/test/sigma_1.00 --skip 20 --batch 400
```

Then try to certify when the training and testing noise is mismatched:
```
python code/certify.py imagenet models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.50 data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.50 --skip 100 --batch 400
python code/certify.py imagenet models/imagenet/resnet50/noise_1.00/checkpoint.pth.tar 0.50 data/certify/imagenet/resnet50/noise_1.00/test/sigma_0.50 --skip 100 --batch 400

python code/certify.py cifar10 models/cifar10/resnet110/noise_0.00/checkpoint.pth.tar 0.50 data/predict/cifar10/resnet110/noise_0.00/test/sigma_0.50 --skip 20 --batch 400
python code/certify.py cifar10 models/cifar10/resnet110/noise_0.12/checkpoint.pth.tar 0.50 data/predict/cifar10/resnet110/noise_0.12/test/sigma_0.50 --skip 20 --batch 400
python code/certify.py cifar10 models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.50 data/predict/cifar10/resnet110/noise_0.25/test/sigma_0.50 --skip 20 --batch 400
python code/certify.py cifar10 models/cifar10/resnet110/noise_1.00/checkpoint.pth.tar 0.50 data/predict/cifar10/resnet110/noise_1.00/test/sigma_0.50 --skip 20 --batch 400
```

Prediction experiments on ImageNet:
```
python code/predict.py imagenet models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 data/predict/imagenet/resnet50/noise_0.25/test/N_100 --N 100 --skip 100 --batch 400
python code/predict.py imagenet models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 data/predict/imagenet/resnet50/noise_0.25/test/N_1000 --N 1000 --skip 100 --batch 400
python code/predict.py imagenet models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 data/predict/imagenet/resnet50/noise_0.25/test/N_10000 --N 10000 --skip 100 --batch 400
python code/predict.py imagenet models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 data/predict/imagenet/resnet50/noise_0.25/test/N_100000 --N 100000 --skip 100 --batch 400
```

Finally, to visualize noisy images:
```
python code/visualize.py imagenet figures/example_images/imagenet 100 0.0 0.25 0.5 1.0
python code/visualize.py imagenet figures/example_images/imagenet 5400 0.0 0.25 0.5 1.0
python code/visualize.py imagenet figures/example_images/imagenet 9025 0.0 0.25 0.5 1.0
python code/visualize.py imagenet figures/example_images/imagenet 19411 0.0 0.25 0.5 1.0

python code/visualize.py cifar10 figures/example_images/cifar10 10 0.0 0.25 0.5 1.0
python code/visualize.py cifar10 figures/example_images/cifar10 20 0.0 0.25 0.5 1.0
python code/visualize.py cifar10 figures/example_images/cifar10 70 0.0 0.25 0.5 1.0
python code/visualize.py cifar10 figures/example_images/cifar10 110 0.0 0.25 0.5 1.0
```