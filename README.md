# Camera Model Identification

## Competition
https://www.kaggle.com/c/sp-society-camera-model-identification

## Install dependencies
```
conda create -n tensorflow pip
source activate tensorflow
pip install tensorflow-gpu==1.5.0
conda install cudnn=7 opencv h5py
```

## Training
To train an Inception-ResNet-v2:
```
python trainer.py --model=keras_inception_resnet_v2
```

To visualize the training logs on Tensorboard run:
```
tensorboard --logdir=output
```
