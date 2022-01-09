# Music-Style-Transfer

This is music style transfer with tensorflow 2.0. One of the architectures is based on [1] and [2], but we implement it from scratch. 

Original:

[moonlight sonata](https://www.youtube.com/watch?v=xyh9PXa4gFI&list=PLeProSFvyWw68b5xNHqyqEdWtQu2TdpVw&index=1)

ResNet (Jazz):

[moonlight sonata](https://www.youtube.com/watch?v=7p_HqSkUg3c&list=PLeProSFvyWw68b5xNHqyqEdWtQu2TdpVw&index=2&t=1s)

LSTM V2 (Jazz):

[moonlight sonata](https://www.youtube.com/watch?v=xvBukVHvGYs&list=PLeProSFvyWw68b5xNHqyqEdWtQu2TdpVw&index=3)

## Our Environment
- Operating System:
  - Archlinux/Windows10
- CUDA:
  - CUDA V11.5.50 
- Nvidia driver:
  - 495.44
- Python:
  - python 3.7.3
- Tensorflow:
  - tensorflow-2.6.0

## Downloading the Preprocessd Dataset
[Dataset](https://drive.google.com/file/d/1EfS1p3_B6X-ibbZCNwAuyMOpFIHgpE45/view?usp=sharing)
## Downloading the Pretrain Model for Classifier
[Model](https://drive.google.com/drive/folders/1ErFniyIiK7ov3RFbWh0V-o8yxx73a1jG?usp=sharing)

## Train MusicStyle Transform with CNN 
```
cd ./MusicStyleResNet
```
## Train MusicStyle Transform with CNN and evaluation
```
cd ./MusicStyleResNet
```
```
python main.py --load_classifier ../Classifier/checkpoints/Classifier/{timestamp}/{checkpoint}.hdf5
```
## Continue Training
```
cd ./MusicStyleResNet
```
```
python main.py --load_checkpoint ./checkpoints/ResNet/{timestamp}/{000-0.250-1}
```
## Test
```
cd ./MusicStyleResNet
```
```
python main.py --load_classifier ../Classifier/checkpoints/Classifier/{timestamp}/{checkpoint}.hdf5 --load_checkpoint ./checkpoints/ResNet/{timestamp}/{000-0.250-1} --phase test
```

## Generate Sample Output
```
cd ./MusicStyle{model}
```
```
python main.py --load_checkpoint ./checkpoints/ResNet/{timestamp}/{000-0.250-1} --phase sample --sample-midi {midi filepath}
```

# Train Genre Classifier 
```
cd ./Classifier
```
```
python main.py
```
# Test Genre Classifier
```
cd ./MusicStyleResNet
```
```
python main.py --phase test --load_classifier ./checkpoints/Classifier/{timestamp}/{checkpoint}.hdf5
```

## References

1. [sumuzhao/CycleGAN-Music-Style-Transfer](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer)

2. [Gino Brunner, Yuyi Wang, Roger Wattenhofer, and SumuZhao, "Symbolic music genre transfer with cyclegan" ICTAI](https://arxiv.org/abs/1809.07575)
