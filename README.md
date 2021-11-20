# Music-Style-Transfer

This is music style transfer with tensorflow 2.0. The one of the architectures is based on [1], but we implement it from scratch. 

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
[Dataset](https://drive.google.com/drive/u/2/folders/1BhQ58MjpiCXUqKqYfO4cEN1vw9FVZoEU)
## Downloading the Pretrain Model for Classifier
[Model](https://drive.google.com/drive/u/2/folders/14JcMWwWwcgDP3kXNV7HsmVgcKLiHBwJI)

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
