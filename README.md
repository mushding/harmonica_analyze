Harmonica Analyzer 口琴音色辨識網站
===
## 0. Introduction
Learn harmonica is easier then other instruments.
Especially those who didn’t know about music theory,
still can learn harmonica easily.

For beginners, there are no sufficient resource on internet
to learn harmonica himself.
So we decide to build a website, a simple and convenient way, 
which can tell the bad part or something can improve.

We arrive at a conclusion that there are *three* problems
the beginners usually make mistake.

![](https://i.imgur.com/SQKC2Gc.png)

Normal is the correct sound

Double is play two at same time

Flat is the sound is not beautiful

## 1. Proposed Scheme

### process flow diagram

![](https://i.imgur.com/2AdRDFl.png)

![](https://i.imgur.com/1EGCQk7.png)

* The harmonica mistake dataset
    * record the normal harmonica sound
    * record  the mistake harmonica sound
        * flat
        * double
    * Cut with 0.5sec per segment
    * Move the segment with 0.1sec,then cut another.
    * With this method we can generate hundreds of data.
* Turn the data into MCFF

### Convolutional Neural Network Model

* CNN model (three Conv1D convolution layers)
    * Convolution layer1 with kernel size : 20
    * Max Pooling layer with pool size : 5
    * Convolution layer2 with kernel size : 20
    * Max Pooling layer with pool size : 5
    * Convolution layer3 with kernel size : 20
    * Max Pooling layer with pool size : 5
    * Flattening layer

* CNN model (two Conv2D convolution layers)
    * Convolution layer1 with kernel size : 5x5
    * Max Pooling layer with pool size : 2x2
    * Convolution layer2 with kernel size : 5x5
    * Max Pooling layer with pool size : 2x2
    * Flattening layer

![](https://i.imgur.com/8fN23SK.png)

## 2. Experiment Results

* Conv1D 
    * adv
        * only need 1 dimension that can speedup for data loading and model training.
    * dis
        * the accuracy is unsatisfactory.
* Conv2D
    * adv
        * the accuracy is much higher than conv1d .
    * dis
        * need for much time to loading ,the model is bigger than 1d.
* In this case we need higher accuracy 

![](https://i.imgur.com/qJ8W85P.png)

## 3. Usage
All the projects were built on website. So just click the website below the see the project.

https://www.haranalyzer.site/#/home

![](https://i.imgur.com/at5XuLu.png)



