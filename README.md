# A Neural Algorithm of Artistic Style

["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)
 (Gatys, et al. 2015) is the source to this project idea. The implementation of this content-and-style transfer network is a collaboration with [@ruggeri](https://github.com/ruggeri).


## Implementation

The goal of this project is to transfer the style of an artwork to the content of a photograph. We use the VGG recognition network and the paper's clever perspective on understanding the "style" of an artwork (similar to an image's "texture").

## Results

#### 1
![milan-style](/docs/result_milan.jpg)

**Figure 1:** Content is captured from the Duomo di Milano image. Styles from Cézanne and Monet are transferred with some success. I decide to experiment more with hyper-parameters to tune the model.

#### 2
![shrine-style](/docs/result_shrine.jpg)

**Figure 2:** Content: Itsukushima Shrine, Style: Cézanne. Learning rate: 10.0, Epochs: 3000. This takes 25 minutes to train on AWS EC2 instance-- performance is what I want to improve next.

#### 3
![starry-style](/docs/result_starry-night.jpg)

**Figure 3:** Content: Tubingen. Style: Van Gogh. I saved the image after every 100 epochs as the model trained, obtaining the learning process in action!
<p align="center">
  <img src="/docs/starry_tubingen_ab.gif">
</p>

## Future Directions

* ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155) (Johnson, et al. 2016)
* Speed up style transfer by training a network that generates the style transferred images. This will use a deep convolutional generator network (along with batch normalization and residual blocks).


## Reference
VGG16 Summary:
* Total params: 14,714,688
* Trainable params: 0
* Non-trainable params: 14,714,688

|Layer (type) |                Output Shape   |           Param # |
| --- | --- | --- |
|input_1 (InputLayer)  |       (None, 768, 1024, 3)  |    0   |
|block1_conv1 (Conv2D)  |      (None, 768, 1024, 64)  |   1792    |  
|block1_conv2 (Conv2D)    |    (None, 768, 1024, 64)  |   36928     |
|block1_pool (MaxPooling2D)  | (None, 384, 512, 64)    |  0         |
|block2_conv1 (Conv2D)   |     (None, 384, 512, 128)   |  73856     |
|block2_conv2 (Conv2D)   |     (None, 384, 512, 128)   |  147584    |
|block2_pool (MaxPooling2D)  | (None, 192, 256, 128)  |   0        |
|block3_conv1 (Conv2D)    |    (None, 192, 256, 256)  |   295168   |
|block3_conv2 (Conv2D)    |    (None, 192, 256, 256)   |  590080    |
|block3_conv3 (Conv2D)    |    (None, 192, 256, 256)  |   590080    |
|block3_pool (MaxPooling2D) |  (None, 96, 128, 256)  |    0         |
|block4_conv1 (Conv2D)   |     (None, 96, 128, 512)  |    1180160   |
|block4_conv2 (Conv2D)    |    (None, 96, 128, 512)   |   2359808   |
|block4_conv3 (Conv2D)    |    (None, 96, 128, 512)   |   2359808   |
|block4_pool (MaxPooling2D) |  (None, 48, 64, 512)    |   0         |
|block5_conv1 (Conv2D)     |   (None, 48, 64, 512)    |   2359808   |
|block5_conv2 (Conv2D)     |   (None, 48, 64, 512)    |   2359808   |
|block5_conv3 (Conv2D)    |    (None, 48, 64, 512)    |   2359808   |
|block5_pool (MaxPooling2D) |  (None, 24, 32, 512)    |   0         |
|global_average_pooling2d_1 |( (None, 512)            |   0         |
