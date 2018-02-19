from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Lambda
import keras.backend as K
import numpy as np

# sample images are 768x1024, coco dataset images are 256x256.
vgg_model = VGG16(include_top=False, input_shape=[768, 1024, 3], pooling="avg")

for layer in vgg_model.layers:
    # freeze layers
    layer.trainable = False

# vgg_model.summary()

# last layer we need
content_layer = vgg_model.get_layer(name='block5_conv2')
input_tensor = vgg_model.input
content_tensor = content_layer.output

# print(content_value)
# print(content_value.shape)

output1 = vgg_model.get_layer(name='block1_conv1').output
output2 = vgg_model.get_layer(name='block2_conv1').output
output3 = vgg_model.get_layer(name='block3_conv1').output
output4 = vgg_model.get_layer(name='block4_conv1').output
output5 = vgg_model.get_layer(name='block5_conv1').output

def build_style_matrix(vgg_feature_tensor):
    perm_feature_tensor = K.permute_dimensions(
        vgg_feature_tensor, (0, 3, 1, 2)
    )
    # print(vgg_feature_tensor.shape)
    num_channels = perm_feature_tensor.shape[1].value
    height = perm_feature_tensor.shape[2].value
    width = perm_feature_tensor.shape[3].value
    flattened_feature_tensor = K.reshape(
        perm_feature_tensor, (-1, num_channels, height * width))

    transposed_feature_tensor = K.permute_dimensions(
        flattened_feature_tensor, (0, 2, 1))

    style_matrix = (K.batch_dot
        (flattened_feature_tensor, transposed_feature_tensor) /
        (num_channels * height * width))

    return style_matrix

# keras layer is like a resuable component. layer turns input into output.

style_layer = Lambda(build_style_matrix)
style_tensor1 = style_layer(output1)
style_tensor2 = style_layer(output2)
style_tensor3 = style_layer(output3)
style_tensor4 = style_layer(output4)
style_tensor5 = style_layer(output5)

featurization_model = Model(inputs=input_tensor, outputs=[
    content_tensor,
    style_tensor1,
    style_tensor2,
    style_tensor3,
    style_tensor4,
    style_tensor5,
    ])

# image_data = load_image('./images/tubingen1024.jpg')
