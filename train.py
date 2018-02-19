from featurization_model import featurization_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import keras.backend as K
from utils import load_image, save_image
import numpy as np

print("Calculating content features")
content_im_data = load_image('./input/content_milan.jpg')
content_value, *_ = featurization_model.predict(
    np.expand_dims(content_im_data, axis=0)
)
print(content_value.shape)

print("Calculating style matrices")
style_im_data = load_image('./images/style_monet2.jpg')
_, *style_values = featurization_model.predict(
    np.expand_dims(style_im_data, axis=0)
)
value_shapes = [value.shape for value in style_values] #note
print(value_shapes)

target_values = [content_value, *style_values]
