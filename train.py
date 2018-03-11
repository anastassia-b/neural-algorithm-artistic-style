from featurization_model import featurization_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import keras.backend as K
from utils import load_image, save_image
import numpy as np

print("Calculating content features")
content_im_data = load_image('./input/content_mit.jpg')
content_value, *_ = featurization_model.predict(
    np.expand_dims(content_im_data, axis=0)
)
print(content_value.shape)

print("Calculating style matrices")
style_im_data = load_image('./input/style_klimt_kiss.jpg')
_, *style_values = featurization_model.predict(
    np.expand_dims(style_im_data, axis=0)
)
value_shapes = [value.shape for value in style_values] #note
print(value_shapes)

target_values = [content_value, *style_values]

input_tensor = Input(shape=(1,))
image_layer = Dense((768 * 1024 * 3), activation='linear', use_bias=False)

image_tensor = image_layer(input_tensor)
reshaped_image = Reshape((768, 1024, 3))(image_tensor)

content_tensor, *style_tensors = featurization_model(reshaped_image)
feature_tensors = [content_tensor, *style_tensors]

training_model = Model(inputs=input_tensor, outputs=feature_tensor)
training_model.summary()

def save_int_image(epoch_idx, logs):
    if epoch_idx % 100 == 99:
        flattened_image_data = image_layer.get_weights()[0]
        image_data = np.reshape(flattened_image_data, (768, 1024, 3))
        save_image(f'./images/results{epoch_idx:04}.jpg', image_data)

optimizer = Adam(lr=10.0) #change
training_model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    loss_weights=[2.5, *([1]*5)]
)

training_model.fit(
    np.ones([1, 1]),
    target_values,
    batch_size=1,
    epochs=3000,
    callbacks=[LambdaCallback(on_epoch_end=save_int_image)]
)
