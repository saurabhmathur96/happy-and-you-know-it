from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Convolution2D, Dense, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D, merge
import numpy as np

class FCN12(object):
    def __init__(self, input_shape, n_filter=4, n_class=7):
        model = Sequential()
        model.add(Convolution2D(n_filter, 3, 3, border_mode="same", input_shape=input_shape+(1,), name="conv2d_1"))
        model.add(BatchNormalization(axis=1, name="batchnorm_1"))
        model.add(Activation("relu", name="relu_1"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same", name="maxpool2d_1"))

        model.add(Convolution2D(n_filter, 3, 3, border_mode="same", input_shape=input_shape, name="conv2d_2"))
        model.add(BatchNormalization(axis=1, name="batchnorm_2"))
        model.add(Activation("relu", name="relu_2"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same", name="maxpool2d_2"))

        model.add(Convolution2D(n_class, 3, 3, border_mode="same", name="conv2d_3"))
        model.add(BatchNormalization(axis=1, name="batchnorm_3"))
        model.add(GlobalAveragePooling2D(name="global_average_pool_2d_3"))
        model.add(Activation("softmax"))
        model.compile(optimizer=Adam(.1), loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model
    
    def fit_generator(self, train_generator, test_generator=None, samples_per_epoch=28709, nb_val_samples=3589, nb_epoch=1):
        self.model.fit_generator(train_generator, validation_data=test_generator, samples_per_epoch=samples_per_epoch, nb_val_samples=nb_val_samples, nb_epoch=nb_epoch)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save_weights(path)
        return path
    
    def load(self, path):
        self.model.load_weights(path)


def conv_bn(inputs, n_filter, n_row, r_col, strides=(1, 1)):
        c = Convolution2D(n_filter, n_row, r_col, subsample=strides, border_mode="same", init="he_normal")(inputs)
        b = BatchNormalization()(c)
        
        return b

def conv_bn_relu(inputs, n_filter, n_row, r_col, strides=(1, 1)):
        r = conv_bn(inputs, n_filter, n_row, r_col, strides)
        
        return Activation("relu")(r)

def resnet_basic(inputs, n_filter):
        c1 = conv_bn_relu(inputs, n_filter, 3, 3)
        c2 = conv_bn(c1, n_filter, 3, 3)
        p = merge([c2, inputs], mode="sum")
        
        return Activation("relu")(p)

def resnet_basic_inc(inputs, n_filter, strides=(2, 2)):
        c1 = conv_bn_relu(inputs, n_filter, 3, 3, strides)
        c2 = conv_bn(c1, n_filter, 3, 3)
        s = conv_bn(inputs, n_filter, 1, 1, strides)
        p = merge([c2, s], mode="sum")
        
        return Activation("relu")(p)

def resnet_basic_stack(inputs, n_filter, n_layer):
        layer = inputs
        for _ in range(n_layer):
            layer = resnet_basic(layer, n_filter)
        return layer

class ResNet20(object):
    def __init__(self, input_shape, n_filter=None, n_class=7):
        if n_filter is None:
            n_filter = (16, 32, 64)
        n_stack_layer = 3 

        input_layer = Input(shape=input_shape+(1,))
        block_1 = conv_bn_relu(input_layer, n_filter[0], 3, 3)
        stack_1 = resnet_basic_stack(block_1, n_filter[0], n_stack_layer)

        block_2 = resnet_basic_inc(stack_1, n_filter[1])
        stack_2 = resnet_basic_stack(block_2, n_filter[1], n_stack_layer-1)

        block_3 = resnet_basic_inc(stack_2, n_filter[2])
        stack_3 = resnet_basic_stack(block_3, n_filter[2], n_stack_layer-1)

        # Global average pooling and output
        pool = AveragePooling2D(pool_size=(8, 8))(stack_3)
        flat = Flatten()(pool)
        output_layer = Dense(n_class)(flat)
        output_layer_batch_norm = BatchNormalization()(output_layer)
        output_activation = Activation("softmax")(output_layer_batch_norm)

        model = Model(input=input_layer, output=output_activation)

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model
    
    def fit_generator(self, train_generator, test_generator=None, samples_per_epoch=28709, nb_val_samples=3589, nb_epoch=1):
        self.model.fit_generator(train_generator, validation_data=test_generator, samples_per_epoch=samples_per_epoch, nb_val_samples=nb_val_samples, nb_epoch=nb_epoch)
    
    def predict(self, X):
        X = np.array(X).reshape(-1, 48, 48, 1)
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save_weights(path)
        return path
    
    def load(self, path):
        self.model.load_weights(path)
    