
import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import keras
from keras.layers import Dense,Dropout,Conv2D,Input,MaxPool2D,Flatten,Activation, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D, Conv2D, Concatenate
from keras.models import Model
keras.backend.set_image_data_format('channels_last')

#import model from keras if using single level classification

#from keras.applications.resnet import ResNet50, ResNet101
#from keras_applications.resnet_v2 import ResNet50V2, ResNet101V2
#from keras_applications.mobilenet import MobileNet
#from keras_applications.mobilenet_v2 import MobileNetV2
#from keras_applications.resnext import ResNeXt50, ResNeXt101
#from keras_applications.densenet import DenseNet121, DenseNet169, DenseNet201

# import modified densenet hirarchical model if using hirarchical classification

from keras_densenet_modified import DenseNet201_hir

def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 #if backend.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def model_one_class(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):
    # for results of sota papers
    inputs = Input(input_shape)
    base_model= ResNet50(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNet101(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNet50V2(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNet101V2(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet121(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet169(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet201(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= MobileNet(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= MobileNetV2(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= ResNeXt50( input_tensor = inputs, include_top = False, weights = None,backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    #base_model= DenseNet121(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    

    x=  base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs=inputs, outputs= [x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model


def dense201_hirar(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):
    
    # for variant 1 in the paper

    inputs = Input(input_shape)
    base_model= DenseNet201_hir(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
   
    [x1,x2,x] = base_model.output

    x1 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class6_last')(x1)
    x1 = Activation('relu', name='relu_class6_last')(x1)                                                                                                                                                                                                                                                                    
    x1 = GlobalAveragePooling2D(name='GAvgPool_class6_last')(x1)
    x2 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class20_last')(x2)
    x2 = Activation('relu', name='relu_class20_last')(x2)
    x2 = GlobalAveragePooling2D(name='GAvgPool_class20_last')(x2)
    x = GlobalAveragePooling2D()(x)

    x1 = Dense(class_6, activation= 'softmax')(x1)
    x2 = Dense(class_20, activation= 'softmax')(x2)
    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs, [x1,x2,x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model

def dense201_hirar_6same20(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):
    
    # for variant 2 in the paper
    inputs = Input(input_shape)
    base_model= DenseNet201_hir(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)

    [null,x2,x] = base_model.output

    x1 = BatchNormalization(epsilon=1.001e-5, name = 'bn_class6_last')(x2)
    x1 = Activation('relu', name='relu_class6_last')(x1)
    x1 = GlobalAveragePooling2D(name='GAvgPool_class6_last')(x1)
    x2 = BatchNormalization(epsilon=1.001e-5, name = 'bn_class20_last')(x2)
    x2 = Activation('relu', name='relu_class20_last')(x2)
    x2 = GlobalAveragePooling2D(name='GAvgPool_class20_last')(x2)
    x = GlobalAveragePooling2D()(x)

    x1 = Dense(class_6, activation= 'softmax')(x1)

    x2 = Dense(class_20, activation= 'softmax')(x2)

    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs, [x1,x2,x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model


def dense201_hirar_new(
        input_shape = (224,224,3),
        class_6=6,
        class_20=20,
        class_82=82):

    # for variant 3 in the paper

    inputs = Input(input_shape)
    base_model= DenseNet201_hir(include_top=False, weights=None, input_tensor = inputs, backend = keras.backend , layers = keras.layers , models = keras.models , utils = keras.utils)
    
    [x1,x2,x] = base_model.output

    x1 = dense_block(x1, 32, name='denseblockClass6')


    x1 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class6_last')(x1)
    x1 = Activation('relu', name='relu_class6_last')(x1)
    x1 = GlobalAveragePooling2D(name='GAvgPool_class6_last')(x1)
    x2 = BatchNormalization( epsilon=1.001e-5, name = 'bn_class20_last')(x2)
    x2 = Activation('relu', name='relu_class20_last')(x2)
    x2 = GlobalAveragePooling2D(name='GAvgPool_class20_last')(x2)
    x = GlobalAveragePooling2D()(x)

    x1 = Dense(class_6, activation= 'softmax')(x1)

    x2 = Dense(class_20, activation= 'softmax')(x2)

    x = Dense(class_82, activation='softmax')(x)

    model = Model(inputs, [x1,x2,x])

    for layer in base_model.layers:
        layer.trainable = True
    
    return model
