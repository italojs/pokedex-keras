from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Add

class Resnet50():
    def __identity_block(self, input_tensor, kernel_size, filters, stage, block):
        bn_axis = 3
        
        conv_base_name = 'res' + str(stage) + block + '_branch'
        bn_base_name = 'bn' + str(stage) + block + '_branch'

        arch = Conv2D(filters[0], kernel_size=(1, 1), name=conv_base_name + '2a')(input_tensor)
        arch = BatchNormalization(axis=bn_axis, name=bn_base_name + '2a')(arch)
        arch = Activation('relu')(arch)

        arch = Conv2D(filters[1], kernel_size, padding='same', name=conv_base_name + '2b')(arch)
        arch = BatchNormalization(axis=bn_axis, name=bn_base_name + '2b')(arch)
        arch = Activation('relu')(arch)

        arch = Conv2D(filters[2], (1, 1), name=conv_base_name + '2c')(arch)
        arch = BatchNormalization(axis=bn_axis, name=bn_base_name + '2c')(arch)

        arch = Add()([arch, input_tensor])
        arch = Activation('relu')(arch)
        
        return arch

    def __convolutional_block(self, input_tensor, kernel_size, stride, filters, stage, block):
        bn_axis = 3
        
        conv_base_name = 'res' + str(stage) + block + '_branch'
        bn_base_name = 'bn' + str(stage) + block + '_branch'
        
        arch = Conv2D(filters[0], kernel_size=(1, 1), strides=(stride, stride), name=conv_base_name+'2a')(input_tensor)
        arch = BatchNormalization(axis=bn_axis, name=bn_base_name+'2a')(arch)
        arch = Activation('relu')(arch)
        
        arch = Conv2D(filters[1], kernel_size=(kernel_size, kernel_size), padding='same', name=conv_base_name+'2b')(arch)
        arch = BatchNormalization(axis=bn_axis, name=bn_base_name+'2b')(arch)
        arch = Activation('relu')(arch)
        
        arch = Conv2D(filters[2], kernel_size=(1, 1), name=conv_base_name+'2c')(arch)
        arch = BatchNormalization(axis=bn_axis, name=bn_base_name+'2c')(arch)
        
        arch_shortcut = Conv2D(filters[2], kernel_size=(1, 1), strides=(stride, stride), name=conv_base_name+'1')(input_tensor)
        arch_shortcut = BatchNormalization(axis=bn_axis, name=bn_base_name+'1')(x_shortcut)
        
        arch = Add()([arch, arch_shortcut])
        arch = Activation('relu')(arch)
        
        return arch

    def add_stage(self, arch, kernel_size, stride, filters, stage, blocks):
        arch = self.__convolutional_block(arch, kernel_size=kernel_size, stride=stride, filters=filters, stage=stage, block='0')
        for block in range(blocks):
            arch = self.__identity_block(arch, kernel_size=kernel_size, filters=filters, stage=stage, block=str(block+1))
        return arch
        
    def build(self, classes):
        model_input = Input((224, 224, 3), name='input')

        arch = ZeroPadding2D(padding=(3, 3), name='pad_conv1')(model_input)
        
        arch = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(arch)
        arch = BatchNormalization(axis=3, name='bn_conv1')(arch)
        arch = Activation('relu')(arch)
        arch = ZeroPadding2D(padding=(1, 1), name='pad_pool1')(arch)
        arch = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='max_pool2d_1')(arch)
        
        arch = self.add_stage(arch, kernel_size=3, stride=1, filters=[64, 64, 256], stage=2, blocks=2)
        arch = self.add_stage(arch, kernel_size=3, stride=2, filters=[128, 128, 512], stage=3, blocks=3)
        arch = self.add_stage(arch, kernel_size=3, stride=2, filters=[256, 256, 1024], stage=4, blocks=6)
        arch = self.add_stage(arch, kernel_size=3, stride=2, filters=[512, 512, 2048], stage=5, blocks=3)

        arch = GlobalAveragePooling2D(name='average_pool')(arch)
        
        arch = Dense(units=classes, activation='softmax', name='fc' + str(classes))(arch)
        
        model = Model(inputs=model_input, outputs=arch, name='model_resNet50')
        
        return model