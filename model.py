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
        f1, f2, f3 = filters
        bn_axis = 3
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(f1, kernel_size=(1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(f2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(f3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        
        return x

    def __convolutional_block(self, input_tensor, kernel_size, stride, filters, stage, block):
        f1, f2, f3 = filters
        bn_axis = 3
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        x = Conv2D(f1, kernel_size=(1, 1), strides=(stride, stride), name=conv_name_base+'2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(f2, kernel_size=(kernel_size, kernel_size), padding='same', name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(f3, kernel_size=(1, 1), name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base+'2c')(x)
        
        x_shortcut = Conv2D(f3, kernel_size=(1, 1), strides=(stride, stride), name=conv_name_base+'1')(input_tensor)
        x_shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base+'1')(x_shortcut)
        
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)
        
        return x

    def add_stage(self, x, kernel_size, stride, filters, stage, blocks):
        x = self.__convolutional_block(x, kernel_size=kernel_size, stride=stride, filters=filters, stage=stage, block='0')
        for block in blocks:
            x = self.__identity_block(x, kernel_size=kernel_size, filters=filters, stage=stage, block=str(block+1))
        return x
        
    def build(self, classes):
        x_input = Input((224, 224, 3), name='input_1')

        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x_input)
        
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='max_pooling2d_1')(x)
        
        x = self.add_stage(x, kernel_size=3, stride=1, filters=[64, 64, 256], stage=2, blocks=2)
        x = self.add_stage(x, kernel_size=3, stride=2, filters=[128, 128, 512], stage=3, blocks=3)
        x = self.add_stage(x, kernel_size=3, stride=2, filters=[256, 256, 1024], stage=4, blocks=6)
        x = self.add_stage(x, kernel_size=3, stride=2, filters=[256, 256, 1024], stage=4, blocks=6)
        x = self.add_stage(x, kernel_size=3, stride=2, filters=[512, 512, 2048], stage=5, blocks=3)

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        
        x = Dense(units=classes, activation='softmax', name='fc' + str(classes))(x)
        
        model = Model(inputs=x_input, outputs=x, name='myResNet50')
        
        return model