from .model import KerasModel
import keras
from keras.backend import square
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Conv2D, Dropout, BatchNormalization, \
                         Reshape, Activation, Flatten, AveragePooling2D, Conv3D


class Shallow_CNN(KerasModel):

    def create_model(self, nb_classes, augmented_data=False, print_summary=False, downsampled=False, loss='categorical_crossentropy', opt='adam', met=['accuracy']):

        CLASS_COUNT = nb_classes
        model = Sequential()
        # augmented_data = False
        # print_summary=False

        if augmented_data and downsampled:
            chans = 22 #3
            sp = 1001 #512
            f = 40
            ks = 25
            # Conv Block 1
            model.add(Conv2D(input_shape=(chans, sp, 1), filters=f, kernel_size=(1, ks), strides=(1, 1),
                             padding='valid', activation=None))
            model.add(Reshape(target_shape=(chans, sp-(ks-1), f, 1)))
            model.add(Dropout(0.5))

            # Conv Block 2
            model.add(Conv3D(filters=f, kernel_size=(chans, 1, f), padding='valid',
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))  # keras.backend.square: custom squaring activation function
            model.add(Flatten())
            model.add(Reshape(target_shape=(sp-(ks-1), f, 1)))
            model.add(Dropout(0.5))
            # Pooling
            model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1), data_format='channels_last'))
            model.add(Activation(activation='elu'))  # keras.backend.log: custom log function
            
        if augmented_data and not downsampled:
            chans = 22 #3
            sp = 1001 #1024
            ks = 25
            f=40
            # Conv Block 1
            model.add(Conv2D(input_shape=(chans, sp, 1), filters=f, kernel_size=(1, ks), strides=(1, 1),
                             padding='valid', activation=None))
            model.add(Reshape(target_shape=(chans, sp-(ks-1), f, 1)))
            model.add(Dropout(0.5))

            # Conv Block 2
            model.add(Conv3D(filters=f, kernel_size=(chans, 1, f), padding='valid',
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(keras.backend.square))  # keras.backend.square: custom squaring activation function
            model.add(Flatten())
            model.add(Reshape(target_shape=(sp-(ks-1), f, 1)))
            model.add(Dropout(0.5))
            # Pooling
            model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1), data_format='channels_last'))
            model.add(Activation(keras.backend.log))  # keras.backend.log: custom log function


        else:
            chans = 22
            sp = 1001
            f = 40
            ks = 25
            # Conv Block 1
            model.add(Conv2D(input_shape=(chans, sp, 1), filters=f, kernel_size=(1, ks), strides=(1, 1),
                             padding='valid', activation=None))
            model.add(Reshape(target_shape=(chans, sp-(ks-1), f, 1)))
            model.add(Dropout(0.5))

            # Conv Block 2
            model.add(Conv3D(filters=f, kernel_size=(chans, 1, f), padding='valid',
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))  # keras.backend.square, custom squaring activation function
            model.add(Flatten())
            model.add(Reshape(target_shape=(sp-(ks-1), f, 1)))
            model.add(Dropout(0.5))

            # Pooling
            model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1), data_format='channels_last'))
            model.add(Activation(activation='elu'))  # keras.backend.log, custom log function

        # Classification
        model.add(Flatten())
        model.add(Dense(CLASS_COUNT))
        model.add(Activation('softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=met)

        # assign and return
        self.model = model
        return model

