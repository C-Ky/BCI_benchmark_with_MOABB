from .model import KerasModel
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Conv2D, Dropout, BatchNormalization, \
                         Reshape, Activation, Flatten, AveragePooling2D, Conv3D, MaxPool2D


class Deep_CNN(KerasModel):

    def create_model(self, nb_classes, augmented_data=False, print_summary=False, downsampled=False, loss='categorical_crossentropy', opt='adam', met=['accuracy']):
        CLASS_COUNT = nb_classes

        model = Sequential()
        if augmented_data and downsampled:
            input_shape = (3, 1280, 1)
            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))
            model.add(Reshape(target_shape=(3, 1271, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(1271, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(138, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(43, 100, 1)))
            model.add(Dropout(0.5))

        if augmented_data and not downsampled:
            input_shape = (3, 1024, 1)
            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))

            model.add(Reshape(target_shape=(3, 1015, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(1015, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())

            model.add(Reshape(target_shape=(109, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(33, 100, 1)))
            model.add(Dropout(0.5))

        if not augmented_data and not downsampled:
            sp = 1001 #2560
            chans = 22 #3
            filters = 25
            input_shape = (chans, sp, 1)

            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=filters, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))

            model.add(Reshape(target_shape=(chans, sp-(10-1), filters, 1)))
            model.add(Conv3D(filters=filters, kernel_size=(chans, 1, filters),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(sp-(10-1), filters, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            filters2 = 50
            # Conv Pool Block 2
            model.add(Conv2D(filters=filters2, kernel_size=(10, filters)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())

            n1 = 107 #280=2560/3/3 - 3
            model.add(Reshape(target_shape=(n1, filters2, 1)))
            model.add(Dropout(0.5))

            filters3 = 100
            # Conv Pool Block 3
            model.add(Conv2D(filters=filters3, kernel_size=(10, filters2)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())

            n2 = 32 #90=280*3 - 3
            model.add(Reshape(target_shape=(n2, filters3, 1)))
            model.add(Dropout(0.5))

        if not augmented_data and downsampled:
            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))
            model.add(Reshape(target_shape=(3, 1271, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(1271, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(138, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(43, 100, 1)))
            model.add(Dropout(0.5))

        # Conv Pool Block 4
        model.add(Conv2D(filters=200, kernel_size=(10, 100)))

        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
        model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))

        # Softmax for classification
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
