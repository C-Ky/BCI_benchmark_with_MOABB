from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM as _LSTM

class LSTM(KerasModel):

    def create_model(self, nb_classes, input_shape, num_hidden_neurons=128,
                      num_layers=1, dropout=0.2, recurrent_dropout=0.2,
                      print_summary=False, loss='categorical_crossentropy', opt='adam', met=['accuracy']):

        model = Sequential()
        if num_layers > 1:
            for i in range(1, num_layers, 1):
                model.add(_LSTM(num_hidden_neurons, input_shape=input_shape,
                               return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
            model.add(_LSTM(num_hidden_neurons))
        else:
            model.add(_LSTM(num_hidden_neurons, input_shape=input_shape, dropout=dropout,
                           recurrent_dropout=recurrent_dropout))
        model.add(Dense(nb_classes, activation='softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=met)

        # assign and return
        self.model = model
        return model


