from tensorflow import keras
from keras import layers, optimizers, losses, Model
from models.nn_model import NNModel



class FullyConvolutionalNetwork(NNModel):

    def __init__(self, X_train, X_test, y_train, y_test, metadata, random_state=42):
        super().__init__(X_train, X_test, y_train, y_test, metadata, 'FullyConvolutionalNetwork', random_state)

        if len(self.metadata['class_values']) > 2:
            last_layer_activation = 'softmax'
        else:
            last_layer_activation = 'sigmoid'


        # self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[2], self.X_train.shape[1])
        # self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2], self.X_test.shape[1])


        inputs = layers.Input(shape= (self.X_train.shape[1], self.X_train.shape[2]))

        output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (inputs) 
        output = keras.layers.Activation('relu') (output)
        output = layers.BatchNormalization() (output) 

        output = layers.Conv1D(filters=256, padding="same", strides=1, kernel_size= (5, ), data_format='channels_first') (output)
        output = keras.layers.Activation('relu') (output)
        output = layers.BatchNormalization() (output) 

        output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (3, ), data_format='channels_first') (output)
        output = keras.layers.Activation('relu') (output)
        output = layers.BatchNormalization() (output) 

        output = layers.GlobalAveragePooling1D() (output)

        # output = layers.Dense(units= 128, activation='relu') (output)
        # output = layers.Dense(units= 128, activation='relu') (output)
        # output = layers.Dense(units= 128, activation='relu') (output)

        output = layers.Dense(units= len(self.metadata['class_values']), activation=last_layer_activation) (output)

        self.model = Model(
            inputs,
            output,
            name = self.model_name
        )

        print(self.model.summary())

    def compile(self):
        return super().compile()
    

    def fit(self):
        return super().fit()
    
    def training_process(self):
        return super().training_process()

