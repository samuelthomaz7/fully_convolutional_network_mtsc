from tensorflow import keras
from keras import layers, optimizers, losses, Model
from models.nn_model import NNModel



class ResNet(NNModel):

    def __init__(self, X_train, X_test, y_train, y_test, metadata, random_state=42):
        super().__init__(X_train, X_test, y_train, y_test, metadata, 'ResNet', random_state)

        if len(self.metadata['class_values']) > 2:
            last_layer_activation = 'softmax'
        else:
            last_layer_activation = 'sigmoid'



        # --------------------------- WIP ------------------------------------------

        # inputs = layers.Input(shape= (self.X_train.shape[1], self.X_train.shape[2]))

        # output = layers.Conv1D(filters=64, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (inputs) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 
        # output = layers.Conv1D(filters=64, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 
        # output = layers.Conv1D(filters=64, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 

        # output_1st_block = layers.add([inputs, output])

        # output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output_1st_block) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 
        # output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 
        # output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 

        # output_2nd_block = layers.add([inputs, output_1st_block])


        # output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output_2nd_block) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 
        # output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 
        # output = layers.Conv1D(filters=128, padding="same", strides=1, kernel_size= (8, ), data_format='channels_first') (output) 
        # output = keras.layers.Activation('relu') (output)
        # output = layers.BatchNormalization() (output) 

        # output_3rd_block = layers.add([inputs, output_2nd_block])

        output = layers.GlobalAveragePooling1D() (output_3rd_block)
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

