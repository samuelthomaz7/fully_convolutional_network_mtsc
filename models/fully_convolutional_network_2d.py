from tensorflow import keras
from keras import layers, optimizers, losses, Model
from models.nn_model import NNModel



class FullyConvolutionalNetwork2D(NNModel):

    def __init__(self, X_train, X_test, y_train, y_test, metadata, random_state=42):
        super().__init__(X_train, X_test, y_train, y_test, metadata, 'FullyConvolutionalNetwork2D', random_state)

        if len(self.metadata['class_values']) > 2:
            last_layer_activation = 'softmax'
        else:
            last_layer_activation = 'sigmoid'

        
        self.X_train = self.X_train.reshape(self.X_train.shape + (1,))
        self.X_test = self.X_test.reshape(self.X_test.shape + (1,))


        inputs = keras.layers.Input(shape= (self.X_train.shape[1:]))

        conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(inputs)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)
        
        conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        
        conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
        full = keras.layers.GlobalAveragePooling2D()(conv3)
        output = keras.layers.Dense(len(self.metadata['class_values']), activation='softmax')(full)

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

