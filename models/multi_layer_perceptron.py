from tensorflow import keras
from keras import layers, optimizers, losses, Model
from models.nn_model import NNModel



class MultiLayerPerceprtron(NNModel):

    def __init__(self, X_train, X_test, y_train, y_test, metadata, random_state=42):
        super().__init__(X_train, X_test, y_train, y_test, metadata, 'MultiLayerPerceprtron', random_state)


        inputs = layers.Input(shape= (X_train.shape[1], X_train.shape[2]))
        output = layers.Flatten() (inputs)
        output = layers.Dense(units= 500, activation= 'relu') (output)
        output = layers.Dropout(rate = 0.1) (output)
        output = layers.Dense(units= 500, activation= 'relu') (output)
        output = layers.Dropout(rate = 0.2) (output)
        output = layers.Dense(units= 500, activation= 'relu') (output)
        output = layers.Dropout(rate = 0.3) (output)
        output = layers.Dense(units= len(self.metadata['class_values']), activation= 'softmax') (output)

        self.model = Model(
            inputs, 
            output, 
            name = self.model_name
        )

    def compile(self):
        return super().compile()
    
    def fit(self):
        return super().fit()
    
    def training_process(self):
        return super().training_process()

