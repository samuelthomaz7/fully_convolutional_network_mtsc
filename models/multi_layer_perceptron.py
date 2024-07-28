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

        if len(self.metadata['class_values']) > 2:
            loss = losses.CategoricalCrossentropy()
        else:
            loss = losses.BinaryCrossentropy()

        self.complied_model = self.model.compile(
            optimizer= optimizers.Adam(learning_rate= 0.001),
            metrics= [
                keras.metrics.Accuracy(),
                keras.metrics.F1Score()
            ],
            loss= loss
        )
        
        return self.complied_model
    

    def fit(self):

        self.history = self.model.fit(
            x = self.X_train,
            y = self.y_train,
            validation_data = (self.X_test, self.y_test),
            epochs= self.epochs,
            callbacks=self.callbacks
        )


        return self.history
    
    def training_proces(self):
        return super().training_proces()

