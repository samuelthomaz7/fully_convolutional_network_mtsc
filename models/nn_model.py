
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, Model, callbacks
from utils_file import set_seeds


class NNModel():


    def __init__(self, X_train, X_test, y_train, y_test, metadata, model_name, random_state = 42):
        
        set_seeds(seed=random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata
        self.random_state = random_state
        self.model_name = model_name
        self.epochs = 10000
        self.num_classes = self.metadata['class_values']            

        

        self.model_folder = self.model_name + '_' + self.metadata['problemname'].replace(' ', '_') + '_' + str(self.random_state)

        if 'model_checkpoints' not in os.listdir('.'):
            os.mkdir('./model_checkpoints')
        
        if self.model_folder not in os.listdir('./model_checkpoints'):
            os.mkdir('./model_checkpoints/' + self.model_folder)


        self.callbacks = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience= int(0.5*self.epochs),
                verbose= False
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience= int(0.1*self.epochs),
                verbose= False,
                factor= 0.9
            ),
            callbacks.ModelCheckpoint(
                filepath = './model_checkpoints/' + self.model_folder + '/checkpoint.keras',
                monitor='val_loss',
                verbose=False,


            )
        ]


            


    
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
            callbacks=self.callbacks,
            verbose = False
        )


        return self.history


    def training_process(self):

        self.compile()
        self.fit()
