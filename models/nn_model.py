
import os
from tensorflow import keras
from keras import callbacks

class NNModel():


    def __init__(self, X_train, X_test, y_train, y_test, metadata, model_name, random_state = 42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata
        self.random_state = random_state
        self.model_name = model_name
        self.epochs = 1000

        self.model_folder = self.model_name + '_' + self.metadata['problemname'].replace(' ', '_') + '_' + str(self.random_state)

        if 'model_checkpoints' not in os.listdir('.'):
            os.mkdir('./model_checkpoints')
        
        if self.model_folder not in os.listdir('./model_checkpoints'):
            os.mkdir('./model_checkpoints/' + self.model_folder)


        self.callbacks = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience= int(0.5*self.epochs),
                verbose= True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience= int(0.1*self.epochs),
                verbose= True,
                factor= 0.9
            ),
            callbacks.ModelCheckpoint(
                filepath = './model_checkpoints/' + self.model_folder + '/checkpoint.keras',
                monitor='val_loss',
                verbose=True,

            )
        ]


            


    def compile(self):
        pass

    

    def fit(self):
        self.history = None
        
        return self.history


    def training_proces(self):

        self.compile()
        self.fit()
