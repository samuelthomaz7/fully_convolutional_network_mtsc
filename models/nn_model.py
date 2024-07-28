
from tensorflow import keras
from keras import callbacks

class NNModel():


    def __init__(self, X_train, X_test, y_train, y_test, metadata, random_state = 42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata
        self.random_state = random_state
        self.model_name = 'GenericModel'
        self.epochs = 1000


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
