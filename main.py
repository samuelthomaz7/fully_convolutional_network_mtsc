import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras
# from keras import layers, optimizers, losses, Model
# from aeon import datasets
# import os
# from tqdm import tqdm
# import pickle
# from input.reading_datasets import get_all_datasets, read_dataset_from_file
# import seaborn as sns
# from models.fully_convolutional_network_2d import FullyConvolutionalNetwork2D
# from preprocessing.get_dummies_labels import GetDummiesLabels
# from preprocessing.train_test_split_module import TrainTestSplit
# from models.multi_layer_perceptron import MultiLayerPerceprtron

print(tf.config.list_physical_devices('GPU'))


all_datasets = [
        "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions",
        "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms",
        "Epilepsy", "EthanolConcentration", "ERing", "FaceDetection",
        "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "Libras", "LSST", "InsectWingbeat", "MotorImagery",
        "NATOPS", "PenDigits", "PEMS-SF", "Phoneme", "RacketSports",
        "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
        "StandWalkJump", "UWaveGestureLibrary"
    ]

all_datasets.remove('InsectWingbeat')


from models.fully_convolutional_network import FullyConvolutionalNetwork
from utils_file import training_nn_for_seeds

# training_nn_for_seeds(
#     datasets= all_datasets, # type: ignore
#     seeds= list(range(1, 11)),
#     used_model = FullyConvolutionalNetwork
# )

# training_nn_for_seeds(
#     datasets= all_datasets, # type: ignore
#     seeds= list(range(1, 11)),
#     used_model = MultiLayerPerceprtron
# )

# training_nn_for_seeds(
#     datasets= ['AtrialFibrillation'], # type: ignore
#     seeds= list(range(1, 11)),
#     used_model = FullyConvolutionalNetwork
# )

# training_nn_for_seeds(
#     datasets= ['ArticularyWordRecognition'], # type: ignore
#     seeds= list(range(1, 2)),
#     used_model = FullyConvolutionalNetwork2D
# )

training_nn_for_seeds(
    datasets= ['ArticularyWordRecognition'], # type: ignore
    seeds= list(range(1, 2)),
    used_model = FullyConvolutionalNetwork
)