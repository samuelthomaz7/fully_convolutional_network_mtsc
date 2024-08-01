import os
import random
import tensorflow as tf
import numpy as np

from input.reading_datasets import read_dataset_from_file
from preprocessing.get_dummies_labels import GetDummiesLabels
from preprocessing.train_test_split_module import TrainTestSplit

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def training_nn_for_seeds(used_model, datasets = [], seeds = []):
    for dataset in datasets:
        for random_state in seeds:
            used_dataset = read_dataset_from_file(dataset_name = dataset)
            X, y, metadata = used_dataset

            get_dummies_object = GetDummiesLabels(
                X_raw= X,
                y_raw= y,
                metadata= metadata
            )

            X, y = get_dummies_object.transform()

            train_test_object = TrainTestSplit(
                X_raw= X,
                y_raw= y,
                metadata= metadata,
                random_state = random_state
            )

            X_train, X_test, y_train, y_test = train_test_object.transform()

            model = used_model(
                X_train=X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                metadata = metadata,
                random_state = random_state
            )

            if len(os.listdir('./model_checkpoints/' + model.model_folder)) != 0 :
                pass
            else:
                model.training_process()