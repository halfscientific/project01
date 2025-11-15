import pandas as pd
import numpy as np

import os
import sys
from dataclasses import dataclass
import pickle
from src.logger import logging
from src.exception import CustomException


def save_object_pkl(file_path, obj):
    try:
        logging.info("Entered the save_object method of utils")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(
            "Object saved successfully, exited save_object method of utils")
    except Exception as e:
        raise CustomException(e, sys)


def load_object_pkl(file_path):
    try:
        logging.info("ENtered the load_object method of utils")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(
            "Pickled object loaded successfully, exited the load_object method of utils")

        return obj
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(file_path, array):
    try:
        logging.info("Entered the save_numpy_array_data method of utils")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logging.info(
            "Numpy array saved successfully, exited save_numpy_array_data method of utils")

    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path):
    try:
        logging.info("Entered the load_numpy_array_data method of utils")
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)
        logging.info(
            "Numpy array loaded successfully, exited load_numpy_array_data method of utils")
        return array
    except Exception as e:
        raise CustomException(e, sys)
