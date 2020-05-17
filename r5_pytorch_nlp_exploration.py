#!/usr/bin/python
import numpy as np
import xgboost as xgb
import pandas as pd

from torchnlp.datasets import imdb_dataset

# Load the imdb training dataset
train = imdb_dataset(train=True)
train[0]  # RETURNS: {'text': 'For a movie that gets..', 'sentiment': 'pos'}
