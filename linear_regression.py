# !/usr/local/bin/python3.12
"""Machine learning linear regression test code."""
# pylint: disable=unused-import
from __future__ import absolute_import, division, print_function, unicode_literals

import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context   # pylint: disable=protected-access
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
