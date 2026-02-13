# import the actual dataset
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw = mnist.data
Y_raw = mnist.target


