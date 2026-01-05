from sklearn.datasets import fetch_openml
import numpy as np

# as_frames=False gives us pure numpy array
mnist = fetch_openml('mnist_784', version=1 , as_frame=False)

X_raw = mnist.data
Y_raw = mnist.target

print(f'Shape of actual data : {X_raw.shape}')
print(f'Shape of the target values : {Y_raw.shape}')